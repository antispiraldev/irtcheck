#!/usr/bin/env python3
"""Block the git operations that let two agents corrupt each other's work.

irtcheck is built by several agents at once — one brief per worktree, per
branch, per PR (see CLAUDE.md and docs/build-plan.html). Three rules, all aimed
at races rather than at mistakes:

1. Mutating git in the *shared main checkout* is denied. Agents work in
   `.claude/worktrees/<name>`; when they don't, one agent's `checkout` or
   `reset` silently rewrites the tree another is reading. This rule is
   inherited from the AudioBookLib repo, where on 2026-08-04 the main checkout
   was found with HEAD at master but its index and working tree holding the
   tree of a commit ~30 behind, every file staged as a deletion — a
   `git commit` there would have landed a 10k-line deletion on master.

   One carve-out: a repo whose HEAD is unborn. `git worktree add` refuses to
   run before the first commit, so the remedy this rule points at does not
   exist yet, and there is no history for another agent to be reading. The
   bootstrap commit is allowed; every write after it is denied normally. See
   has_commits().

2. Force-pushing, and pushing to master, are denied everywhere. master is the
   branch every brief branches from and merges back into; a direct push to it
   skips CI, and CI is where this repo's merge-time race check lives (the
   `contracts` job — two individually-green branches can both bump
   SCHEMA_VERSION or collide on a CLI flag).

3. `gh pr merge` is denied when the PR is behind its base branch, or when its
   checks are not green. This is the client-side stand-in for branch
   protection's "require branches to be up to date before merging", which a
   private repo on a personal account cannot buy. Nearly every merge here is an
   agent running `gh pr merge`, so enforcing it at that call covers the real
   traffic. It does not cover a human merging from the web UI — that gap is
   accepted, because a person merges deliberately and rarely.

   The failure it prevents is quiet: a PR goes green, master moves underneath
   it, and the stale merge lands anyway. Nothing reports an error — the checks
   that passed simply describe a tree that no longer exists. With four agents
   opening PRs against the same wave, master moves fast.

"Shared main checkout" is not a path match: a linked worktree's git-dir is
`<repo>/.git/worktrees/<name>` while its common-dir is `<repo>/.git`. When the
two are equal you are in a main checkout. That holds for any repo, so this stays
correct if the repo moves or is cloned elsewhere.

Read-only git (status, log, diff, fetch, pull, show) is never blocked, and the
hook only sees Claude's Bash tool — a human typing git in their own terminal is
unaffected.

Escape hatch: set IRTCHECK_GIT_GUARD=off in the environment to disable. Prefer
running from a worktree; the hatch exists for the case where the guard itself is
what's in the way. For rule 3 alone there is a narrower one,
IRTCHECK_ALLOW_STALE_MERGE=1, so that merging during a GitHub Actions outage
does not require switching the whole guard off and losing the master-push
protection with it.

Tests: python3 .claude/hooks/test_git_guard.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

# Verbs that write to the index, the working tree, or a ref. `pull` and `fetch`
# are deliberately absent: keeping the shared checkout current is the one write
# that is safe there, and blocking it would push agents toward worse workarounds.
MUTATING = {
    "commit",
    "push",
    "checkout",
    "switch",
    "reset",
    "restore",
    "rm",
    "clean",
    "stash",
    "merge",
    "rebase",
    "revert",
    "cherry-pick",
}

PROTECTED_BRANCHES = {"master", "main"}

# `git -C <dir>` and `git -c <k=v>` consume their argument; the rest of git's
# pre-subcommand options do not.
GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}

SHELL_OPERATORS = {"&&", "||", ";", "|", "&", "\n"}


def tokenize(command: str) -> list[str] | None:
    """Split a shell command into tokens, keeping operators separate.

    Returns None when the command cannot be parsed (unbalanced quotes, say). The
    caller then allows it: a guard that blocks every command it cannot read is a
    worse failure than the one it prevents, and the parse failure would have to
    survive Bash itself anyway.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return None


def segments(tokens: list[str]) -> list[list[str]]:
    """Split a token list on shell operators into individually-run commands."""
    out: list[list[str]] = [[]]
    for token in tokens:
        # punctuation_chars merges runs of operator characters, so `&&` arrives
        # as one token but so would `&&&`; treat anything operator-ish as a break.
        if token in SHELL_OPERATORS or set(token) <= {"&", "|", ";"}:
            out.append([])
        else:
            out[-1].append(token)
    return [seg for seg in out if seg]


def parse_git(segment: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Return (subcommand, -C directory, remaining args) for a git segment.

    (None, None, []) when the segment does not invoke git.
    """
    index = None
    for i, token in enumerate(segment):
        if token == "git" or token.endswith("/git"):
            index = i
            break
    if index is None:
        return None, None, []

    directory = None
    i = index + 1
    while i < len(segment):
        token = segment[i]
        if token in GIT_OPTS_WITH_ARG:
            if token == "-C" and i + 1 < len(segment):
                directory = segment[i + 1]
            i += 2
            continue
        if token.startswith("-"):
            if token.startswith("--git-dir=") or token.startswith("--work-tree="):
                pass
            i += 1
            continue
        return token, directory, segment[i + 1 :]
    return None, directory, []


def git_query(cwd: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def is_main_checkout(cwd: str) -> bool:
    """True when cwd is a repo's primary working tree, not a linked worktree."""
    if not os.path.isdir(cwd):
        return False
    git_dir = git_query(cwd, "rev-parse", "--absolute-git-dir")
    if git_dir is None:
        return False  # not a repo at all
    common = git_query(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common is None:
        return False
    return os.path.realpath(git_dir) == os.path.realpath(common)


def has_commits(cwd: str) -> bool:
    """False when HEAD is unborn — a repo that has no commits yet.

    Rule 1 exists because other agents read the shared checkout. A repo with no
    history has nothing for them to read: `git worktree add` refuses to run
    against an unborn HEAD, so no linked worktree can exist, and no branch can
    have been handed to anyone. The rule's own remedy — "work in
    .claude/worktrees/<name>" — is not available, which makes denying here a
    demand that cannot be satisfied.

    So the bootstrap commit is allowed, and only the bootstrap commit: the
    moment it lands HEAD resolves and every later write in this tree is denied
    normally. This is narrow by construction, not by trust — it can be true at
    most once in a repository's life.
    """
    return git_query(cwd, "rev-parse", "--verify", "HEAD") is not None


def push_targets_protected(args: list[str], cwd: str) -> bool:
    """True when a `git push` would move master/main."""
    refspecs = [a for a in args if not a.startswith("-")]
    # `git push origin master` — first positional is the remote, rest are refspecs.
    if len(refspecs) > 1:
        for spec in refspecs[1:]:
            dest = spec.split(":")[-1].lstrip("+")
            if dest.removeprefix("refs/heads/") in PROTECTED_BRANCHES:
                return True
        return False
    # No refspec: git pushes the current branch under the default push config.
    branch = git_query(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    return branch in PROTECTED_BRANCHES


# `gh pr merge` flags that consume the token after them, so the PR argument is
# not mistaken for one of their values.
GH_VALUE_FLAGS = {
    "-b", "--body",
    "-F", "--body-file",
    "-t", "--subject",
    "--match-head-commit",
    "--author-email",
}

# statusCheckRollup mixes CheckRun entries (conclusion) with StatusContext
# entries (state). These are the values that mean "do not merge this".
BAD_CHECK_STATES = {
    "FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED",
    "STARTUP_FAILURE", "STALE", "ERROR",
}
GOOD_CHECK_STATES = {"SUCCESS", "NEUTRAL", "SKIPPED"}


def parse_gh_pr_merge(segment: list[str]) -> tuple[bool, str | None]:
    """Return (is_a_pr_merge, pr_argument_or_None) for a `gh pr merge` segment."""
    index = None
    for i, token in enumerate(segment):
        if token == "gh" or token.endswith("/gh"):
            index = i
            break
    if index is None:
        return False, None

    rest = segment[index + 1 :]
    words = [t for t in rest if not t.startswith("-")]
    if words[:2] != ["pr", "merge"]:
        return False, None

    # First positional after `merge`, skipping the values of value-taking flags.
    seen_merge = False
    skip_next = False
    for token in rest:
        if skip_next:
            skip_next = False
            continue
        if token in GH_VALUE_FLAGS:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if not seen_merge:
            if token == "merge":
                seen_merge = True
            continue
        return True, token
    return True, None


def gh_json(args: list[str], cwd: str):
    """Run a gh command expected to print JSON. None on any failure."""
    binary = os.environ.get("IRTCHECK_GH_BIN", "gh")
    try:
        result = subprocess.run(
            [binary, *args], cwd=cwd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return None


def check_pr_merge(target: str | None, cwd: str) -> str | None:
    """Return a denial reason for this `gh pr merge`, or None to allow.

    Anything that cannot be determined — gh missing, unauthenticated, offline,
    an API shape we do not recognise — allows the merge. A guard that blocks
    every merge the moment GitHub is unreachable is worse than the race it
    prevents, and rule 3 is a convenience layer over a paid feature, not a
    security boundary.
    """
    view = ["pr", "view"]
    if target:
        view.append(target)
    view += ["--json", "number,headRefOid,baseRefName,state,statusCheckRollup"]
    pr = gh_json(view, cwd)
    if not isinstance(pr, dict) or "headRefOid" not in pr:
        return None

    number = pr.get("number", "?")
    base = pr.get("baseRefName")
    head = pr.get("headRefOid")

    # Is the PR behind its base? This is exactly the comparison GitHub's
    # "require branches to be up to date" makes, and `compare` reports it
    # without needing either commit present locally.
    if base and head:
        cmp = gh_json(
            ["api", f"repos/{{owner}}/{{repo}}/compare/{base}...{head}"], cwd
        )
        if isinstance(cmp, dict) and isinstance(cmp.get("behind_by"), int):
            behind = cmp["behind_by"]
            if behind > 0:
                return (
                    f"PR #{number} is {behind} commit(s) behind {base}. Its checks "
                    f"describe a tree that no longer exists, so merging it can land "
                    f"a break that nothing tested.\n"
                    f"Fix: `git fetch origin && git merge origin/{base}` on the PR "
                    f"branch, push, and let CI re-run.\n"
                    f"This is the client-side stand-in for branch protection's "
                    f"'require branches to be up to date' — see .claude/hooks/"
                    f"git_guard.py. Deliberate override: IRTCHECK_ALLOW_STALE_MERGE=1"
                )

    # An empty rollup allows. It means either "no checks are configured" or
    # "no check has registered yet" and the two are indistinguishable here —
    # during a GitHub Actions outage, PRs sit with no checks at all rather than
    # pending ones. Denying on empty would block a legitimately checkless PR and
    # would block every merge during such an outage. The staleness test above
    # still applies, and it is the more important of the two.
    rollup = pr.get("statusCheckRollup") or []
    failed, pending = [], []
    for check in rollup:
        if not isinstance(check, dict):
            continue
        name = check.get("name") or check.get("context") or "check"
        verdict = (check.get("conclusion") or check.get("state") or "").upper()
        if verdict in BAD_CHECK_STATES:
            failed.append(name)
        elif verdict not in GOOD_CHECK_STATES:
            pending.append(name)

    if failed:
        return (
            f"PR #{number} has failing checks: {', '.join(sorted(set(failed)))}. "
            "Fix them before merging, or override with IRTCHECK_ALLOW_STALE_MERGE=1 "
            "if you have confirmed the failure is unrelated — a GitHub Actions "
            "outage, say."
        )
    if pending:
        return (
            f"PR #{number} still has checks running or queued: "
            f"{', '.join(sorted(set(pending)))}. Wait for them, or override with "
            "IRTCHECK_ALLOW_STALE_MERGE=1 if GitHub Actions is degraded and you have "
            "verified the branch another way."
        )
    return None


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def main() -> None:
    if os.environ.get("IRTCHECK_GIT_GUARD") == "off":
        return

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    command = (payload.get("tool_input") or {}).get("command") or ""
    # Fast path. "gh" is a substring of ordinary words ("highlights"), which
    # costs a little parsing on those; the segment parser then finds no command
    # and allows. Missing a real `gh pr merge` would be the worse error.
    if "git" not in command and "gh" not in command:
        return

    base_cwd = payload.get("cwd") or os.getcwd()
    tokens = tokenize(command)
    if tokens is None:
        return

    cwd = base_cwd
    for segment in segments(tokens):
        # Track `cd` so `cd /elsewhere && git commit` is judged where it lands.
        if segment[0] == "cd" and len(segment) > 1:
            candidate = os.path.expanduser(segment[1])
            cwd = candidate if os.path.isabs(candidate) else os.path.join(cwd, candidate)
            cwd = os.path.normpath(cwd)
            continue

        is_merge, pr_target = parse_gh_pr_merge(segment)
        if is_merge:
            if os.environ.get("IRTCHECK_ALLOW_STALE_MERGE") != "1":
                reason = check_pr_merge(pr_target, cwd)
                if reason:
                    deny(reason)
            continue

        subcommand, directory, args = parse_git(segment)
        if subcommand is None:
            continue

        target = cwd
        if directory:
            expanded = os.path.expanduser(directory)
            target = expanded if os.path.isabs(expanded) else os.path.normpath(os.path.join(cwd, expanded))

        if subcommand == "push":
            if any(a in ("-f", "--force") for a in args):
                deny(
                    "git push --force is blocked. Rewriting a pushed branch "
                    "destroys work another agent may have based on it. If you "
                    "genuinely need it, ask the user to run it themselves."
                )
            if push_targets_protected(args, target):
                deny(
                    "Pushing to master is blocked. master should only move through "
                    "a merged PR, because the `contracts` CI job is this repo's "
                    "merge-time race check and a direct push skips it. Push your "
                    "branch and open a PR instead (see CLAUDE.md)."
                )

        if subcommand in MUTATING and is_main_checkout(target) and has_commits(target):
            deny(
                f"`git {subcommand}` is blocked in the shared main checkout "
                f"({target}). Other agents and the user read this tree; a write "
                "here races them — in the sibling AudioBookLib repo the same "
                "mistake left the checkout staged to delete 10,333 lines. Use the "
                "EnterWorktree tool and work in .claude/worktrees/<name> instead. "
                "Read-only git is fine here."
            )


if __name__ == "__main__":
    main()
