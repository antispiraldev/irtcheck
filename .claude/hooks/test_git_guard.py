#!/usr/bin/env python3
"""Tests for git_guard.py. Dependency-free — run it directly:

    python3 .claude/hooks/test_git_guard.py

Builds a throwaway repo with a linked worktree in a temp dir rather than
pointing at this checkout, so it tests the main-checkout-vs-worktree rule
itself instead of whatever state the developer's tree happens to be in.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "git_guard.py")


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def build_repo(root):
    """A repo whose primary tree is on master, plus a linked worktree."""
    main = os.path.join(root, "repo")
    os.makedirs(main)
    git(main, "init", "-q", "-b", "master")
    git(main, "config", "user.email", "test@example.com")
    git(main, "config", "user.name", "test")
    open(os.path.join(main, "README"), "w").write("hello\n")
    git(main, "add", "README")
    git(main, "commit", "-qm", "initial")
    worktree = os.path.join(root, "wt")
    git(main, "worktree", "add", "-q", "-b", "feature", worktree)
    return main, worktree


def ask(command, cwd, env=None):
    """Run the guard; return (denied, reason)."""
    payload = json.dumps({"tool_name": "Bash", "cwd": cwd, "tool_input": {"command": command}})
    proc = subprocess.run(
        [sys.executable, GUARD], input=payload, capture_output=True, text=True,
        env=dict(os.environ, **(env or {})),
    )
    assert proc.returncode == 0, f"guard exited {proc.returncode}: {proc.stderr}"
    if not proc.stdout.strip():
        return False, ""
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    return out.get("permissionDecision") == "deny", out.get("permissionDecisionReason", "")


STUB_GH = '''#!/usr/bin/env python3
"""Stand-in for `gh`, driven by env vars, so the merge tests need no network."""
import json, os, sys

args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(" ".join(args) + "\\n")

if os.environ.get("STUB_FAIL") == "1":
    sys.stderr.write("stub: simulated gh failure\\n")
    sys.exit(1)

if args[:2] == ["pr", "view"]:
    print(os.environ.get("STUB_PR", "{}"))
    sys.exit(0)
if args[:1] == ["api"] and "/compare/" in args[1]:
    print(os.environ.get("STUB_COMPARE", "{}"))
    sys.exit(0)
sys.exit(1)
'''


def merge_guard_cases(root, worktree):
    """Scenarios for rule 3: `gh pr merge` staleness and check state."""
    stub = os.path.join(root, "gh-stub")
    with open(stub, "w") as fh:
        fh.write(STUB_GH)
    os.chmod(stub, 0o755)
    log = os.path.join(root, "gh-calls.log")

    green = json.dumps({
        "number": 42, "headRefOid": "a" * 40, "baseRefName": "master",
        "state": "OPEN",
        "statusCheckRollup": [
            {"name": "backend", "conclusion": "SUCCESS"},
            {"name": "frontend", "conclusion": "SUCCESS"},
        ],
    })
    failing = json.dumps({
        "number": 43, "headRefOid": "b" * 40, "baseRefName": "master",
        "state": "OPEN",
        "statusCheckRollup": [{"name": "backend", "conclusion": "FAILURE"}],
    })
    pending = json.dumps({
        "number": 44, "headRefOid": "c" * 40, "baseRefName": "master",
        "state": "OPEN",
        "statusCheckRollup": [{"name": "backend", "conclusion": None, "status": "IN_PROGRESS"}],
    })
    legacy_status = json.dumps({
        "number": 45, "headRefOid": "d" * 40, "baseRefName": "master",
        "state": "OPEN",
        "statusCheckRollup": [{"context": "ci/legacy", "state": "FAILURE"}],
    })
    no_checks = json.dumps({
        "number": 46, "headRefOid": "e" * 40, "baseRefName": "master",
        "state": "OPEN", "statusCheckRollup": [],
    })
    up_to_date = json.dumps({"behind_by": 0, "ahead_by": 3})
    stale = json.dumps({"behind_by": 20, "ahead_by": 1})

    base = {"IRTCHECK_GH_BIN": stub, "STUB_LOG": log}
    #  label, command, env, expect_denied, expected substring
    return [
        ("merge, current with base",
         "gh pr merge 42 --squash",
         {**base, "STUB_PR": green, "STUB_COMPARE": up_to_date}, False, ""),
        ("merge, 20 behind base",
         "gh pr merge 42 --squash",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, True, "20 commit(s) behind"),
        ("merge with a failing check",
         "gh pr merge 43 --squash",
         {**base, "STUB_PR": failing, "STUB_COMPARE": up_to_date}, True, "failing checks"),
        ("merge with a pending check",
         "gh pr merge 44 --squash",
         {**base, "STUB_PR": pending, "STUB_COMPARE": up_to_date}, True, "running or queued"),
        ("legacy StatusContext failure",
         "gh pr merge 45 --squash",
         {**base, "STUB_PR": legacy_status, "STUB_COMPARE": up_to_date}, True, "failing checks"),
        ("PR with no checks configured",
         "gh pr merge 46 --squash",
         {**base, "STUB_PR": no_checks, "STUB_COMPARE": up_to_date}, False, ""),
        ("no PR argument (current branch)",
         "gh pr merge --squash",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, True, "behind"),
        ("--body value not mistaken for the PR arg",
         'gh pr merge --body "merge 999 now" 42 --squash',
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, True, "behind"),
        ("--admin does not bypass",
         "gh pr merge 42 --squash --admin",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, True, "behind"),
        ("compound: fetch && merge is still checked",
         "git fetch origin && gh pr merge 42 --squash",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, True, "behind"),
        ("IRTCHECK_ALLOW_STALE_MERGE=1 overrides",
         "gh pr merge 42 --squash",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale, "IRTCHECK_ALLOW_STALE_MERGE": "1"}, False, ""),
        ("IRTCHECK_GIT_GUARD=off overrides",
         "gh pr merge 42 --squash",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale, "IRTCHECK_GIT_GUARD": "off"}, False, ""),
        ("gh unavailable fails open",
         "gh pr merge 42 --squash",
         {**base, "STUB_FAIL": "1"}, False, ""),
        ("gh pr view is not a merge",
         "gh pr view 42 --json state",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, False, ""),
        ("gh pr create is not a merge",
         "gh pr create --fill",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, False, ""),
        ("gh run list is not a merge",
         "gh run list --limit 5",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, False, ""),
        ("a word containing 'gh' is not a command",
         "echo 'highlights are bright'",
         {**base, "STUB_PR": green, "STUB_COMPARE": stale}, False, ""),
    ]


def main():
    root = tempfile.mkdtemp(prefix="git-guard-test-")
    try:
        MAIN, WT = build_repo(root)
        cases = [
            # Read-only work in the shared checkout stays possible.
            ("read-only in main", "git status", MAIN, False),
            ("log in main", "git log --oneline -5", MAIN, False),
            ("pull in main", "git pull --ff-only", MAIN, False),
            ("fetch in main", "git fetch origin", MAIN, False),
            # Writes to the shared checkout are the corruption vector.
            ("commit in main", "git commit -m 'x'", MAIN, True),
            ("checkout in main", "git checkout HEAD -- .", MAIN, True),
            ("reset in main", "git reset --hard origin/master", MAIN, True),
            ("restore in main", "git restore --source=abc123 .", MAIN, True),
            ("clean in main", "git clean -fd", MAIN, True),
            ("stash in main", "git stash", MAIN, True),
            ("merge in main", "git merge origin/master", MAIN, True),
            ("rebase in main", "git rebase master", MAIN, True),
            # The same writes are the whole point of a worktree.
            ("commit in worktree", "git commit -m 'x'", WT, False),
            ("checkout in worktree", "git checkout -b feature2", WT, False),
            ("reset in worktree", "git reset --hard HEAD", WT, False),
            # Reaching back into the shared checkout is still reaching into it.
            ("cd to main then commit", f"cd {MAIN} && git commit -m x", WT, True),
            ("git -C main reset", f"git -C {MAIN} reset --hard", WT, True),
            ("git -C worktree reset", f"git -C {WT} reset --hard", MAIN, False),
            # Force-push and master-push are denied regardless of location.
            ("force push", "git push --force origin feature", WT, True),
            ("force push short flag", "git push -f origin feature", WT, True),
            ("push to master", "git push origin master", WT, True),
            ("push HEAD:master", "git push origin HEAD:master", WT, True),
            ("push refs/heads/master", "git push origin HEAD:refs/heads/master", WT, True),
            ("push to main", "git push origin main", WT, True),
            ("push feature branch", "git push -u origin feature", WT, False),
            ("bare push from worktree", "git push", WT, False),
            ("bare push from main on master", "git push", MAIN, True),
            # Non-git, and git-adjacent, commands pass through.
            ("non-git command", "npm run build", MAIN, False),
            ("git substring only", "echo 'no git here'", MAIN, False),
            ("gh is not git", "gh pr create --fill", WT, False),
            # Compound commands are judged segment by segment.
            ("compound read-only", "git fetch && git status && git log", MAIN, False),
            ("compound hiding a commit", "git fetch && git status && git commit -m x", MAIN, True),
            # An unparseable command is allowed through rather than blocking work.
            ("unbalanced quotes fail open", "git commit -m 'unterminated", MAIN, False),
        ]

        failures = 0
        for label, command, cwd, expected in cases:
            denied, _ = ask(command, cwd)
            ok = denied == expected
            failures += 0 if ok else 1
            got = "DENY" if denied else "allow"
            want = "DENY" if expected else "allow"
            print(f"{'ok  ' if ok else 'FAIL'}  {label:<32} -> {got:<5} (want {want})")

        # The escape hatch has to work, or the guard is a trap.
        env = dict(os.environ, IRTCHECK_GIT_GUARD="off")
        payload = json.dumps({"tool_name": "Bash", "cwd": MAIN, "tool_input": {"command": "git commit -m x"}})
        proc = subprocess.run([sys.executable, GUARD], input=payload, capture_output=True, text=True, env=env)
        ok = proc.stdout.strip() == ""
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'}  {'IRTCHECK_GIT_GUARD=off bypasses':<32} -> {'allow' if ok else 'DENY':<5} (want allow)")

        # Rule 3: `gh pr merge` staleness and check state.
        merge_cases = merge_guard_cases(root, WT)
        print()
        for label, command, env, expected, needle in merge_cases:
            denied, reason = ask(command, WT, env=env)
            ok = denied == expected and (not needle or needle in reason)
            failures += 0 if ok else 1
            got = "DENY" if denied else "allow"
            want = "DENY" if expected else "allow"
            note = "" if ok or not denied else f"  [{reason.splitlines()[0][:60]}]"
            print(f"{'ok  ' if ok else 'FAIL'}  {label:<40} -> {got:<5} (want {want}){note}")

        # The cases above deny regardless of which PR was looked up, so they
        # cannot prove the argument parser picked the right one. Assert on what
        # the guard actually asked gh for.
        stub = os.path.join(root, "gh-stub")
        target_checks = [
            ("gh pr merge 42 --squash", "pr view 42"),
            ('gh pr merge --body "merge 999 now" 42 --squash', "pr view 42"),
            ("gh pr merge --squash", "pr view --json"),  # no target => current branch
            ("gh pr merge -t 'subject 7' 108 --admin", "pr view 108"),
        ]
        print()
        for command, expected_call in target_checks:
            log = os.path.join(root, f"log-{abs(hash(command))}.txt")
            ask(command, WT, env={
                "IRTCHECK_GH_BIN": stub, "STUB_LOG": log,
                "STUB_PR": json.dumps({"number": 1, "headRefOid": "f" * 40,
                                       "baseRefName": "master", "statusCheckRollup": []}),
                "STUB_COMPARE": json.dumps({"behind_by": 0}),
            })
            calls = open(log).read() if os.path.exists(log) else ""
            ok = expected_call in calls
            failures += 0 if ok else 1
            print(f"{'ok  ' if ok else 'FAIL'}  resolved target: {command[:44]:<46} "
                  f"-> {'saw ' + repr(expected_call) if ok else 'GOT ' + repr(calls.strip()[:60])}")

        # --- the unborn-HEAD carve-out ------------------------------------
        #
        # A repo with no commits gets exactly one allowed write in its main
        # checkout: the bootstrap commit. `git worktree add` refuses to run
        # before the first commit, so the remedy rule 1 points at does not
        # exist yet and denying would be a demand that cannot be met. The
        # moment that commit lands, the normal rule resumes — which is the
        # second half of this test and the half that matters.
        print()
        empty = os.path.join(root, "empty")
        os.makedirs(empty)
        git(empty, "init", "-q", "-b", "master")
        git(empty, "config", "user.email", "test@example.com")
        git(empty, "config", "user.name", "test")
        open(os.path.join(empty, "README"), "w").write("hi\n")
        git(empty, "add", "README")

        bootstrap_cases = []
        denied, _ = ask("git commit -m bootstrap", empty)
        bootstrap_cases.append(("bootstrap commit into an empty repo", denied, False))

        # Still not a licence to push to master, even here.
        denied, _ = ask("git push origin master", empty)
        bootstrap_cases.append(("push to master is still denied", denied, True))

        git(empty, "commit", "-qm", "bootstrap")

        denied, _ = ask("git commit -m second", empty)
        bootstrap_cases.append(("the very next commit is denied", denied, True))
        denied, _ = ask("git reset --hard", empty)
        bootstrap_cases.append(("reset after bootstrap is denied", denied, True))

        for label, got, want in bootstrap_cases:
            ok = got == want
            failures += 0 if ok else 1
            verdict = "DENY" if got else "allow"
            wanted = "DENY" if want else "allow"
            print(f"{'ok  ' if ok else 'FAIL'}  {label:<40} -> {verdict:<5} (want {wanted})")

        total = (len(cases) + 1 + len(merge_cases) + len(target_checks)
                 + len(bootstrap_cases))
        print(f"\n{total - failures}/{total} passed")
        return 1 if failures else 0
    finally:
        # `git worktree add` leaves the worktree registered in repo/.git; the
        # whole repo is inside root, so removing root is enough.
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
