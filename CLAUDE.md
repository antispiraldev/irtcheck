# irtcheck

A CLI that fits an item response theory model to an existing eval response
matrix and reports which items carry measurement signal, which are dead weight,
and what a minimal high-information subset looks like. It does **not** run
evals: no API keys, no inference cost, no network at runtime.

Three documents, split by how often they change:

- `docs/spec.md` — the design this is built from. Changes rarely; when it does,
  say so in the PR body.
- `docs/build-plan.html` — who owns what, wave by wave, and the three spike
  findings that changed the spec. Read your brief before starting.
- `CLAUDE.md` — this file: constraints that bite while working in the repo.

## Commands

    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
    .venv/bin/python -m pytest                  # everything
    .venv/bin/python -m pytest -m "not slow"    # skips the fitting tests
    .venv/bin/ruff check .
    python3 .claude/hooks/test_git_guard.py     # dependency-free, run it directly

`torch` is a real dependency but a heavy one. To work on anything except the
fitter, install without it and save yourself the download — this is also
exactly what CI's `tests-light` job does:

    pip install typer rich numpy scipy pytest && pip install -e . --no-deps

## The lazy torch boundary

**`src/irtcheck/fit/` is the only package permitted to import torch or pyro,
and only inside function bodies or at its own module scope — never from a
module something else imports eagerly.**

This is not tidiness. The tool is two-stage on purpose: `fit` is slow and
writes a cached `.irt` artifact, and `report`, `select`, `validate` and the
HTML output read that artifact and are instant. Keeping torch out of their
import path means analysing a fit works on a machine that has never installed
it, and it means `irtcheck --help` does not pay a multi-second import.

Two things enforce it, deliberately from different angles:

- `tests/test_contracts.py` walks the AST of every module and fails on a
  top-level `import torch` outside `fit/`;
- CI's `contracts` and `tests-light` jobs install *without* torch at all, so
  the boundary breaking is a collection error rather than a subtle one.

`synth.py` is the reason this is affordable: it fabricates valid `IrtFit`
artifacts from known parameters with numpy alone, so almost the whole suite
runs with no fitter present.

## Working in parallel

**Agents work in `.claude/worktrees/<name>`, never in the shared checkout.**
`.claude/hooks/git_guard.py` enforces it: a `PreToolUse` hook denies `commit`,
`checkout`, `reset`, `restore`, `clean`, `stash`, `merge`, `rebase`, `revert`,
`cherry-pick`, `rm` and `push` when the command would land in a repo's primary
working tree, and denies force-pushes and pushes to `master` from anywhere.
Read-only git — `status`, `log`, `diff`, `fetch`, `pull` — is untouched, and so
is a human typing git in their own terminal; hooks only see Claude's Bash tool.
It fires on every Bash call, so it also catches `cd /main && git commit` and
`git -C /main reset`. Escape hatch, for when the guard itself is what's wrong:
`IRTCHECK_GIT_GUARD=off`.

It is a port from the sibling AudioBookLib repo, kept close to its original so
fixes can travel between the two — which is why `.claude` is excluded from
ruff.

**`gh pr merge` is denied on a PR that is behind its base, or whose checks are
red or still running.** Same hook. It is the client-side stand-in for branch
protection's *require branches to be up to date before merging*, which a
private repo on a personal account cannot buy. The failure it prevents is
quiet: a PR goes green, `master` moves underneath it, and the stale merge lands
with nothing reporting an error, because the checks that passed describe a tree
that no longer exists. With four agents opening PRs into one wave, `master`
moves fast. Fix a denial by merging `origin/master` into the PR branch and
letting CI re-run. Narrow override: `IRTCHECK_ALLOW_STALE_MERGE=1`.

**One brief, one worktree, one branch, one PR.** Your brief's *Owns* list in
`docs/build-plan.html` is exhaustive. If you need to edit a file outside it,
stop and say so rather than editing it — that file almost certainly belongs to
someone building right now.

**Avoid stacked PRs.** A chain of PRs based on each other can merge into their
intermediate branches instead of `master` and be orphaned on branch delete.

**`gh pr edit` does not work on this repo. Use `gh api -X PATCH
repos/{owner}/{repo}/pulls/{number}` instead.** It exits 1 with only a
Projects-classic GraphQL deprecation notice and no other output — and it does
*not* write the change. The danger is the silence: a `--body-file` update looks
like it succeeded, so read the body back if you must use it. Two agents hit
this independently during wave 1.

**`master` moves faster than CI completes.** With four agents merging into one
wave, a branch can go stale between `git push` and its checks going green —
this happened twice in wave 1. Re-check `gh pr view <n> --json
mergeStateStatus` immediately before merging, not from an earlier read. The
guard denies a stale merge, so the cost of forgetting is a denial, not a bad
merge; but the denial is easier to understand if you were expecting it.

**Fixtures are namespaced per brief.** Only wave 0 writes `tests/fixtures/`
directly; an adapter's fixtures go in `tests/fixtures/adapters/`. Two agents
both writing `tests/fixtures/matrix.jsonl` with different contents is a merge
that succeeds and a test suite that means nothing.

## Two files were frozen for wave 1

`src/irtcheck/cli.py` and `pyproject.toml` were written in wave 0 with the
whole command surface and the whole dependency set already in them, precisely
so that no wave-1 agent had to touch either. Every command is registered and
every flag declared; each delegates to a module in `commands/` owned by exactly
one brief. **Wave 1 is merged and the freeze is lifted**, but changing them is
still a deliberate single-owner change rather than something done in passing —
say so in the PR body, and never edit either while a wave is in flight.

The freeze had a cost worth remembering: the adapters needed four per-format
overrides and could not add flags, so they shipped as `IRTCHECK_*` environment
variables. They are flags now (`--model-id`, `--metric`, `--lmeval-filter`,
`--scorer`) and the variables still work, with the flag winning when both are
set. See `ADAPTER_OVERRIDES` in `commands/fit.py` for why the values travel
through `os.environ` rather than as arguments — the readers take `read(path)`
and nothing else, and that narrow signature is what lets `io/__init__.py`
discover adapters without a shared dispatch table.

## Introspecting the CLI surface

**Read the built click command — `typer.main.get_command(app).commands` — not
`callback.__annotations__`.** Two independent things defeat the obvious
approach, and together they made `test_no_two_commands_disagree_about_a_flag`
pass vacuously through the whole of wave 1, the exact period it existed to
police:

- `cli.py` has `from __future__ import annotations`, so every annotation is a
  *string* with no `__metadata__`. Resolving needs
  `typing.get_type_hints(..., include_extras=True)`.
- even resolved, typer's `OptionInfo.param_decls` is usually empty: in the
  `Annotated` style a lone positional argument to `typer.Option` is taken as
  `default` and the flag name is derived from the parameter name.
  `typer.Option("-o", "--output")` records `--output`;
  `typer.Option("--respondent-key")` records nothing.

`test_the_flag_inventory_is_not_empty` now guards the guard, so a future change
to the introspection fails loudly instead of silently disabling the check.

## The merge-time race check

`tests/test_contracts.py`, run as its own CI job ahead of the others. Every
other test fails on a single bad branch; these fail on a *combination* of
branches that were each green alone. It asserts:

- `SCHEMA_VERSION` is defined exactly once. Two agents both bumping it is the
  classic shape: individually green, silently broken once combined.
- An artifact round-trips **byte-identically**. This caught two real bugs on
  the day it was written: `sort_keys=True` was moving `schema_version` out of
  the lead position, and `gzip.GzipFile(filename=path)` was writing the
  *filename* into the gzip header, so the same fit saved to two paths differed.
- Every registered command answers `--help`, and no two commands document one
  flag differently.
- The lazy torch boundary above.

## Conventions

- Work on a branch, open a PR. `master` only moves through a merged PR, because
  `contracts` is a merge-time check and a direct push skips it.
- **`dead` and `insufficient-data` are different claims and must never both be
  set.** `dead` means we are confident the item does not discriminate — the
  upper end of its `a` interval is below `DEAD_THRESHOLD`. `insufficient-data`
  means we cannot tell, because the interval spans zero. The first is a finding
  about the suite; the second is a finding about the data the user supplied,
  and it excludes the item from ranking and selection rather than ranking it
  anyway. `IrtFit.validate()` rejects an item carrying both.
  At five to fifteen respondents **most low-information items land in
  `insufficient-data`, and `dead` is rare** — reaching it needs an interval
  narrow enough to sit wholly under the threshold, which needs roughly a
  hundred respondents. That is the honest answer, and saying so is the product.
- **`derives_from` is load-bearing.** Every respondent records the real model
  it came from. Leave-one-model-out must hold out *every pseudo-respondent
  derived from a model*, not one row: drop one and that model's other prompt
  variants leak its answers into the fit that chooses the anchor set, and the
  headline rank correlation comes out inflated with nothing to catch it.
  `ResponseMatrix.drop_model()` is the only correct way to do it.
- **Never coerce a continuous score to binary silently.** `coerce_correct`
  refuses `0.87`. Thresholding a `raw_score` is a documented escape hatch that
  has to print its threshold in the report header, not a quiet cast.
- **Respondent count is printed in every report header**, and it is the count
  of *real models*, not respondents. Reporting 60 respondents when they are
  twelve temperature samples of five models misrepresents the one thing this
  tool exists to be honest about.
- The fit artifact carries its own responses, so `irtcheck validate suite.irt`
  works with one argument. `--no-embed-responses` opts out and `validate` then
  explains what is missing.
- No `Co-Authored-By` lines in commits.

## Testing a stochastic fit

SVI is stochastic, and a flaky test across four concurrent agents costs more
than it catches. Fit tests seed `pyro.set_rng_seed`, run on CPU, and assert
**recovery of synthetic ground truth within a stated tolerance** rather than
exact values. They carry the `slow` marker and run in their own CI job.

Ground truth only exists in `synth.py`. Every claim the tool makes about the
fitter is checked against parameters generated there.

## Known constraint: py-irt is not a dependency

The spec originally said to fit with `py-irt`. It cannot be a runtime
dependency: every release from 0.4 onward declares
`Requires-Python >=3.9,<3.12`, and on 3.12+ `pip install py-irt` silently
resolves to **0.1.1** — an abandoned 2020 fork whose `fit()` builds its summary
list on the last line and returns without it. Its pins (`typer<0.15`,
`rich<14`) would also become ours.

So the 2PL is ours, written directly in Pyro, and py-irt appears only in
`crosscheck/` — a wave-2, not-shipped comparison run on a 3.11 venv alongside
R's `mirt`. Watch for sign and scale convention differences between packages
when reading those results; the identification choice is recorded in the
artifact's `model.identification` field for exactly that reason.
