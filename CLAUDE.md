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
- **`dead`, `inverted` and `insufficient-data` are three different claims and
  no item may carry two.** They are three readings of one interval on `a`, and
  `IrtFit.validate()` rejects any item carrying more than one:

      spans zero               -> insufficient-data   we cannot tell
      inside ±DEAD_THRESHOLD   -> dead                confidently negligible
      wholly below zero, wider -> inverted            confidently backwards

  `insufficient-data` is a finding about the *data the user supplied*; `dead`
  is a finding about the *suite*; `inverted` is a finding about the *item* —
  weaker respondents get it right more often, which usually means a mis-keyed
  answer. An inverted item is the opposite of dead weight: it carries real
  information pointing the wrong way, so the report says "check the key" rather
  than "drop it".
  **`inverted` was split out of `dead` in schema 2**, and it was not a
  hypothetical tidy-up: under the old rule `hdi_high < DEAD_THRESHOLD` was
  satisfied by `[0.05, 0.30]` and `[-1.17, -0.22]` alike, and on the real HELM
  matrix *every* item that ever reached `dead` did so by the negative route —
  32 of 3,551 at 95 respondents, 3 at twelve, none by the small-positive one.
  The flag fired 35 times across two real fits and described the wrong thing
  every time. See `docs/validation.md` §2e.
  **Inverted items are excluded from `usable_items()` and `dead` items are
  not**, and the asymmetry is the point. `dead` can stay eligible because
  selection maximises information, information goes as `a²`, and a dead item's
  `a` is confidently near zero — so it is never picked. That argument does not
  extend to an inverted item, whose `|a|` is large: measured before the split,
  an anchor set of 100 picked one with `a = -1.72` and a set of 400 picked all
  three the suite had. An anchor set is scored by plain accuracy, and an item a
  stronger model reliably gets *wrong* subtracts from the signal.
  At five to fifteen respondents **most low-information items land in
  `insufficient-data`, and `dead` is rare** — and rarer than this file used to
  claim. `dead` needs the whole interval inside `±DEAD_THRESHOLD`, which caps
  `sd(a)` at `0.35 / (2 × 1.96) = 0.089`. Measured against synthetic ground
  truth, **no item reaches `dead` at 300 respondents and eighteen of 200 do at
  1,000** — a thousand respondents, not the hundred stated here through waves
  0-2. On real data it is rarer still: the HELM fits produce *no* `dead` items
  at either twelve or 95 respondents, because the items that used to land there
  are `inverted` instead. That is the honest answer, and saying so is the
  product.
- **`inverted` finds about half of mis-keyed items, and that is a known fitter
  bug, not a threshold.** Measured in `docs/validation.md` §4: `a` and `b`
  enter as `a(theta - b)`, so an item crossing from positive to negative `a` at
  fixed accuracy sends `b` through infinity, and the prior on `b` walls it off.
  Items near 50% accuracy cross; the rest stay positive and, as respondents
  are added, become *usable* or `dead` — 68 of the 69 `dead` flags in the
  sweep were planted mis-keyed items. Do not read an `inverted` count as a
  census, and do not "fix" it by loosening the flag.
- **The report's refusal reads the `insufficient-data` share, and §4 found
  that is the wrong quantity.** The share has a floor set by the suite, so it
  refuses at 15-25 models on anchor sets within 0.02 of the oracle, while the
  failure that does track respondent count — `select` returning fewer items
  than asked — is met with a warning that padding "would be worse than a short
  one", and measured, the short set ranks worse than the padded set *and* than
  a random one.
  `REFUSAL_SHARE` and `THIN_MODEL_COUNT` are unchanged pending that decision;
  do not retune the constants as if the shape were right.
- **`derives_from` is load-bearing.** Every respondent records the real model
  it came from. Leave-one-model-out must hold out *every pseudo-respondent
  derived from a model*, not one row: drop one and that model's other prompt
  variants leak its answers into the fit that chooses the anchor set, and the
  headline rank correlation comes out inflated with nothing to catch it.
  `ResponseMatrix.drop_model()` is the only correct way to do it.
- **The reported interval on `a` and `b` is not the guide's marginal, and the
  distinction is load-bearing.** `insufficient-data` fires when the interval on
  `a_i` spans zero, so the width *is* the claim. Mean-field SVI makes each
  marginal variance the conditional `1/precision_ii` rather than the marginal
  `Sigma_ii`, which for a 2PL is badly too narrow because `a_i` and `b_i` enter
  the likelihood only through `a_i (theta_j - b_i)` — and too narrow is the
  direction that makes the tool refuse too little. Measured coverage of the
  variational interval was 80.6-87.5% against a nominal 95%. The item intervals
  therefore come from the 2x2 expected information of `(a_i, b_i)` conditional
  on `theta_hat`; coverage 91.1-93.3%. The point estimates are still
  variational. See `src/irtcheck/fit/intervals.py` for the derivation and
  `tests/test_intervals.py`, which measures it in CI — the claim went unchecked
  and inverted in a docstring for three waves, which is why it is now a test.
  **`theta` needed a different correction and got one.** Its error was not
  width but *scale*: `theta ~ N(0, 1)` is a fixed ruler, a fit standardises to
  its own sample, and `n` draws from `N(0, 1)` have a sample sd that is not 1 —
  so every ability came out stretched by a shared factor that no per-respondent
  width covers. The variational interval covered 64.4% at fifteen respondents
  and information alone covered 64.4% too; adding the ruler's own uncertainty,
  `1/sqrt(2(n-1))` in quadrature and proportional to `|theta_j|`, reaches 88.9%.
  It vanishes as respondents are added, which is why all three methods agree at
  300.
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
- **No `Co-Authored-By` lines in commits.** This rule stands, and the history
  does not match it: 21 such trailers are already in `master`, in two different
  spellings, because the wave briefs asked for them and the briefs were wrong.
  Two agents noticed the contradiction independently and resolved it in opposite
  directions. It is not worth rewriting merged history over, so the trailers
  stay where they are and new commits do without. If you would rather have them,
  change this line — but do not leave the two disagreeing again.

## Releasing

**The version lives only in `src/irtcheck/__init__.py`.** `pyproject.toml`
reads it through `[tool.hatch.version]`, and `test_the_package_version_has_one_source`
fails if a static version comes back — it used to exist in both, and the tag
gate read one while `irtcheck --version` read the other.

    gh workflow run release.yml --ref <branch> -f target=testpypi   # rehearse
    git tag -a vX.Y.Z -m "irtcheck X.Y.Z" && git push origin vX.Y.Z   # publish

A rehearsal builds `<version>.dev<run id><attempt>`, publishes it to TestPyPI,
installs it back from TestPyPI on 3.11-3.13, and runs
`.github/release/quickstart.sh` against that install. A tag must match
`__version__` or `.github/release/decide.py` refuses before anything is built.
Both indexes use Trusted Publishing, so there are no tokens to manage, and a
publish job fails until its pending publisher and GitHub environment
(`pypi`, `testpypi`) exist. **PyPI never accepts a filename twice**, so a
real version number is spent the moment its upload succeeds — rehearse first.

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
`crosscheck/` — a not-shipped comparison on a 3.11 venv alongside R's `mirt`.
**That comparison has been run.** Against `mirt` 1.41 on a dense 300 x 60
matrix: Spearman 0.949 on `a`, 0.9991 on `b`, mean `|dP|` 0.0092, with `mirt`
and an independent scipy marginal-ML agreeing at logLik -9431.926306 both ways.
The convention differences are real and were reconciled explicitly rather than
assumed — `b = -d/a1` against `mirt`'s own `IRTpars`, and py-irt's `export()`
identified as a LogNormal *median* rather than a mean. **`mirt` does not
constrain `a > 0`** (2 of 59 slopes negative on dense, 22 of 183 on sparse), so
`mirt` and not py-irt is the reference matching our identification convention;
it is recorded in the artifact's `model.identification` field for that reason.
The comparison found a real bug — see the interval convention above, and
`docs/validation.md` section 2f.
