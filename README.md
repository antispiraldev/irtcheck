# irtcheck

Fit an item response theory model to an eval response matrix you already have,
and find out which items carry measurement signal, which are dead weight, and
what a minimal high-information subset looks like.

> **Status: wave 0.** The contracts, the CLI surface and the test harness are
> in place. `fit`, `report`, `select` and `validate` are registered but not yet
> implemented — each belongs to a wave-1 brief. See
> [`docs/build-plan.html`](docs/build-plan.html) for who owns what, and
> [`docs/spec.md`](docs/spec.md) for the design this is built from.
> This README is a placeholder; `wave2/docs` writes the real one.

## What it is for

Eval suites are scored by aggregate accuracy over a fixed item set, which
treats every item as equally informative. In practice items vary enormously in
difficulty and in how sharply they separate stronger models from weaker ones. A
suite can be thousands of items and only a few hundred of measurement value,
and nobody running it knows which.

irtcheck answers three questions about a suite you already have:

1. Which items discriminate, and which give everyone the same answer?
2. Is the suite measuring precisely in the ability range your models occupy?
3. What is the smallest subset that reproduces the full-suite ranking?

It does **not** run evals. No API keys, no inference cost, no network at
runtime.

## Input

One JSON object per line:

```json
{"model_id": "claude-sonnet-4-5", "item_id": "mmlu_hs_bio_0412", "correct": 1}
```

`model_id`, `item_id` and `correct` are required. `subject`, `split`,
`raw_score` and `prompt_variant` are carried through to reports when present,
and any other field can be named in `--respondent-key`.

Adapters for CSV, lm-eval-harness sample logs and Inspect eval logs are a
wave-1 brief. Two notes on those, established by the wave-0 spike:

- **Inspect** records per-sample scores in every eval log by default.
- **lm-eval-harness** only writes them under `--log_samples` alongside
  `--output_path`. An existing run without those flags has aggregates only and
  has to be re-run.

## Commands

```
irtcheck fit responses.jsonl -o suite.irt
irtcheck report suite.irt [--html out.html]
irtcheck select suite.irt -n 100 [-o anchor.json]
irtcheck validate suite.irt
```

Fitting is the slow part, so it is separated behind a cached artifact:
`report`, `select` and `validate` read `suite.irt` and are instant.

## Respondent count is the binding constraint

Psychometrics assumes hundreds of respondents. You have five to fifteen models.
Three things follow, and all of them are load-bearing:

- **Pseudo-respondents.** Temperature samples, prompt variants, checkpoints and
  quantizations of one model all count as separate respondents for parameter
  estimation. `--respondent-key model_id,prompt_variant` composes them. Their
  abilities are nuisance parameters and are never reported as model rankings.
- **Hierarchical priors**, so thin items degrade toward the suite-level
  distribution instead of producing extreme point estimates.
- **Refusal.** Items whose discrimination interval spans zero are flagged
  `insufficient-data` and excluded from ranking and selection rather than
  silently ranked. Respondent count is printed in every report header.

The tool being willing to say *you need more respondents before I can rank
these items* is a feature, not an error path.

## Development

```
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest              # everything
.venv/bin/python -m pytest -m 'not slow'  # skips the fitting tests
```

`torch` and `pyro-ppl` are real dependencies but are imported lazily, inside
`irtcheck.fit`. Analysing a cached artifact works without them. See
[`CLAUDE.md`](CLAUDE.md).

## Licence

MIT.
