# irtcheck — eval suite health check

A CLI that fits an item response theory model to an existing eval response matrix and reports which items carry measurement signal, which are dead weight, and what a minimal high-information subset looks like.

## Problem

Eval suites are scored by aggregate accuracy over a fixed item set, which treats every item as equally informative. In practice items vary enormously in difficulty and in how well they separate stronger models from weaker ones. A suite can be thousands of items and only a few hundred of measurement value, and nobody running it knows which.

This tool answers three questions about a suite the user already has:

1. Which items discriminate, and which give everyone the same answer?
2. Is the suite measuring precisely in the ability range the user's models actually occupy?
3. What is the smallest subset that reproduces the full-suite ranking?

## Non-goals (v1)

Explicitly out of scope. Each is a plausible v2 and each doubles the timeline.

- Running evals. The tool does not call model APIs, does not need keys, does not incur inference cost.
- Contamination detection.
- Rubric / LLM-judge criterion analysis (this is the intended v2 direction; do not build it now).
- Leaderboard scraping or hosted service.
- Multidimensional IRT.

## Input contract

Consume a response matrix produced by someone else's harness. Canonical format is JSONL, one record per (model, item) response:

```json
{"model_id": "claude-sonnet-4-5", "item_id": "mmlu_hs_bio_0412", "correct": 1}
```

Required fields: `model_id` (str), `item_id` (str), `correct` (0/1 or bool).
Optional fields carried through to reports if present: `subject`, `split`, `raw_score`, `prompt_variant`.

Adapters to write:

- plain JSONL (above) — the reference format
- CSV with the same three columns
- lm-eval-harness sample logs
- Inspect eval logs

**Do this first:** before committing to the adapter interface, inspect what lm-eval-harness and Inspect actually write to disk. The riskiest assumption in this project is that a parseable per-item response record exists in the wild. If it doesn't, the JSONL contract plus a documented "here's how to emit this from your harness" recipe is the fallback, and that changes the README more than the code.

### Pseudo-respondents

Respondent count is the binding constraint (see below). The input contract must let a user inflate it cheaply: temperature samples, prompt variants, checkpoints, and quantizations of the same model all count as separate respondents for parameter estimation. Support a `--respondent-key` flag that composes the respondent identity from multiple fields, e.g. `--respondent-key model_id,prompt_variant`. Abilities of pseudo-respondents are nuisance parameters; they are not reported as model rankings.

## Model

Two-parameter logistic. Probability that respondent *j* answers item *i* correctly:

```
P(correct) = 1 / (1 + exp(-a_i * (theta_j - b_i)))
```

- `theta_j` — respondent latent ability
- `b_i` — item difficulty (ability at which the item is a coin flip)
- `a_i` — item discrimination (how sharply the item separates around `b_i`)

1PL/Rasch is not sufficient: it constrains discrimination to be equal across items, and unequal discrimination is the entire point of the tool. 3PL (adds a guessing floor `c_i`) is a reasonable flag for multiple-choice suites but not the default — it needs more data than the small-N case can support.

Fit with `py-irt` (Bayesian, pyro backend). Hierarchical priors over item parameters so that sparse items are pooled toward the suite-level distribution rather than estimated independently.

### Small-N handling

Assume 5–15 real models. Psychometrics assumes hundreds of respondents; this tool cannot. Three mitigations, all required:

1. Pseudo-respondents via `--respondent-key` (above).
2. Hierarchical priors — partial pooling so thin items degrade toward the prior instead of producing extreme point estimates.
3. **Refusal.** Report credible intervals on `a_i`. Items whose interval spans zero are flagged `insufficient-data` and excluded from ranking and selection, not silently ranked. Print respondent count prominently in every report header.

The tool being willing to say "you need more respondents before I can rank these items" is a core feature, not an error path. Where output is uncertain, the message should point at adding respondents as the next step.

## Commands

Two-stage, with a cached fit artifact. Fitting is the slow part; separating it means report/select/validate are instant and presentation can be iterated on without refitting.

```
irtcheck fit responses.jsonl -o suite.irt
irtcheck report suite.irt [--html out.html]
irtcheck select suite.irt -n 100 [-o anchor.json]
irtcheck validate suite.irt
```

- `fit` — parse, build response matrix, fit 2PL, persist parameters + posterior summaries + metadata (respondent count, item count, fit diagnostics).
- `report` — per-item table (difficulty, discrimination, credible intervals, flags) plus suite-level diagnostics: respondent count, dead-item count, ceiling/floor rates, test information curve.
- `select` — emit a static anchor set of N items as JSON (item ids). Selection maximizes test information over the observed ability distribution.
- `validate` — holdout rank correlation (see below). This is the headline number.

### Static, not adaptive

Default output is a fixed anchor set. A regression suite needs the same items every run or scores aren't comparable over time. Adaptive item selection demos well but is wrong for the primary use case; put it behind `--adaptive` as a secondary feature if there's time.

## Validation

Selecting items using all models and then reporting that the subset reproduces the full-suite ranking is circular and worthless. Do leave-one-model-out:

1. Hold out model *k*.
2. Fit the 2PL on the remaining *k-1* respondents.
3. Select an anchor set of size *n* from that fit.
4. Score the held-out model on the anchor set only.
5. Compare its rank against its full-suite ground-truth rank.
6. Loop over all models; report rank correlation (Spearman / Kendall tau) as a function of *n*.

This is the demo. Build it early — it constrains everything else, and the number it produces is what makes the tool credible rather than a toy.

## Output

- Terminal: rich table for `report`, plain summary for `validate`. Respect `--json` for machine consumption.
- `--html`: single self-contained file, no external assets. The centerpiece is the **test information curve plotted against the ability distribution of the respondents in the matrix.** That plot is the pitch — it shows at a glance when a suite is measuring precisely in an ability range none of the user's models occupy.
- Header block on every report: respondent count, item count, dead items, ceiling/floor rates, fit diagnostics.

## Stack

- Python 3.11+, `pip install irtcheck`, no R dependency. Requiring rpy2 or a local R install would kill adoption for a CLI aimed at ML engineers.
- `py-irt` (pyro) for fitting, `typer` or `click` for CLI, `rich` for tables, `matplotlib` for the information curve.
- No network access at runtime.

### One-time cross-check (not shipped)

Fit one matrix in R's `mirt` and confirm discrimination and difficulty estimates agree with the `py-irt` output within tolerance. Watch for sign conventions and scaling differences between packages. This catches parameterization and convergence bugs and is worth an afternoon; it also means the README can say the estimates were validated against the reference implementation.

## Build order

1. Input parsing + response matrix construction, plain JSONL only.
2. `fit` with py-irt, hierarchical priors, persisted artifact.
3. `validate` — holdout loop and rank correlation. Do this before polishing reports.
4. `report` terminal output with flags and credible intervals.
5. `select` anchor-set emission.
6. HTML report with the information curve.
7. Harness adapters (lm-eval-harness, Inspect, CSV).
8. Optional: `--adaptive`, 3PL flag.

## Open questions

- Do lm-eval-harness and Inspect emit per-item records by default, or does the user have to opt in? Determines whether adapters are step 7 or step 1.
- Minimum viable respondent count before the tool refuses outright — pick a threshold empirically from synthetic data rather than guessing.
- Whether to support thresholding a continuous `raw_score` into binary as a documented escape hatch. If yes, always print the threshold in the report.
