# irtcheck

Fit an item response theory model to an eval response matrix **you already
have**, and find out which items carry measurement signal, which are dead
weight, and what a minimal high-information subset looks like.

Eval suites are scored by aggregate accuracy over a fixed item set, which
treats every item as equally informative. In practice items vary enormously in
difficulty and in how sharply they separate stronger models from weaker ones. A
suite can be thousands of items and only a few hundred of measurement value,
and nobody running it knows which.

irtcheck answers three questions about a suite you already ran:

1. Which items discriminate, and which give everyone the same answer?
2. Is the suite measuring precisely in the ability range your models occupy?
3. What is the smallest subset that reproduces the full-suite ranking?

It does **not** run evals. No API keys, no inference cost, no network at
runtime.

---

## Install

```
pip install irtcheck
```

Python 3.11, 3.12 and 3.13. `torch` and `pyro-ppl` come with it because
`irtcheck fit` needs them — but they are imported lazily, so `report` and
`select` run on a machine that has never installed torch, given a `.irt` file
from one that has. (`validate` refits, so it does need them.) If you only ever
analyse fits someone else produced:

```
pip install typer rich numpy scipy matplotlib
pip install irtcheck --no-deps
```

## Quickstart

You need one line per graded response. Three fields:

```json
{"model_id": "claude-sonnet-4-5", "item_id": "mmlu_hs_bio_0412", "correct": 1}
```

Then:

```console
$ irtcheck fit responses.jsonl --respondent-key model_id,prompt_variant -o suite.irt
20 respondents (10 real models, key model_id+prompt_variant), 600 items, 12,000 responses (100% of the grid)
wrote suite.irt (71 KiB) in 9.7s — ELBO -7,389.7 over 2,000 epochs, seed 0
  256 insufficient-data · 3 dead · 22 ceiling · 16 floor
next: irtcheck report suite.irt

$ irtcheck report suite.irt --limit 8
      respondents  10 real models → 20 respondents (2.0 per model, --respondent-key model_id,prompt_variant)
            items  600 items  (306 usable for ranking and selection)
             dead  3 (0.5%)  confidently do not discriminate (a interval entirely below 0.35)
insufficient-data  256 (42.7%)  cannot tell — a interval spans zero; excluded from ranking and selection
  ceiling / floor  ceiling 22 (3.7%, p ≥ 0.99)   floor 16 (2.7%, p ≤ 0.01)
        off-range  0 (0.0%)  difficulty outside the ability range these respondents occupy

  item          a     a 95% HDI      b      b 95% HDI    n   p(correct)  flags
  item_00518  2.27   [0.94, 3.61]  -0.09  [-0.61, 0.43]  20        0.50
  item_00165  2.16   [0.85, 3.48]  -0.26  [-0.84, 0.31]  20        0.55
  ...

$ irtcheck select suite.irt -n 50 -o anchor.json
50 of 306 usable items (600 total) · 10 models (20 respondents, key model_id+prompt_variant)
Mean ability standard error over these models: 0.220 (1.000 with no items at all).
Marginal gain: first item 0.6285, last item 0.3553. Where that flattens is where n stops buying precision.

$ irtcheck validate suite.irt
Leave-one-model-out · 10 models · 600 items · 20 respondents (key model_id+prompt_variant)
    n  items  Spearman  Kendall tau
   25     25    +0.994       +0.978
   50     50    +0.988       +0.956
  100    100    +1.000       +1.000
  200    200    +1.000       +1.000
  400    252*   +1.000       +1.000
```

That last table is the headline: a 25-item subset, chosen by a fit that never
saw the model it was then used to rank, put ten models in almost exactly their
full-suite order.

> **Those numbers are from synthetic data, and the real ones are much weaker.**
> On a real 12-model × 3,551-item matrix the same command gives Spearman
> **+0.662 at n=25** and **+0.867 at n=400** — not +0.994 and +1.000. The
> synthetic table checks that the code is correct; it is not evidence that
> twenty-five items are enough for your suite. Read
> [Does it actually work?](#does-it-actually-work) before quoting either.

Fitting is the slow part, so it is behind a cached artifact: `report`,
`select` and `validate` read `suite.irt` and are instant (`validate` refits, so
it is not — see below).

---

## If you use lm-eval-harness, read this first

**lm-eval only writes per-item records when you ask it to.** A normal run keeps
task-level aggregates and throws the per-sample detail away, so a finished run
cannot be read after the fact. It has to be re-run with both flags:

```
lm_eval --model hf --model_args pretrained=<model> --tasks <tasks> \
        --output_path runs/ --log_samples
```

This is the one place irtcheck asks you to run an eval, and we would rather say
it at the top than have you discover it after downloading the tool. Everything
else about irtcheck is built so that you never have to.

`--log_samples` produces, per model:

```
runs/<model_name_sanitized>/results_<timestamp>.json            aggregates
runs/<model_name_sanitized>/samples_<task>_<timestamp>.jsonl    one file per task
```

Point irtcheck at one `samples_*.jsonl`, or at the directory holding several:

```
irtcheck fit runs/ --format lmeval -o suite.irt
```

A directory needs an explicit `--format`, because sniffing has to open a file.

**Inspect (`inspect_ai`) has no equivalent problem.** It records per-sample
scores in every eval log by default, so a log written months ago is readable
as-is:

```
irtcheck fit logs/ --format inspect_ai -o suite.irt
```

---

## Input

### The reference format

JSONL, one object per (model, item) response. `model_id`, `item_id` and
`correct` are required:

```json
{"model_id": "claude-sonnet-4-5", "item_id": "mmlu_hs_bio_0412", "correct": 1}
```

`subject`, `split`, `raw_score` and `prompt_variant` are carried through to
reports when present, and any other field can be named in `--respondent-key`.

### Adapters

| `--format`    | Reads                                                        |
| ------------- | ------------------------------------------------------------ |
| `jsonl`       | the reference format above                                   |
| `csv`         | CSV or TSV with the same columns                             |
| `lmeval`      | lm-eval-harness `samples_*.jsonl` (needs `--log_samples`)    |
| `inspect_ai`  | Inspect `.json` eval logs, and `.eval` zips where the stdlib can decompress them |

A single file is sniffed; a directory needs `--format`.

Two things worth knowing about the adapters before you are surprised by them:

- **`.eval` files from current Inspect are zstandard-compressed**, which
  Python's `zipfile` cannot read before 3.14 and which irtcheck will not add a
  dependency for. Such a file is refused with instructions rather than
  half-read: `inspect log convert run.eval --to json --output-dir logs-json/`.
  Older (0.3.50-era) deflate-compressed `.eval` files read fine.
- **`--epochs 3` in Inspect scores each sample several times.** That is one
  item answered repeatedly by one model, not several items. The epoch is kept
  on the record, so the honest reading is to make the repeats
  pseudo-respondents with `--respondent-key model_id,epoch`.

### Four decisions the adapters will not guess

A harness log is sometimes genuinely ambiguous, and rather than pick for you
the adapters refuse and name the choice. These four flags on `irtcheck fit`
settle them:

| Flag                     | Used when                                                                       |
| ------------------------ | ------------------------------------------------------------------------------- |
| `--model-id`             | A `samples_*.jsonl` has been moved out of its `runs/<model>/` directory and has no `results_*.json` beside it, so there is nothing to take the model name from. Without it the read fails rather than inventing a respondent. |
| `--metric`               | A task logs several per-sample metrics (`acc` and `acc_norm`, say). The default order is `acc`, `exact_match`, `em`, `acc_norm`; between unfamiliar ones the adapter refuses to choose. |
| `--lmeval-filter`        | A task logs every doc once per answer-extraction filter (gsm8k: `strict-match` and `flexible-extract`). Those are the same item scored two ways, not two items, so the adapter keeps one filter per file — the first it sees by default — and warns about the rest. |
| `--scorer`               | An Inspect sample carries scores from several scorers and no convention says which is the eval's headline. With one scorer it is used; with several the read fails and names them. |

Each also has an `IRTCHECK_*` environment variable, which is how wave 1 shipped
these while the CLI surface was frozen. They are still honoured, and the flag
wins when both are set:

| Flag              | Variable                   |
| ----------------- | -------------------------- |
| `--model-id`      | `IRTCHECK_LMEVAL_MODEL_ID` |
| `--metric`        | `IRTCHECK_LMEVAL_METRIC`   |
| `--lmeval-filter` | `IRTCHECK_LMEVAL_FILTER`   |
| `--scorer`        | `IRTCHECK_INSPECT_SCORER`  |

### When your harness writes nothing irtcheck can read

This is the fallback the whole input contract rests on, and it is deliberately
tiny. If your scoring loop already knows which model got which item right, it
already has everything irtcheck needs:

```python
import csv

with open("responses.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, ["model_id", "item_id", "correct"])
    writer.writeheader()
    for result in your_results:            # whatever your loop already has
        writer.writerow(
            {
                "model_id": result.model,  # "claude-sonnet-4-5"
                "item_id": result.item,    # stable across runs, unique across tasks
                "correct": int(result.passed),
            }
        )
```

```
irtcheck fit responses.csv -o suite.irt
```

Three things are worth getting right while you are in there, because they are
expensive to fix later:

- **`item_id` must be stable across runs and unique across tasks.** It is the
  join key between every model's answers. A row index is fine; a row index
  without the task name in front of it is not, because task A's item 0 and task
  B's item 0 will silently become one item.
- **`correct` must be 0 or 1.** irtcheck refuses `0.87` rather than picking a
  threshold for you. If your scorer is continuous, threshold it yourself, put
  the original in `raw_score`, and say what cutoff you used. A quiet cast to
  binary is a wrong difficulty estimate that nothing downstream can detect.
- **More models beat more items.** Which is the next section.

---

## Respondent count is the binding constraint

Psychometrics assumes hundreds of respondents. You have five to fifteen models.
Everything awkward about using IRT here follows from that one fact, and
`--respondent-key` is the cheapest thing you can do about it.

Temperature samples, prompt variants, checkpoints and quantizations of one
model all count as separate respondents for parameter estimation. Compose them:

```
irtcheck fit responses.jsonl --respondent-key model_id,prompt_variant -o suite.irt
```

Five models with four prompt variants each is twenty respondents, and twenty
respondents estimate item parameters far better than five do. The cost is a
row in your input per (model, variant, item), which is inference you may
already have run.

Two things keep this honest rather than a way of inflating a number:

- **Pseudo-respondent abilities are nuisance parameters.** They are estimated
  and then not reported. Twelve temperature samples of one model are not twelve
  models, and no ranking irtcheck prints will pretend otherwise.
- **Every report header counts real models, not respondents.** It prints both
  when they differ, with the ratio. Reporting 60 respondents when they are
  twelve samples of five models would misrepresent the one thing this tool
  exists to be honest about.

Pseudo-respondents buy information about *items* and almost none about the
*ability scale*, so they do not make a five-model ranking trustworthy. Below
five real models the report says so.

---

## `dead` and `insufficient-data` are different claims

This is the heart of the tool, so it is worth being precise. Both flags appear
on items that are not earning their place, and they mean opposite things.

**`dead`** — we are confident the item does not discriminate. The upper end of
its 95% interval on `a` is below `0.35`. Whatever ability a respondent has, this
item barely changes its mind. **This is a finding about your suite**: the item
is dead weight and you can drop it.

**`insufficient-data`** — we cannot tell. The interval on `a` spans zero, so
the fit cannot distinguish "separates strong models from weak ones" from
"separates weak from strong" from "does nothing". **This is a finding about the
data you gave us.** Such items are excluded from ranking and from selection
rather than ranked anyway, and the report points at adding respondents.

They are never both set on one item; `IrtFit.validate()` rejects an artifact
that claims both about anything.

**At five to fifteen respondents, most low-information items land in
`insufficient-data` and `dead` is rare.** In the quickstart above, 20
respondents gave 256 insufficient-data and 3 dead out of 600 items. That is not
a bad run. Reaching `dead` needs an interval narrow enough to sit *wholly*
below the threshold, and that takes on the order of a hundred respondents. A
report that summed the two into "bad items: 259" would erase exactly the
distinction that matters, and would make the honest small-N answer look like a
broken fit.

When more than half the suite is unrankable, `report` leads with a refusal
instead of a table, and names `--respondent-key` as the next step. The tool
being willing to say *you need more respondents before I can rank these items*
is the product, not an error path.

Three more flags, all about items that cannot separate anybody regardless of
what the model says about `a`:

| Flag        | Means                                                                  |
| ----------- | ---------------------------------------------------------------------- |
| `ceiling`   | `p(correct) ≥ 0.99` — everyone gets it right                           |
| `floor`     | `p(correct) ≤ 0.01` — everyone gets it wrong                           |
| `off-range` | difficulty well outside the ability range your respondents occupy. The item may discriminate beautifully, just not for anyone in this matrix — which is the finding the test information curve exists to make visible. |

---

## Commands

### `fit`

```
irtcheck fit responses.jsonl -o suite.irt
    --respondent-key model_id,prompt_variant   compose pseudo-respondents
    --format lmeval                            force an adapter (required for a directory)
    --priors hierarchical|vague                partial pooling, or not
    --epochs 2000                              SVI steps
    --seed 0                                   fits are reproducible
    --device cpu|cuda
    --no-embed-responses                       do not carry the matrix in the artifact
    --model-id / --metric / --lmeval-filter / --scorer
                                               the four adapter decisions above
```

Fits a two-parameter logistic model,

```
P(correct) = 1 / (1 + exp(-a_i * (theta_j - b_i)))
```

in Pyro by stochastic variational inference, and writes posterior summaries,
flags, diagnostics and (by default) the response matrix itself to a gzipped
JSON artifact.

`hierarchical` priors are the default and partially pool item parameters toward
the suite-level distribution, so a thin item degrades toward the prior instead
of producing an extreme point estimate. `vague` uses fixed wide priors with no
pooling at all, and exists to show how much the pooling is doing.

**Discrimination pools toward zero, not toward a learned positive mean**, and
that is the decision the whole refusal behaviour rests on. The textbook prior
`a_i ~ N(mu_a, sigma_a)` with `mu_a` learned works exactly as designed and
ruins the product: it pools every item toward the suite average, so genuinely
dead items come back looking respectable and no item's interval ever spans
zero — which makes `insufficient-data` unreachable. A `LogNormal` prior on `a`
makes it unreachable by construction. So the population prior is
`sigma_a ~ LogNormal(0, 0.5)`, `a_i ~ N(0, sigma_a)`: still partial pooling,
but what a thin item is pulled toward is *does not discriminate*. An item earns
a discrimination from its data or it does not get one. Difficulty has no such
tension and keeps the conventional form with both hyperparameters learned.

Identification, recorded in the artifact's `model.identification` field:
`theta ~ N(0, 1)` fixed, which sets location and scale; `a` is estimated on the
real line and **not** constrained positive, which is the point above and also
what `mirt` does; the reflection `(a, b, theta) -> (-a, -b, -theta)` is resolved
toward positive mean `a`. A 95% interval on `a` may therefore legitimately
contain zero — that is not a bug, it is the flag.

1PL/Rasch is not offered: it constrains discrimination to be equal across
items, and unequal discrimination is the entire point of the tool. 3PL is a
reasonable idea for multiple-choice suites and needs more data than the small-N
case has.

The artifact carries its own responses, which is why `irtcheck validate
suite.irt` works with one argument. `--no-embed-responses` opts out and
`validate` then explains what is missing.

### `report`

```
irtcheck report suite.irt
    --html out.html          a self-contained HTML report — NOT IMPLEMENTED YET
    --json                   machine-readable instead of a table
    --sort discrimination|difficulty|id
    --limit 40               0 for all items
```

Per-item parameters with credible intervals and flags, over a header that
counts real models, items, usable items, and every flag separately. The
`--json` payload is generated from the same column list as the table, so a
column the table shows and the JSON omits is not expressible.

`--html` **is not implemented yet.** It is declared, and it exits with an error
saying so rather than writing a file that is not the report the flag promises.
When it lands it writes one self-contained file with no external assets, whose
centrepiece is the test information curve plotted against the ability
distribution of the respondents in your matrix — the plot that shows, at a
glance, a suite measuring precisely in an ability range none of your models
occupy. The terminal report and `--json` are complete.

### `select`

```
irtcheck select suite.irt -n 100 -o anchor.json
    --adaptive               per-respondent selection — NOT IMPLEMENTED YET
```

Emits the N most informative items as JSON. Item information for a 2PL is
`I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))`, and **the integral is
taken against the posterior ability means of the respondents in your matrix**,
not against a textbook N(0, 1) grid. That is the difference between "is this a
good test?" in the abstract and "does this test measure the models I actually
have?", and the two answers diverge whenever a suite was built for a generation
of models that has since been outgrown.

Items flagged `insufficient-data`, `ceiling` or `floor`, and items with no
responses in this fit, are not eligible. `dead` items stay eligible and are
simply never worth picking — their information is near zero by definition —
which is a property that falls out rather than one that had to be special-cased.
Asking for more items than are eligible gives you what there is and says so
rather than padding the set with items the report said it could not read.

Output is a fixed set, because a regression suite needs the same items every
run or scores are not comparable over time. `--adaptive` would select a
different set per respondent; it is declared and **not implemented**, exiting
with an explanation rather than quietly returning the fixed set. Deliberately
so — adaptive selection demos well and is the wrong default here.

### `validate`

```
irtcheck validate suite.irt
    --sizes 25,50,100,200,400
    --json
    --seed 0
    --epochs 2000
```

The headline number, and the only command here that is slow — it refits the
model once per *real model*, so budget roughly (models × fit time) plus
selection. Twelve models × 3,551 items is twelve refits.

Selecting items using every model and then reporting that the subset reproduces
the ranking of those same models is circular and worthless: the anchor set was
chosen knowing the answer. So instead, for each model *k*: hold it out, fit on
what remains, select an anchor set of size *n* from *that* fit, score model *k*
on the anchor items only, and compare its rank to its full-suite rank. Report
Spearman and Kendall tau against *n*.

**Holding out a model holds out every pseudo-respondent derived from it.** With
`--respondent-key model_id,prompt_variant` one model is several respondents;
dropping one row would leak that model's other variants into the fit that
chooses the anchor set, and the correlation would come out inflated with nothing
anywhere to catch it.

The headline columns are plain accuracy over the anchor items, because that is
what you actually do with an anchor set once you have it. The dim columns
re-estimate ability from the same responses with item parameters held fixed; the
two diverge exactly when an anchor set skews hard or easy, which is worth
seeing.

---

## Does it actually work?

Partly, and the parts matter. The per-item analysis holds up on real data; the
small-anchor-set claim does not hold up nearly as well as the synthetic number
suggests. All three answers below are measured, and the least flattering one is
last rather than omitted.

### On synthetic data: yes, and that only proves the pipeline is correct

The numbers in the quickstart are real measurements, on a matrix generated from
a known 2PL by `irtcheck.synth` — 10 models × 2 prompt variants, 600 items,
seed 5:

| n   | items | Spearman | Kendall tau |
| --- | ----- | -------- | ----------- |
| 25  | 25    | +0.994   | +0.978      |
| 50  | 50    | +0.988   | +0.956      |
| 100 | 100   | +1.000   | +1.000      |
| 200 | 200   | +1.000   | +1.000      |
| 400 | 252   | +1.000   | +1.000      |

(n=400 returns 252 items because only that many were eligible in the held-out
fits — the rest are `insufficient-data`, `ceiling` or `floor`. `select` reports
the shortfall rather than padding the set.)

**This is synthetic data, and the items really do come from a 2PL because we
drew them from one.** Real eval items do not. So this table is a correctness
check on the pipeline — the fitter recovers parameters, selection picks
informative items, leave-one-model-out is wired up without leakage — and it is
*not* evidence about how well IRT describes your suite. Read it as "the code
does what it says", not "IRT works on evals".

An unlabelled version of this table would be the most dishonest thing this
README could contain, which is why it is labelled twice.

### On real data: a public response matrix exists, and the tool runs on it

HELM publishes its full benchmark output, per instance, to an open Google Cloud
Storage bucket. Eighteen HELM Lite scenarios across every published version give
**95 models × 3,551 items, 323,656 real graded responses**, every item answered
by at least 87 of the 95 models. Cut to a realistic shortlist — twelve models
spanning 0.29 to 0.78 full-suite accuracy — `irtcheck fit` takes 29 s and the
flag distribution lands almost exactly where the synthetic matrix does at
comparable respondent count:

|                     | 12 real models × 3,551 items | synthetic, 20 respondents × 600 items |
| ------------------- | ---------------------------- | ------------------------------------- |
| usable for ranking  | 1,664 (46.9%)                | 306 (51.0%)                           |
| `insufficient-data` | 1,602 (45.1%)                | 256 (42.7%)                           |
| `dead`              | 29 (0.8%)                    | 3 (0.5%)                              |
| `ceiling` / `floor` | 3.8% / 4.3%                  | 3.7% / 2.7%                           |

So the small-N behaviour this README describes is not an artefact of how
`synth.py` draws parameters — it is what real model responses do too.

And grouping the output by scenario produces the kind of finding the tool was
written for, which synthetic data could not have produced:
`legalbench_international_citizenship_questions` is the **largest** scenario in
that matrix (1,000 items, 28% of it) and the **weakest** — 72% of its items
unrankable at twelve models, and the lowest mean discrimination of any scenario
present. OpenBookQA is half the size and yields more usable items.

A second finding came out of the same run and is the best illustration of why
per-item analysis is worth the trouble. On that scenario the median accuracy
across all 89 models that ran it is 0.536 — chance, for a two-way question —
and four models sit far *below* chance, the lowest at 0.034. Scoring 0.034
where guessing gives 0.5 is not a knowledge result; it is a systematic
answer-format or answer-extraction failure, and three of the four are otherwise
strong models that an aggregate leaderboard therefore ranks below models
answering at random. The signal was in *which* responses were wrong, which is
exactly what an aggregate score discards.

**And going from 12 to 95 respondents on the same 3,551 items confirms, on real
data, the claim this tool's refusal behaviour rests on:**

| items out of 3,551 | 12 models   | 95 models   |
| ------------------ | ----------- | ----------- |
| usable for ranking | 1,664 (47%) | 3,272 (92%) |
| `insufficient-data`| 1,602 (45%) | 225 (6.3%)  |
| `dead`             | 29 (0.8%)   | 94 (2.6%)   |

Nothing about the items changed. The only thing that changed was how much
evidence there was about each one. "We cannot tell" became "we can tell", and
2.6% of the suite turned out to be genuinely dead weight — a claim that was
simply not available at twelve models. That is `insufficient-data` being a
statement about your data, demonstrated rather than asserted.

### On real data: the anchor-set claim is much weaker, and you should know that

This is the least flattering measurement here and the one most worth reading.
`irtcheck validate` on that same twelve-model real matrix:

| n   | items | Spearman | Kendall tau | synthetic Spearman |
| --- | ----- | -------- | ----------- | ------------------ |
| 25  | 25    | +0.662   | +0.504      | +0.994             |
| 50  | 50    | +0.775   | +0.585      | +0.988             |
| 100 | 100   | +0.629   | +0.455      | +1.000             |
| 200 | 200   | +0.830   | +0.687      | +1.000             |
| 400 | 400   | +0.867   | +0.727      | +1.000             |

**On this real matrix a 25-item anchor set does not reproduce the ranking.** Four
hundred items reach +0.867, which is useful; twenty-five do not, and the curve
is not even monotone in n. Four candidate causes — a unidimensional 2PL fitted
across maths, law and commonsense, twelve respondents being thin for a rank
correlation, mislabelled responses, and multiple-choice guessing a 2PL has no
parameter for — are laid out in [`docs/validation.md`](docs/validation.md) **as
hypotheses, none of them measured**, along with how two of them could be tested
on the same data.

This does not invalidate `report` or `select`: the flags, the
`insufficient-data`/`dead` distinction and the per-scenario finding above all
behaved as designed on real data. It does mean the anchor-set claim needs
stating with an honest n. **And it means `validate` is doing its job** — a tool
that reported +0.99 here would be broken.

The full method, the per-scenario tables, a non-IRT cross-check that agrees,
the places where it disagrees, and the commands to reproduce all of it are in
[`docs/validation.md`](docs/validation.md).

### What we have not done

- **No cross-check against `mirt` or `py-irt` yet.** Comparing our Pyro 2PL
  against established implementations is a planned exercise (`crosscheck/`, on a
  3.11 venv, not shipped) and is not finished. Nothing here should be read as
  "agrees with mirt".
- **No performance claims beyond three measurements.** `fit` took 9.7 s on 20
  respondents × 600 items, 29.3 s on 12 × 3,551, and 23.5 s on 95 × 3,551
  (323,656 responses) — all on CPU at the default 2,000 SVI steps. Those are
  the only timings we stand behind. Note the 95-respondent fit was *not* slower
  than the 12-respondent one on the same items: cost here is dominated by item
  count and step count, not respondents. It will not extrapolate cleanly to
  your matrix.
- **`py-irt` is not a dependency**, despite the design having originally called
  for it: every release from 0.4 onward declares
  `Requires-Python >=3.9,<3.12`, and on 3.12+ `pip install py-irt` silently
  resolves to an abandoned 2020 fork whose `fit()` returns without its results.
  The 2PL is ours, written directly in Pyro, which is what buys 3.12 and 3.13.

---

## Development

```
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest                    # everything
.venv/bin/python -m pytest -m "not slow"      # skips the fitting tests
.venv/bin/ruff check .
```

`torch` is a heavy download and most of the suite does not need it, because
`synth.py` fabricates valid artifacts from known parameters with numpy alone.
To work on anything except the fitter:

```
pip install typer rich numpy scipy pytest && pip install -e . --no-deps
```

`src/irtcheck/fit/` is the only package permitted to import torch or pyro, and
only inside function bodies or at its own module scope. Two CI jobs enforce it
from different angles. See [`CLAUDE.md`](CLAUDE.md) for that and the rest of the
constraints that bite, [`docs/spec.md`](docs/spec.md) for the design, and
[`docs/build-plan.html`](docs/build-plan.html) for how it was built.

## Licence

MIT.
