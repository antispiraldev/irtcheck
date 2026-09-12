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
`irtcheck fit` needs them — but they are imported lazily, so `report`
(including `--html`) and `select` run on a machine that has never installed
torch, given a `.irt` file from one that has. (`validate` refits, so it does
need them.) If you only ever analyse fits someone else produced:

```
pip install typer rich numpy scipy
pip install irtcheck --no-deps
```

That is genuinely enough: the HTML report's plot is hand-built SVG, so nothing
in the analysis path imports matplotlib either, despite it being a declared
dependency.

## Quickstart

You need one line per graded response. Three fields:

```json
{"model_id": "claude-sonnet-4-5", "item_id": "mmlu_hs_bio_0412", "correct": 1}
```

Then:

```console
$ irtcheck fit responses.jsonl --respondent-key model_id,prompt_variant -o suite.irt
20 respondents (10 real models, key model_id+prompt_variant), 600 items, 12,000 responses (100% of the grid)
wrote suite.irt (70 KiB) in 11.3s — ELBO -7,544.9 over 2,000 epochs, seed 0
  429 insufficient-data · 1 dead · 23 ceiling · 17 floor
note: 429 of 600 items have a discrimination interval spanning zero — with this
many respondents the data cannot tell whether they separate anyone. They are
excluded from ranking and selection. More respondents is the fix.
next: irtcheck report suite.irt

$ irtcheck report suite.irt --limit 8
╭──────────────────────── cannot rank these items yet ─────────────────────────╮
│  Not enough respondents to rank these items: 429 of 600 items (72%) are      │
│  insufficient-data.                                                          │
│                                                                              │
│  Their discrimination interval spans zero, so this fit cannot tell whether   │
│  they separate stronger models from weaker ones. That is a statement about   │
│  the data supplied, not a verdict on the items.                              │
│                                                                              │
│  Separately, 1 item is flagged dead: there the fit is confident the item     │
│  does not discriminate. That is a finding about the suite.                   │
│                                                                              │
│  Cheapest next step — raise respondent count without running new models.     │
╰──────────────────────────────────────────────────────────────────────────────╯

      respondents  10 real models → 20 respondents (2.0 per model, --respondent-key model_id,prompt_variant)
            items  600 items  (131 usable for ranking and selection)
             dead  1 (0.2%)  confidently do not discriminate (a interval entirely below 0.35)
insufficient-data  429 (71.5%)  cannot tell — a interval spans zero; excluded from ranking and selection
  ceiling / floor  ceiling 23 (3.8%, p ≥ 0.99)   floor 17 (2.8%, p ≤ 0.01)
        off-range  0 (0.0%)  difficulty outside the ability range these respondents occupy

  item          a     a 95% HDI      b       b 95% HDI     n   p(correct)  flags
  item_00318  1.88   [0.56, 3.19]  -0.81  [-1.49, -0.14]  20        0.75
  item_00165  1.84   [0.49, 3.19]  -1.06  [-1.80, -0.31]  20        0.80
  ...

$ irtcheck report suite.irt --html suite.html
wrote suite.html          # self-contained, no external assets; the information curve is in here

$ irtcheck select suite.irt -n 50 -o anchor.json
50 of 131 usable items (600 total) · 10 models (20 respondents, key model_id+prompt_variant)
Mean ability standard error over these models: 0.234 (1.000 with no items at all).
Marginal gain: first item 0.5491, last item 0.2883. Where that flattens is where n stops buying precision.

$ irtcheck validate suite.irt
Leave-one-model-out · 10 models · 600 items · 20 respondents (key model_id+prompt_variant)
    n  items  Spearman  Kendall tau  Spearman (theta)  tau (theta)
   25     25    +0.921       +0.796            +0.927       +0.822
   50     50    +0.964       +0.911            +0.964       +0.911
  100     74*   +0.976       +0.911            +0.952       +0.867
  200     74*   +0.964       +0.911            +0.952       +0.867
  400     74*   +0.964       +0.911            +0.952       +0.867
```

**Ten models is below where this tool can rank items, and it says so.** That is
the quickstart on purpose: 72% of the suite comes back "cannot tell", `report`
leads with a refusal instead of a table, and the 131 items that survive still
recover the full-suite ranking to +0.92 from 25 of them. A tool that printed a
confident table here would be making most of it up.

That last table is the headline: a 25-item subset, chosen by a fit that never
saw the model it was then used to rank, put ten models in almost exactly their
full-suite order.

> **Those numbers are from synthetic data, and the real ones are much weaker.**
> On a real 12-model × 3,551-item matrix the same command gives Spearman
> **+0.691 at n=25** and **+0.890 at n=400** — not +0.921 and +0.976. The
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
respondents gave 429 insufficient-data and 1 dead out of 600 items. That is not
a bad run — it is the only honest reading of twenty responses per item.

`dead` is rare for a reason worth stating exactly, because the obvious guess is
wrong. It needs the interval on `a` to sit *wholly inside* `(0, 0.35)`: it has
to clear zero as well as the threshold, or the item is `insufficient-data`
instead. That caps the standard error at `0.35 / (2 × 1.96) = 0.089`, which is a
great deal of evidence about one item. Measured against synthetic ground truth,
no item reaches `dead` at 300 respondents and eighteen of 200 do at 1,000 — so
it takes on the order of a *thousand* respondents, not a hundred. Below that,
the handful of items that do reach it get there by a second route: an interval
lying wholly *below* zero, which is an item weaker models get right more often.
When `dead` does fire it is reliable — 27 of 29 items flagged across six
synthetic fits really do have a true `a` below the threshold.

A report that summed the two into "bad items: 430" would erase exactly the
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
| `off-range` | difficulty well outside the ability range your respondents occupy. The item may discriminate beautifully, just not for anyone in this matrix — which is the finding `report --html` draws the test information curve to make visible. |

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
    --html out.html          also write a self-contained HTML report
    --json                   machine-readable instead of a table
    --sort discrimination|difficulty|id
    --limit 40               0 for all items
```

Per-item parameters with credible intervals and flags, over a header that
counts real models, items, usable items, and every flag separately. The
`--json` payload is generated from the same column list as the table, so a
column the table shows and the JSON omits is not expressible.

`--html` writes **one self-contained file with no external assets** — no CDN,
no fonts, no image requests, so it survives being emailed or dropped in a
bucket. About 86 KB for a 600-item fit.

Its centrepiece is the **test information curve plotted against the ability
distribution of the respondents in your matrix**, as inline SVG. That is the
plot that shows at a glance when a suite measures precisely in an ability range
none of your models occupy, and it states the finding in words as well as
drawing it. The page's own subtitle, on the quickstart fit above, reads:

```
Test information peaks at theta +0.24; the respondents span -1.52 to +1.94
and receive 85% of that peak.
```

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
**95 models × 3,551 items, 336,375 real graded responses**, every item answered
by at least 93 of the 95 models. Cut to a realistic shortlist — twelve models
spanning 0.29 to 0.78 full-suite accuracy — `irtcheck fit` takes 13 s:

|                     | 12 real models × 3,551 items | synthetic, 20 respondents × 600 items |
| ------------------- | ---------------------------- | ------------------------------------- |
| usable for ranking  | 568 (16.0%)                  | 131 (21.8%)                           |
| `insufficient-data` | 2,980 (83.9%)                | 429 (71.5%)                           |
| `dead`              | 3 (0.08%)                    | 1 (0.2%)                              |
| `ceiling` / `floor` | 3.8% / 4.3%                  | 3.8% / 2.8%                           |

**The small-N behaviour this README describes is not an artefact of how
`synth.py` draws parameters** — it is what real model responses do too, with the
real matrix somewhat harsher than the synthetic one at comparable respondent
count. Both columns were re-measured after the interval fix below, from a matrix
rebuilt with the scripts in [`scripts/`](scripts/). `ceiling` came out at 3.8%
and `floor` at 4.3% before and after, to the decimal, which is the check that
the rebuild is the same matrix: those two flags read observed rates rather than
intervals, so they should not have moved.

And grouping the output by scenario produces the kind of finding the tool was
written for, which synthetic data could not have produced:
`legalbench_international_citizenship_questions` is the **largest** scenario in
that matrix (1,000 items, 28% of it) and the **weakest** — the lowest mean
discrimination of any scenario present (0.54 against OpenBookQA's 2.15), and at
twelve models only 7 of its 1,000 items are usable, against 229 of OpenBookQA's
500. It also holds **26 of the 32 items the 95-model fit is confident
discriminate backwards**, which nothing else in the matrix has any of.

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

| items out of 3,551 | 12 models     | 95 models     |
| ------------------ | ------------- | ------------- |
| usable for ranking | 568 (16.0%)   | 2,918 (82.2%) |
| `insufficient-data`| 2,980 (83.9%) | 601 (16.9%)   |
| `dead`             | 3 (0.08%)     | 32 (0.90%)    |

Nothing about the items changed. The only thing that changed was how much
evidence there was about each one. "We cannot tell" became "we can tell" for
two thirds of the suite. That is `insufficient-data` being a statement about
your data, demonstrated rather than asserted.

The `dead` row turned out to say something other than what the flag claims, and
it is worth knowing before you act on it: every one of those 32 items is
`dead` because its interval on `a` lies wholly *below* zero, meaning weaker
models get it right more often — not because it fails to discriminate.
[`docs/validation.md`](docs/validation.md) §2e has the detail.

### On real data: the anchor-set claim is much weaker, and you should know that

This is the least flattering measurement here and the one most worth reading.
`irtcheck validate` on that same twelve-model real matrix:

| n   | items | Spearman | Kendall tau | synthetic Spearman |
| --- | ----- | -------- | ----------- | ------------------ |
| 25  | 25    | +0.691   | +0.523      | +0.921             |
| 50  | 50    | +0.755   | +0.545      | +0.964             |
| 100 | 100   | +0.687   | +0.504      | +0.976             |
| 200 | 200   | +0.855   | +0.657      | +0.964             |
| 400 | 204\* | +0.890   | +0.748      | +0.964             |

**On this real matrix a 25-item anchor set does not reproduce the ranking.** It
takes a couple of hundred items to reach +0.89, which is useful; twenty-five do
not, and the curve is not even monotone in n. The asterisk is worth a second
look: n=400 could only supply **204** eligible items, and those 204 scored
+0.890 — better than the 400 items the previous, too-narrow intervals made
eligible. Half the anchor set, a better ranking. Four candidate causes — a unidimensional 2PL fitted
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

- **The cross-check against `mirt` and `py-irt` is done, and it found a real
  bug.** Both references ran — R 4.3.3 with `mirt` 1.41, and `py-irt` 0.7.1 on
  CPython 3.11 — in `crosscheck/` (not shipped; it needs toolchains the package
  does not). On a dense 300 × 60 matrix our point estimates agree with `mirt` at
  **Spearman 0.949 on discrimination and 0.9991 on difficulty**, mean absolute
  difference in predicted probability 0.0092. `mirt` and an independently
  written scipy marginal-ML agree with each other at logLik −9431.926306 to six
  decimal places, which is what makes the reference trustworthy rather than just
  present. Two things it found: our reported intervals were about half the width
  they should be (now fixed — see below), and at twelve respondents unpenalised
  ML has no interior maximum at all, which is the empirical case for the
  hierarchical priors. Note `mirt` does **not** constrain `a > 0`, so it is the
  reference whose identification convention matches ours.
- **The reported interval on `a` and `b` is not the guide's.** Mean-field SVI
  understates posterior correlations, so its marginals are too narrow — and for
  this tool that is the dangerous direction, because `insufficient-data` fires
  when the interval spans zero. Measured, the nominal 95% variational interval
  covered the true `a` 80.6–87.5% of the time. The item intervals now come from
  the conditional information matrix instead, which carries the `a`–`b`
  correlation the guide drops: coverage 91.1–93.3%, still a little narrow.
- **The interval on `theta` carries the uncertainty in its own ruler.**
  `theta ~ N(0, 1)` is a fixed prior, so a fit standardises abilities to *its
  own sample* — and fifteen draws from `N(0, 1)` have a sample sd of 0.82 as
  readily as 1.0. Every ability then comes out stretched by a shared factor,
  which no per-respondent width covers: the variational interval covered the
  true ability **64.4%** of the time at fifteen respondents, and recomputing it
  from information alone changed nothing. Admitting the ruler is uncertain —
  the sample sd of `n` draws has relative standard error `1/sqrt(2(n-1))` —
  takes coverage to **88.9%**, and it widens the intervals on the extreme
  models while leaving the middle of the pack alone, which is the right shape
  for a scale error. The correction vanishes as respondents are added.
- **No performance claims beyond three measurements.** `fit` took 11.3 s on 20
  respondents × 600 items, 13.1 s on 12 × 3,551, and 19.4 s on 95 × 3,551
  (336,375 responses) — all on CPU at the default 2,000 SVI steps. Those are
  the only timings we stand behind. Note the 95-respondent fit is barely slower
  than the 12-respondent one on the same items despite eight times the
  responses: cost here is dominated by item count and step count, not
  respondents. It will not extrapolate cleanly to your matrix.
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
