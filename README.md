# irtcheck

Fit an item response theory model to an eval response matrix **you already
have**, and find out which items carry measurement signal, which are dead
weight or backwards, and whether a smaller subset would rank your models as
well as the whole suite.

Eval suites are scored by aggregate accuracy over a fixed item set, which
treats every item as equally informative. In practice items vary enormously in
difficulty and in how sharply they separate stronger models from weaker ones. A
suite can be thousands of items and only a few hundred of measurement value,
and nobody running it knows which.

irtcheck answers three questions about a suite you already ran:

1. Which items discriminate, and which give everyone the same answer?
2. Is the suite measuring precisely in the ability range your models occupy?
3. Does a subset of it rank models as well as the whole thing — and does
   choosing that subset beat drawing it at random?

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
wrote suite.irt (72 KiB) in 11.8s — ELBO -7,389.7 over 2,000 epochs, seed 0
  378 insufficient-data · 0 dead · 1 inverted · 22 ceiling · 16 floor
note: 1 item(s) discriminate backwards — weaker respondents get them right more
often, which usually means a mis-keyed answer. They are excluded from ranking
and selection. Check the key before dropping them: that is information pointing
the wrong way, not dead weight.
note: 378 of 600 items have a discrimination interval spanning zero — with this
many respondents the data cannot tell whether they separate anyone. They are
excluded from ranking and selection. More respondents is the fix.
next: irtcheck report suite.irt

$ irtcheck report suite.irt --limit 8
╭──────────────────────── cannot rank these items yet ─────────────────────────╮
│  Not enough respondents to rank these items: 378 of 600 items (63%) are      │
│  insufficient-data.                                                          │
│                                                                              │
│  Their discrimination interval spans zero, so this fit cannot tell whether   │
│  they separate stronger models from weaker ones. That is a statement about   │
│  the data supplied, not a verdict on the items.                              │
│                                                                              │
│  No items are flagged dead, which is expected rather than suspicious at      │
│  this respondent count. One item is flagged inverted: it discriminates       │
│  backwards, which usually means a mis-keyed answer.                          │
│                                                                              │
│  Cheapest next step — raise respondent count without running new models.     │
╰──────────────────────────────────────────────────────────────────────────────╯

      respondents  10 real models → 20 respondents (2.0 per model, --respondent-key model_id,prompt_variant)
            items  600 items  (190 usable for ranking and selection)
             dead  0 (0.0%)  confidently do not discriminate (the whole a interval lies inside ±0.35)
         inverted  1 (0.2%)  discriminate *backwards* — weaker respondents get these right more often
insufficient-data  378 (63.0%)  cannot tell — a interval spans zero; excluded from ranking and selection
  ceiling / floor  ceiling 22 (3.7%, p ≥ 0.99)   floor 16 (2.7%, p ≤ 0.01)
        off-range  0 (0.0%)  difficulty outside the ability range these respondents occupy

  item          a     a 95% HDI      b       b 95% HDI     n   p(correct)  flags
  item_00518  2.27   [0.83, 3.71]  -0.09  [-0.63, 0.45]   20        0.50
  item_00165  2.16   [0.75, 3.57]  -0.26  [-0.82, 0.29]   20        0.55
  ...

$ irtcheck report suite.irt --html suite.html
wrote suite.html          # self-contained, no external assets; the information curve is in here

$ irtcheck select suite.irt -n 50 -o anchor.json
50 of 190 usable items (600 total) · 10 models (20 respondents, key model_id+prompt_variant)
Mean ability standard error over these models: 0.220 (1.000 with no items at all).
Marginal gain: first item 0.6285, last item 0.3553. Where that flattens is where n stops buying precision.

$ irtcheck validate suite.irt
Leave-one-model-out · 10 models · 600 items · 20 respondents (key model_id+prompt_variant)
    n  items  places off  random  beats random  Spearman  random
   25     25        1.25    0.69            3%    +0.953  +0.946
   50     50        0.90    0.47            0%    +0.975  +0.973
  100    100        0.75    0.25            0%    +0.966  +0.989
  200    200        0.55    0.11            0%    +0.994  +0.996
  400    400        0.10    0.04           22%    +0.997  +0.998
At n=25, n=50, n=100, n=200, n=400, random item sets placed held-out models as well or better.
With 10 models two things are in play. The fit's item estimates are noisy — on synthetic data,
scored this way, choosing beat sampling only from about 50 models. And this table re-scores every
model on a set chosen from their own answers, which costs a chosen set more than a random one; the
ability table below does not, and on synthetic data that is the larger part. If you compare models
by plain accuracy on the set, drop the items `report` flags and draw the rest at random. Placed by
ability instead, the same sets beat most random draws at n=25 (81%), n=50 (73%), n=400 (56%).

Placed by ability — the same sets, scored the way you would use one
    n  places off  random  beats random
   25        0.50    0.69           81%
   50        0.30    0.41           73%
  100        0.30    0.28           41%
  200        0.20    0.15           31%
  400        0.10    0.11           56%
```

**Ten models is below where this tool can rank items, and it says so.** That is
the quickstart on purpose: 63% of the suite comes back "cannot tell", `report`
leads with a refusal instead of a table, and the 190 items that survive still
recover the full-suite ranking to +0.99 from 25 of them. A tool that printed a
confident table here would be making most of it up. (At n=200 and 400 the
held-out fits run out of confident items and `select` pads the rest — see
[`select`](#select).)

That last table is the headline, and it is not flattering: a held-out model,
placed among the others on a set chosen by a fit that never saw it, lands
closer to its full-suite place on a *random* set of the same size. Ten models
is too few to choose items from — the fit's estimates of which items are sharp
are too noisy — and on synthetic data choosing starts to pay at about fifty.
Read [Does it actually work?](#does-it-actually-work) before using an anchor set
as a stand-in for the suite.

Fitting is the slow part, so it is behind a cached artifact: `report`,
`select` and `validate` read `suite.irt` and are instant (`validate` refits, so
it is not — see below).

---

## Known limitations

Measured, not suspected. The first five come from a 336-fit study against
synthetic ground truth,
[`docs/validation.md` §4](https://github.com/antispiraldev/irtcheck/blob/master/docs/validation.md#4-how-many-respondents),
where the responses really were generated by a 2PL — the case most favourable
to this tool. On real suites expect each to be no better.

- **`report`'s refusal fires too readily, and on the wrong quantity.** It leads
  with *cannot rank these items yet* when more than half the items are
  `insufficient-data`. That share has a floor set by the suite as well as the
  data, so at 15–25 models the report refuses on fits whose 50-item anchor sets
  rank new models within 0.02 of the best possible set. Read the refusal as
  "most items are unreadable", not as "the anchor set is unusable" — `validate`
  answers that question directly.
- **Choosing items with IRT is no better than a spreadsheet, on complete
  data.** A set picked by classical item-rest correlation ranked fresh models
  as well as `select`'s did, within a few thousandths either way, at 15 to 100
  models with one response per model; with three prompt variants per model
  `select` led by up to 0.01. What the fit adds is the flags — `insufficient-data`,
  `inverted`, ceiling and floor — and keeping backwards items out of the set.
- **Below six real models the intervals are not 95% intervals.** They covered
  the true discrimination 40–72% of the time at three to five models with one
  response per model. The report only adds a caution below five.
- **`inverted` finds about half of mis-keyed items.** The fitter's
  parameterisation makes it hard for an item's slope to cross zero, so many
  backwards items stay positive and look merely weak — and with more models,
  some become usable or `dead`. Treat an `inverted` count as a floor, not a
  census. It is not yet fixed.
- **At low respondent counts, much of an anchor set can be padding.** When
  fewer items are readable than you ask for, `select` fills the set from
  `insufficient-data` items and says how many. This is on purpose: a short set
  ranked models worse than a random draw of the full size, and the padded one
  beat a random draw by 0.04 Spearman on average. It was level with classical
  item-rest correlation, not better, and a set that is mostly padding is only
  as good as the point estimates behind it.
- **Below about fifty models, a random set beats `select`'s — when every model
  is re-scored on the set.** On synthetic data, sets chosen from the fit placed
  held-out models worse than random sets at 10 and 25 models and clearly better
  from 50. Two things cause it, and the smaller one is the fit's noisy item
  estimates: given the true item parameters the same code wins at every count.
  The larger is the scoring itself, since the other models are re-scored on a
  set chosen from their own answers. Placing the held-out model by ability
  instead — `validate`'s second table — has choosing ahead at every model
  count, 81% of random draws at ten models against 3%.
- **On HELM Lite at 95 models, choosing still lost from 100 items up**, which
  the synthetic runs do not predict. The cause is not measured. `validate`
  prints the random baseline beside every row so you can see which way your
  suite goes. See [Does it actually work?](#does-it-actually-work).
- **`--adaptive` is declared and not implemented**, and exits saying so.
- **`pip install irtcheck` on Linux pulls PyPI's default `torch`**, which
  bundles CUDA and is a download of several gigabytes. If you do not have a GPU,
  install the CPU wheel first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

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

## `dead`, `inverted` and `insufficient-data` are three different claims

This is the heart of the tool, so it is worth being precise. All three appear
on items that are not earning their place, and they mean quite different
things. They are three readings of one 95% interval on `a`:

**`dead`** — we are confident the item does not discriminate. The *whole*
interval lies inside `±0.35`, so `|a|` is confidently negligible. Whatever
ability a respondent has, this item barely changes its mind. **This is a
finding about your suite**: the item is dead weight and you can drop it.

**`inverted`** — we are confident the item discriminates *backwards*. The
interval lies wholly below zero and is not negligible, so **weaker** models get
it right more often than stronger ones. **This is a finding about the item**,
and it is the opposite of dead weight: the item carries real information,
pointing the wrong way. The usual cause is a mis-keyed answer or a question
whose intended answer is wrong. Check the key before dropping it.

**`insufficient-data`** — we cannot tell. The interval on `a` spans zero, so
the fit cannot distinguish "separates strong models from weak ones" from
"separates weak from strong" from "does nothing". **This is a finding about the
data you gave us.** Such items are excluded from ranking and from selection
rather than ranked anyway, and the report points at adding respondents.

No item carries two of them; `IrtFit.validate()` rejects an artifact that
claims more than one about anything.

`inverted` and `insufficient-data` items are excluded from anchor-set
selection; `dead` items are not, and the asymmetry is deliberate. Selection
maximises information, information goes as `a²`, and a dead item's `a` is
confidently near zero — so it is never picked anyway. An inverted item's `|a|`
is *large*, so selection finds it attractive; before these were separated, an
anchor set of 400 on the real matrix below picked every inverted item the suite
had. An anchor set is scored by plain accuracy, and an item a stronger model
reliably gets wrong subtracts from exactly the signal you wanted.

**At five to fifteen respondents, most low-information items land in
`insufficient-data` and `dead` is rare.** In the quickstart above, 20
respondents gave 378 insufficient-data and 0 dead out of 600 items. That is not
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

**Read this before using an anchor set as a smaller stand-in for the suite.**
Choosing the most informative items needs good estimates of which items are
informative, and with few models the fit does not have them: it under-reads
the sharpest items and over-reads some ordinary ones. On synthetic data,
`select`'s sets placed models *worse* than random sets of the same size at 10
and 25 models, and clearly better from 50
([`docs/validation.md` §5](https://github.com/antispiraldev/irtcheck/blob/master/docs/validation.md#5-choosing-items-against-sampling-them)).
On the one real suite measured, HELM Lite, they lost at 95 models too, from
100 items up, for reasons not yet measured.

So below about fifty models, use the fit for what it can say — `dead`,
`inverted`, ceiling and floor, which items to remove — and draw the rest at
random. With more, `select` can beat random; run `validate` on your own suite,
which prints the random baseline beside every row, before relying on it.

Emits the N most informative items as JSON. Item information for a 2PL is
`I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))`, and **the integral is
taken against the posterior ability means of the respondents in your matrix**,
not against a textbook N(0, 1) grid. That is the difference between "is this a
good test?" in the abstract and "does this test measure the models I actually
have?", and the two answers diverge whenever a suite was built for a generation
of models that has since been outgrown.

Items flagged `insufficient-data`, `inverted`, `ceiling` or `floor`, and items
with no responses in this fit, are not eligible. `dead` items stay eligible and
are simply never worth picking — their information is near zero by definition —
which is a property that falls out rather than one that had to be special-cased.

**Asking for more items than are eligible pads the set, and says so.** The
readable items come first; the rest are filled, by the same objective, from
`insufficient-data` items — never from `inverted`, ceiling, floor or unanswered
items, and never from an item whose fitted slope is negative. The summary line
reads `50 items (12 confident, 38 padded)`, and the JSON carries `"padded": 38`,
counted from the end of `item_ids`. Early versions returned the short set
instead, on the argument that padding would undo the refusal; measured, a short
set ranked models worse than a random draw of the full size.
[`docs/validation.md` §4](https://github.com/antispiraldev/irtcheck/blob/master/docs/validation.md#a-short-anchor-set-is-worse-than-a-random-one)
has the numbers. If you want only the confident items, take the first
`count - padded` ids.

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
what remains, select an anchor set of size *n* from *that* fit, **score every
model on that one set** by plain accuracy — which is what you do with an anchor
set — and measure how many places *k* lands from its full-suite place. That is
the headline, *places off*, and 0 is exact.

**Beside every row is the same measurement for 200 random sets of the same
size**, and the share of those draws the anchor set beat. A number with nothing
to compare it to hid the most important result on this page: on real data,
random sets won. When they do, `validate` says so in yellow. The draws are
seeded, so one artifact and one set of flags give the same output. A Spearman
between held-out and full-suite places is printed too, with its own random
column.

**The same sets are placed a second way, by ability.** Scoring every model on
the anchor set has a cost: the set was chosen from the other models' answers, so
those models are spread out by their own noise while the held-out model is not,
which pulls it toward the middle. The second table avoids that — estimate the
held-out model's ability from *its own* answers to the set, using the item
parameters of the fit that chose it, and place it among the other models'
abilities in that same fit, which came from the whole suite and are not
recomputed. On synthetic data the difference is large: at ten models, sets that
beat 3% of random draws when every model is re-scored beat 81% placed this way.
Which table to read is a question about how you will use the set. If you will
compare a new model's accuracy on it against the accuracy of models you already
ran, the first one is your case.

**Holding out a model holds out every pseudo-respondent derived from it.** With
`--respondent-key model_id,prompt_variant` one model is several respondents;
dropping one row would leak that model's other variants into the fit that
chooses the anchor set, and the correlation would come out inflated with nothing
anywhere to catch it.

Earlier versions of `validate` scored each held-out model on *its own* anchor set and
correlated those scores. Different holdouts choose different sets — on the
twelve-model HELM matrix, twelve 100-item sets shared 11 items and ranged from
0.49 to 0.77 mean accuracy — so that number mixed "does the set rank models"
with "how hard did this holdout's set happen to be". The re-estimated-ability
columns existed to correct for that, and went with it.

---

## Does it actually work?

Partly, and the parts matter. The per-item analysis holds up on real data.
Choosing an anchor set beats drawing one at random only with enough models —
about fifty on synthetic data — and on the one real suite measured it lost even
at ninety-five. All three answers below are measured, and the least flattering one is
last rather than omitted.

### On synthetic data: yes, and that only proves the pipeline is correct

The numbers in the quickstart are real measurements, on a matrix generated from
a known 2PL by `irtcheck.synth` — 10 models × 2 prompt variants, 600 items,
seed 5:

| n   | places off | random | beats random | Spearman | random  |
| --- | ---------- | ------ | ------------ | -------- | ------- |
| 25  | 1.25       | 0.69   | 3%           | +0.953   | +0.946  |
| 50  | 0.90       | 0.47   | 0%           | +0.975   | +0.973  |
| 100 | 0.75       | 0.25   | 0%           | +0.966   | +0.989  |
| 200 | 0.55       | 0.11   | 0%           | +0.994   | +0.996  |
| 400 | 0.10       | 0.04   | 22%          | +0.997   | +0.998  |

Out of 10 places. The held-out models land close to their full-suite place
either way, and the Spearman columns are near 1 for both, but at every size a
random set lands them closer. That is the model count, not the code: the same
selection given the true item parameters places every held-out model exactly
from 50 items, and with fifty or more models the fitted version beats random
too. (From n=200 up, the held-out fits run out of confident items and `select`
pads the rest from `insufficient-data` items.)

**This is synthetic data, and the items really do come from a 2PL because we
drew them from one.** Real eval items do not. So this table is a correctness
check on the pipeline — the fitter recovers parameters, leave-one-model-out is
wired up without leakage, and selection needs more than ten models — and it is
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
| usable for ranking  | 568 (16.0%)                  | 190 (31.7%)                           |
| `insufficient-data` | 2,980 (83.9%)                | 378 (63.0%)                           |
| `dead`              | 0                            | 0                                     |
| `inverted`          | 3 (0.08%)                    | 1 (0.2%)                              |
| `ceiling` / `floor` | 3.8% / 4.3%                  | 3.7% / 2.7%                           |

**The small-N behaviour this README describes is not an artefact of how
`synth.py` draws parameters** — it is what real model responses do too, with the
real matrix somewhat harsher than the synthetic one at comparable respondent
count. Both columns were re-measured after the interval fix below, from a matrix
rebuilt with the scripts in [`scripts/`](https://github.com/antispiraldev/irtcheck/tree/master/scripts). `ceiling` came out at 3.8%
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
| `dead`             | 0             | 0             |
| `inverted`         | 3 (0.08%)     | 32 (0.90%)    |

Nothing about the items changed. The only thing that changed was how much
evidence there was about each one. "We cannot tell" became "we can tell" for
two thirds of the suite. That is `insufficient-data` being a statement about
your data, demonstrated rather than asserted.

The `dead` row is where the `inverted` flag came from. Under the original rule
every one of those 32 items counted as `dead` — because its interval on `a`
lies wholly *below* zero, meaning weaker models get it right more often, not
because it fails to discriminate. Split apart, this matrix has **no** `dead`
items at either respondent count and 32 `inverted` ones.
[`docs/validation.md`](https://github.com/antispiraldev/irtcheck/blob/master/docs/validation.md) §2e has the detail.

### On real data: choosing items lost to sampling them

This is the least flattering measurement here and the one most worth reading.
`irtcheck validate` on the same HELM matrix, with every model scored on each
held-out model's anchor set and 200 random sets of the same size alongside.
Places off, lower is better, and the share of random draws each selector beat:

| n   | 12 models: random | `select`    | 95 models: random | `select`    |
| --- | ----------------- | ----------- | ----------------- | ----------- |
| 25  | 1.34              | 1.88 · 9%   | 12.16             | 9.75 · 84%  |
| 50  | 0.92              | 2.00 · 0%   | 9.35              | 9.88 · 38%  |
| 100 | 0.65              | 1.96 · 0%   | 6.97              | 10.83 · 0%  |
| 200 | 0.38              | 1.92 · 0%   | 5.06              | 9.15 · 0%   |
| 400 | 0.19              | 2.08 · 0%   | 3.50              | 8.40 · 0%   |

(All twelve models held out in the first pair of columns; twenty-four, spread
across the accuracy ranking, out of 95 in the second.)

**From 100 items up, random sets placed held-out models better, at both model
counts.** The one win for choosing is 95 models and 25 items. And the chosen
sets stop improving while random ones do not: `select` sits near two places
off at every size at twelve models while random falls from 1.34 to 0.19 —
error that does not shrink as items are added is a bias, not noise.

**At twelve models this is expected.** Synthetic data, where the answer is
known, loses the same way at ten and twenty-five models, and for a measured
reason: the fit's item estimates are too noisy to choose from. Classical
item-rest correlation, which estimates from the same few models without the
2PL, does no better.

**At ninety-five it is not.** Synthetic data at a hundred models has `select`
winning clearly up to 200 items; here it lost from 100. Something about the
real suite costs selection at that count, and it is not measured. HELM Lite is
maths, law and general knowledge while a 2PL has one dimension, which makes
that a candidate, alongside guessing floors and label noise. Splitting the
budget across the eighteen scenarios by size did not help at that count: 8.52
places off at n=400, against `select`'s 8.40 and random's 3.50.

This does not touch `report`: the flags, the `insufficient-data`/`dead`/
`inverted` distinction and the per-scenario findings above are not scored by
this and behaved as designed on real data. It does change what `select` is for
— see [`select`](#select). **And it means `validate` is doing its job**: the
old protocol, with no random column, reported these same anchor sets as
reasonable.

Every holdout, all four selectors, the Spearman view and the old protocol's
numbers for comparison are in
[`docs/validation.md` §5](https://github.com/antispiraldev/irtcheck/blob/master/docs/validation.md#5-choosing-items-against-sampling-them),
and [`studies/helm_selection/`](https://github.com/antispiraldev/irtcheck/tree/master/studies/helm_selection)
reproduces them.

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
from different angles. See [`CLAUDE.md`](https://github.com/antispiraldev/irtcheck/blob/master/CLAUDE.md) for that and the rest of the
constraints that bite, [`docs/spec.md`](https://github.com/antispiraldev/irtcheck/blob/master/docs/spec.md) for the design, and
[`docs/build-plan.html`](https://github.com/antispiraldev/irtcheck/blob/master/docs/build-plan.html) for how it was built.

## Licence

MIT — see [`LICENSE`](https://github.com/antispiraldev/irtcheck/blob/master/LICENSE).
