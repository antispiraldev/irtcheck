# Validation

What we have actually measured, how, and what each number is and is not
evidence for. Every figure on this page was produced by running the shipped
CLI; nothing here is estimated or carried over from a paper.

The short version, good news and bad news together:

- **On synthetic data the pipeline is correct.** A 25-item anchor set chosen by
  a fit that never saw the held-out model reproduces a ten-model ranking at
  Spearman +0.994. This is a check that the code does what it says, because the
  items really were drawn from a 2PL. It is not a claim about real suites.
- **A real per-item response matrix exists and is public**, and the tool runs
  on it: HELM Lite's released per-instance predictions, 95 models × 3,551
  items, no authentication needed.
- **On that real matrix the item-level machinery holds up.** The flag
  distribution at twelve models lands almost exactly where the synthetic one
  does, the `insufficient-data`-collapses-with-respondents claim is confirmed
  going from 12 to 95 models, and the per-scenario output produces a finding
  synthetic data could not have — including a scoring artefact that makes four
  strong models score below chance.
- **On that real matrix the anchor-set claim is much weaker.**
  Leave-one-model-out gives Spearman **+0.662 at n=25** and **+0.867 at n=400**,
  against +0.994 and +1.000 synthetic. Twenty-five items do not reproduce the
  ranking here. This is the most important number on the page and the reason
  the synthetic table is labelled as carefully as it is.
- **We have not cross-checked the fitter against `mirt` or `py-irt`.** That is
  a separate, unfinished exercise, and nothing here should be read as agreeing
  with either.

---

## 1. Synthetic: does the pipeline do what it says?

`irtcheck.synth` draws item and respondent parameters from a known 2PL and
samples a response matrix from them, so ground truth exists. The configuration
below is 10 models × 2 prompt variants (20 respondents), 600 items, seed 5.

```python
# /tmp/gen_synth.py
import json

from irtcheck.synth import make_truth, responses_from_truth, synthetic_records

truth = make_truth(n_models=10, n_items=600, variants_per_model=2, seed=5)
records = synthetic_records(truth, responses_from_truth(truth, seed=6))
with open("synth_responses.jsonl", "w") as handle:
    for r in records:
        row = {"model_id": r.model_id, "item_id": r.item_id, "correct": r.correct}
        row.update({k: v for k, v in r.extra.items() if v is not None})
        handle.write(json.dumps(row) + "\n")
```

```
irtcheck fit synth_responses.jsonl --respondent-key model_id,prompt_variant -o synth.irt
irtcheck validate synth.irt
```

The fit: **9.7 s** on CPU (32-core x86-64, torch CPU wheel), 12,000 responses,
100% dense.

| n   | items | Spearman | Kendall tau | Spearman (theta) | tau (theta) |
| --- | ----- | -------- | ----------- | ---------------- | ----------- |
| 25  | 25    | +0.994   | +0.978      | +0.988           | +0.956      |
| 50  | 50    | +0.988   | +0.956      | +0.976           | +0.911      |
| 100 | 100   | +1.000   | +1.000      | +1.000           | +1.000      |
| 200 | 200   | +1.000   | +1.000      | +1.000           | +1.000      |
| 400 | 252\* | +1.000   | +1.000      | +1.000           | +1.000      |

\* fewer than 400 items were eligible: the rest are flagged
`insufficient-data`, `ceiling` or `floor` in the held-out fits.

n=50 scoring marginally below n=25 is not a bug and not worth reading into.
With ten models a single swapped adjacent pair moves Spearman by about 0.012,
which is the entire difference between those two rows. The trend across n is
the signal; any one cell is one swapped pair wide.

### What this does and does not establish

**It establishes** that the fitter recovers structure, that selection picks
items carrying information about the respondents present, and that
leave-one-model-out is wired up without leakage — a held-out model's prompt
variants really are dropped with it, via `ResponseMatrix.drop_model()` and
`derives_from`. Break any of those and this table collapses.

**It does not establish** that IRT describes real eval suites. The items here
were generated from a 2PL, so a 2PL fits them by construction. Real items have
guessing floors, correlated content, contamination and multidimensional
structure, none of which this matrix contains. Treat the table as a
regression test on the implementation, not as a claim about your suite.

The header from the same fit is worth showing, because it is the honest
small-N picture:

```
      respondents  10 real models → 20 respondents (2.0 per model)
            items  600 items  (306 usable for ranking and selection)
             dead  3 (0.5%)
insufficient-data  256 (42.7%)
  ceiling / floor  ceiling 22 (3.7%)   floor 16 (2.7%)
```

43% of items unrankable at 20 respondents, and almost nothing reaching `dead`.
That is the expected result, for the reason the README gives: `dead` needs an
interval narrow enough to sit wholly below 0.35.

---

## 2. Real data: HELM Lite's per-instance predictions

The riskiest assumption in this project was that a parseable per-item response
record exists in the wild for more than a handful of models. It does, and it is
public, unauthenticated, and large.

**Source.** HELM publishes its full benchmark output to a public Google Cloud
Storage bucket, `gs://crfm-helm-public`. Each run directory
`lite/benchmark_output/runs/<version>/<scenario>,model=<model>/` contains a
`display_predictions.json`, a list of per-instance records:

```json
{"instance_id": "id4957", "predicted_text": "B", "mapped_output": "quit eating lunch out",
 "stats": {"num_prompt_tokens": 314.0, "exact_match": 1.0}}
```

`instance_id` is stable for a scenario across models, and `stats` carries a
binary per-instance metric. That is exactly irtcheck's input contract with the
field names changed. No authentication and no HuggingFace token is needed; the
bucket's object listing API is open.

**Note on the obvious alternative.** The Open LLM Leaderboard v2 `-details`
datasets on HuggingFace (`open-llm-leaderboard/<org>__<model>-details`) are
lm-eval `samples_*` logs for ~90 tasks and hundreds of models — the ideal
input, since irtcheck has an lm-eval adapter. The repository listing API is
open but **file downloads are gated**: `resolve/main/...` returns *"Access to
dataset ... is restricted. You must have access to it and be authenticated"*.
They are usable with an accepted licence and a token, and not usable for an
unauthenticated reproduction, which is why HELM was used instead.

**What we built.** 18 HELM Lite scenarios (5 MMLU subjects, OpenBookQA, 7 MATH
level-1 subjects, 5 LegalBench subsets), collected across all 14 published
`lite` versions and deduplicated to the latest run per (scenario, model):

```
95 models × 3,551 items, 323,656 responses, 0 read errors
metrics used: exact_match, quasi_exact_match, math_equiv_chain_of_thought
every item answered by between 87 and 95 of the 95 models
```

The matrix is unusually dense for real data because HELM runs every model on
every scenario it publishes.

### 2a. Twelve models — the case irtcheck is built for

The 95-model matrix is not the situation the tool is designed around. So we cut
it to twelve models spaced evenly across the observed accuracy range, keeping
only items all twelve answered — a dense 12 × 3,551 rectangle, which is what a
team comparing a shortlist actually has:

| full-suite accuracy | model                                    |
| ------------------- | ---------------------------------------- |
| 0.292               | `tiiuae_falcon-7b`                       |
| 0.428               | `anthropic_claude-3-haiku-20240307`      |
| 0.471               | `meta_llama-3.2-11b-vision-instruct-turbo` |
| 0.513               | `cohere_command-r`                       |
| 0.568               | `mistralai_mistral-7b-v0.1`              |
| 0.610               | `qwen_qwen1.5-7b`                        |
| 0.633               | `microsoft_phi-3-small-8k-instruct`      |
| 0.650               | `google_gemini-2.0-flash-exp`            |
| 0.681               | `meta_llama-3.2-90b-vision-instruct-turbo` |
| 0.694               | `meta_llama-3.1-405b-instruct-turbo`     |
| 0.731               | `meta_llama-3-70b`                       |
| 0.778               | `google_gemini-1.5-pro-002`              |

`irtcheck fit` on this took **29.3 s** (42,612 responses, 2,000 SVI steps,
CPU). The report header:

| | 12 models × 3,551 items | synthetic, 20 respondents × 600 items |
| --- | --- | --- |
| usable for ranking  | 1,664 (46.9%) | 306 (51.0%) |
| `insufficient-data` | 1,602 (45.1%) | 256 (42.7%) |
| `dead`              | 29 (0.8%)     | 3 (0.5%)      |
| `ceiling`           | 3.8%          | 3.7%          |
| `floor`             | 4.3%          | 2.7%          |

**The real matrix behaves like the synthetic one at comparable respondent
count.** Slightly under half the items unrankable, `dead` under one per cent,
single-digit ceiling and floor rates. `report` raises a caution rather than a
refusal (45% is below the 50% refusal threshold) and points at
`--respondent-key`.

That correspondence is the most useful thing on this page. The small-N
behaviour the README describes — most weak items landing in
`insufficient-data`, `dead` staying rare — is not an artefact of how `synth.py`
draws parameters. It is what happens on real model responses too.

### 2b. What the real data says that synthetic data could not

Grouping the per-item output by HELM scenario gives a signal ranking over real
benchmarks. `n` is items, `usable` is items carrying no flag, and mean `a` is
over the usable ones:

| scenario                                         |    n | usable | ins-data | dead | mean `a` |
| ------------------------------------------------ | ---: | -----: | -------: | ---: | -------: |
| `openbookqa`                                     |  500 |    361 |       66 |    2 |     1.64 |
| `mmlu_computer_security`                         |  111 |     56 |       30 |    1 |     1.68 |
| `math_number_theory`                             |   30 |     13 |       17 |    0 |     1.55 |
| `mmlu_abstract_algebra`                          |  111 |     38 |       66 |    2 |     1.51 |
| `mmlu_college_chemistry`                         |  108 |     51 |       42 |    3 |     1.51 |
| `math_precalculus`                               |   57 |     27 |       30 |    0 |     1.51 |
| `mmlu_us_foreign_policy`                         |  111 |     76 |       17 |    0 |     1.47 |
| `math_counting_and_probability`                  |   39 |     25 |       13 |    0 |     1.48 |
| `math_intermediate_algebra`                      |   52 |     24 |       26 |    0 |     1.46 |
| `legalbench_function_of_decision_section`        |  367 |    152 |      109 |    3 |     1.45 |
| `math_algebra`                                   |  135 |    108 |       27 |    0 |     1.44 |
| `mmlu_econometrics`                              |  126 |     62 |       56 |    4 |     1.44 |
| `math_prealgebra`                                |   86 |     60 |       24 |    1 |     1.43 |
| `math_geometry`                                  |   38 |     15 |       21 |    0 |     1.40 |
| `legalbench_abercrombie`                         |   95 |     40 |       48 |    4 |     1.38 |
| `legalbench_proa`                                |   95 |     62 |       15 |    0 |     1.37 |
| `legalbench_corporate_lobbying`                  |  490 |    210 |      276 |    2 |     1.36 |
| `legalbench_international_citizenship_questions` | 1000 |    255 |      719 |    7 |     1.09 |

The finding the tool exists to produce is in the last row.
`legalbench_international_citizenship_questions` is the **largest** scenario in
this matrix — 1,000 items, 28% of it — and the **weakest**: 72% of its items
are unrankable at twelve models and its usable items have the lowest mean
discrimination of any scenario here. OpenBookQA is half its size and yields
361 usable items to its 255.

An aggregate score over this item set is dominated, by item count, by the
scenario carrying the least measurement signal per item. **That is the claim
irtcheck was written to be able to make, and it needed real data to make it** —
a synthetic matrix has no scenarios and no editorial history, so nothing about
`synth.py` could have surfaced it.

**A cross-check that does not use IRT at all agrees.** For each scenario,
Spearman between the twelve models' accuracy *on that scenario* and their
accuracy on the full item set asks the same question the discrimination
estimate does, by much cruder means:

| scenario                                         | rho  |
| ------------------------------------------------ | ---- |
| `mmlu_college_chemistry`                         | 0.88 |
| `openbookqa`                                     | 0.85 |
| `legalbench_function_of_decision_section`        | 0.86 |
| …                                                |      |
| `legalbench_corporate_lobbying`                  | 0.64 |
| `legalbench_international_citizenship_questions` | 0.45 |
| `mmlu_us_foreign_policy`                         | 0.43 |

The citizenship scenario is second-lowest on this measure too, and
`corporate_lobbying` — the other large, weak LegalBench subset — is
third-lowest. Two independent methods picking out the same two scenarios is
worth more than either alone.

**And the two measures disagree about `mmlu_us_foreign_policy`**, which has the
lowest rho (0.43) but a high mean `a` (1.47) and only 17 of 111 items
unrankable. With twelve models a scenario-level rank correlation is a handful of
swapped pairs wide, and this scenario is easy enough that accuracy differences
between the stronger models are small; the item-level estimate has 111 items to
work with and the scenario-level one has twelve points. The disagreement is
recorded rather than resolved, because resolving it would need more models.

### An artefact the tool surfaced by accident

On the citizenship scenario the median across all 89 models that ran it is
**0.536** — chance, for a two-way question. Four models sit far *below* chance:

```
0.034  anthropic_claude-3-haiku-20240307
0.066  mistralai_mistral-medium-2312
0.078  google_gemini-1.5-pro-preview-0409
0.099  mistralai_mistral-small-2402
```

Scoring 0.034 where guessing gives 0.5 is not a knowledge result. It is a
systematic answer-format or answer-extraction failure — the model is reliably
producing something the scorer reads as the wrong label. Three of those four are
otherwise strong models, so an aggregate leaderboard over this scenario ranks
them below models that answered at random.

This is a contributing cause of the weak discrimination above, and it is the
kind of thing per-item analysis finds and an aggregate score cannot: the signal
is in *which* responses are wrong, not in how many.

It is also a caution about reading section 2b as a verdict on LegalBench.
"Discriminates poorly among these twelve models, partly because the scorer
mislabels four of them" is a much narrower claim than "is a bad benchmark", and
these are 2024-era HELM Lite runs besides.

### 2c. The headline claim on real data, and it is much weaker

This is the number that matters, and it is the least flattering thing on this
page. `irtcheck validate` on the same twelve-model matrix — twelve holdout
refits, an anchor set chosen each time by a fit that never saw the held-out
model:

| n   | items | Spearman | Kendall tau | Spearman (theta) | tau (theta) |
| --- | ----- | -------- | ----------- | ---------------- | ----------- |
| 25  | 25    | +0.662   | +0.504      | +0.797           | +0.606      |
| 50  | 50    | +0.775   | +0.585      | +0.853           | +0.667      |
| 100 | 100   | +0.629   | +0.455      | +0.804           | +0.606      |
| 200 | 200   | +0.830   | +0.687      | +0.902           | +0.758      |
| 400 | 400   | +0.867   | +0.727      | +0.888           | +0.758      |

Against the synthetic matrix's +0.994 at n=25 and +1.000 from n=100. **On this
real matrix, a 25-item anchor set does not reproduce the ranking**, and 400
items get to +0.867 — respectable, but nothing like the synthetic result, and
not the "1% of the items, 2% error" figure that circulates about IRT-selected
benchmark subsets.

Three things about this table are worth stating plainly rather than explaining
away.

**The synthetic number was measuring the implementation, not the method.** That
was said before this was run, and this is what it looks like when it turns out
to have been the right caveat. Anyone quoting +0.994 as evidence that small
anchor sets work on eval suites would have been wrong, and the gap between the
two tables is the reason the README labels the synthetic one twice.

**It is not monotone in n.** n=100 (+0.629) scores below n=50 (+0.775). With
twelve models a single adjacent swap moves Spearman by only about 0.007, so a
0.15 drop is roughly twenty rank-units of churn and not one unlucky pair — the
selected sets at those two sizes genuinely differ in how well they order these
models. Read the trend, not any row.

**The `theta` columns beat the accuracy columns at every size**, by 0.06 to 0.18
Spearman. `validate`'s own documentation says those two diverge when an anchor
set skews hard or easy, so that gap is the tool reporting that its anchor sets
are mis-centred for these respondents. Re-estimating ability with the item
parameters held fixed corrects for it; plain accuracy over the anchor items,
which is what a user would actually do, does not.

#### Why, as far as we can tell

Stated as hypotheses, because that is what they are. None of the four is
measured here; two of them could be, on this same data, and §2d says how.

- **The matrix is not unidimensional, and a 2PL assumes it is.** One `theta`
  per model cannot represent a model that is good at competition maths and bad
  at statutory interpretation, and this matrix spans MATH, LegalBench, MMLU and
  OpenBookQA. Multidimensional IRT is an explicit non-goal in `docs/spec.md`,
  so this is a known limit being hit, not a surprise. **Testable on this data:**
  see [§2d](#2d-two-of-those-hypotheses-are-testable-on-this-data).
- **Twelve models is thin for a rank correlation.** `validate` warns below six;
  twelve is not a lot more. **Testable on this data:** see [§2d](#2d-two-of-those-hypotheses-are-testable-on-this-data).
- **Some responses are mislabelled.** Four models score far below chance on the
  citizenship scenario (above), which corrupts their ability estimates and
  therefore every item parameter fitted alongside them. Not separately
  quantified.
- **Guessing.** These are mostly multiple-choice items with a floor near 0.25,
  which a 2PL has no parameter for. A 3PL adds one and needs more data than
  twelve respondents provide — which is why `docs/spec.md` does not make it the
  default.

#### What this means for the tool

It does not invalidate `report` or `select`: the per-item flags, the
`insufficient-data`/`dead` distinction, and the per-scenario finding in §2b all
hold up and are the parts that behaved as designed on real data. It does mean
**the anchor-set claim needs stating with an honest n**. On this matrix, at
twelve models, a few hundred items reproduce the ranking usefully and
twenty-five do not.

It also means `validate` is doing its job. A tool that reported +0.99 here
would be broken; this one reported +0.66 and flagged, through the theta
columns, why.

### 2d. Two of those hypotheses are testable on this data

Both are cheap to run and neither was finished in time for this write-up, so
they are recorded here as the next two measurements rather than as results. The
scripts are trivial given `helm_responses.jsonl` from §2's reproduction notes.

**Is the 2PL misspecified because the matrix is multidimensional?** Restrict to
one kind of item — the five MMLU subjects plus OpenBookQA, 1,052 items, all
multiple-choice knowledge questions rather than a mix of maths, statutory
interpretation and commonsense — keep the same twelve models, and re-run
`validate`. If rank recovery improves materially, unidimensionality is the
binding problem and the answer is to fit per domain rather than across a mixed
suite. If it does not, the 2PL is not the limit here.

**Is twelve respondents simply too few?** Take 25 models instead of 12 over the
same 3,551 items and re-run. `validate` warns below six models; twelve is not
much more, and respondent count binds everything else in this tool. The 95-model
result in §2e shows what respondents do to *item* estimates; this would show
what they do to *rank recovery*, which is a different question and the one the
headline claim depends on.

Between them these separate "IRT does not describe this suite" from "we did not
give it enough models", which is the single most useful thing left to know
about this data. Until they are run, §2c's causes stay hypotheses.

### 2e. Ninety-five models — `dead` becomes reachable

Fitting all 95 models is outside the tool's design range and is interesting for
one reason: CLAUDE.md claims `dead` needs "roughly a hundred respondents"
before an interval can sit wholly below the threshold. 95 real models is the
first chance to check that against real responses rather than a generated
matrix. The fit takes **23.5 s** for 323,656 responses (96% dense).

| items out of 3,551 | 12 models   | 95 models   |
| ------------------ | ----------- | ----------- |
| usable for ranking | 1,664 (47%) | 3,272 (92%) |
| `insufficient-data`| 1,602 (45%) | 225 (6.3%)  |
| `dead`             | 29 (0.8%)   | 94 (2.6%)   |
| `ceiling`          | 3.8%        | 0.0%        |
| `floor`            | 4.3%        | 1.5%        |
| `report` verdict   | caution     | none        |

**The claim holds, measured on real data.** Going from 12 to 95 respondents on
the *same 3,551 items* collapses `insufficient-data` by a factor of seven and
roughly triples `dead`. Nothing about the items changed; the only thing that
changed was how much evidence there was about each one. "We cannot tell"
becomes "we can tell", and about 2.6% of the suite turns out to be genuinely
dead weight — a claim that was simply not available at twelve models.

This is the strongest argument for `--respondent-key` in the documentation, and
it is also the clearest demonstration that `insufficient-data` is a statement
about the data rather than about the items.

### Reproducing section 2

The scripts are not shipped in the package — they fetch from the network, which
the tool itself never does at runtime — but they are short. `enum_helm.py`
lists every `lite` run prefix via the bucket's JSON API and writes
`helm_runs.json`; `fetch_helm.py` pulls one `display_predictions.json` per
(scenario, model) with a 24-way thread pool and emits the JSONL;
`subsample_helm.py` cuts it to twelve models. The cost is 1,673 HTTP requests
and a few hundred megabytes of JSON, which is where the wall time goes.

The whole of it is:

```
curl "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o?prefix=lite/benchmark_output/runs/&delimiter=/"
curl "https://storage.googleapis.com/crfm-helm-public/<run prefix>/display_predictions.json"
```

and then, per record, `{"model_id": <model>, "item_id": "<scenario>_<instance_id>",
"correct": <stats[metric]>, "subject": <scenario>}`.

---

## 3. What we have not measured

- **No comparison against `mirt` or `py-irt`.** Our 2PL is written directly in
  Pyro because `py-irt` cannot be a dependency on 3.12+ (see CLAUDE.md). A
  one-time comparison against R's `mirt` and against `py-irt` on a 3.11 venv is
  planned in `crosscheck/` and is **not finished**. Nothing on this page or in
  the README should be read as "agrees with mirt". When it lands, watch for sign
  and scale convention differences between packages; the identification choice
  is recorded in each artifact's `model.identification` field for that reason.
- **No real-data leave-one-model-out beyond twelve and twenty-five models.**
  The full 95-model sweep is 95 refits and was not run. Whether rank recovery
  keeps improving past twenty-five respondents is therefore open, and it is the
  single most useful thing anyone could measure next on this data.
- **None of the causes of the §2c shortfall is measured.** All four are
  hypotheses. Two of them are testable on exactly this data and §2d says how;
  they are the next thing to run.
- **No timings beyond the three above** (9.7 s for 20 × 600, 29.3 s for
  12 × 3,551, 23.5 s for 95 × 3,551, all CPU, 2,000 SVI steps). They will not
  extrapolate cleanly: cost scales with observed responses and with item count,
  and SVI step count is a flag.
- **No claim about how many respondents you need.** The fitter's recovery curve
  against respondent count was measured during wave 1 and is recorded in the
  commit that added it (`git log` for "the 2PL, its posterior, and a
  discrimination that can be refused"): 0.50 at 15 respondents, 0.75 at 60,
  0.87 at 150, 0.93 at 300, for `a` at 500 items. That is synthetic too.
