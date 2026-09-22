# Does choosing items beat sampling them?

**Not shipped.** A one-time study on real data, excluded from the sdist like
`studies/min_respondents/`. The write-up, and what it changed, is
[`docs/validation.md` §5](../../docs/validation.md#5-choosing-items-against-sampling-them).

The question: on HELM Lite, does an anchor set chosen by the fit place a model
it never saw better than a random set of the same size? It runs through
`irtcheck.validate.leave_one_model_out` itself, so each column is what
`irtcheck validate` prints for that way of choosing items.

    # the two artifacts: irtcheck fit on the matrices scripts/README.md rebuilds
    irtcheck fit helm_12.jsonl        -o helm12.irt
    irtcheck fit helm_responses.jsonl -o helm95.irt

    # ~3 minutes, twelve holdout fits
    PYTHONPATH=src .venv/bin/python studies/helm_selection/run.py \
        --fit helm12.irt --cache ~/.cache/irtcheck-helm12 > studies/helm_selection/helm12.txt

    # ~25 minutes on three workers, twenty-four holdout fits spread across the ranking
    PYTHONPATH=src .venv/bin/python studies/helm_selection/run.py \
        --fit helm95.irt --cache ~/.cache/irtcheck-helm95 --holdouts 24 --workers 3 \
        > studies/helm_selection/helm95.txt

## What is compared

| selector    | how it chooses                                                            |
| ----------- | ------------------------------------------------------------------------- |
| `tool`      | `select_item_ids(fit, n)` — what `irtcheck select` ships                   |
| `per-scen`  | n split across the 18 scenarios by size, most informative within each     |
| `item-rest` | top n by item-rest correlation over the held-out matrix — no IRT at all   |
| `random`    | uniform draw from the items the held-out fit has responses for, 200 sets  |

**Places off** is the headline: choose the set without model k, score every
model on it by plain accuracy, and measure how far k lands from its full-suite
place. The last table is the protocol `validate` used before this study — each
held-out model scored on its own set — kept so the old numbers can be traced.

## One subject at a time

The full suite at 95 models has `select` losing from 100 items up, where
synthetic data says it should win. The candidate explanation was that HELM Lite
measures several abilities and a 2PL assumes one. `slices.py` cuts the three
largest single-subject groups out of the same matrix, and each is fitted and
run exactly as above:

    python studies/helm_selection/slices.py -i helm_responses.jsonl
    irtcheck fit slice_mmlu.jsonl -o slice_mmlu.irt      # likewise openbookqa, citizenship
    PYTHONPATH=src .venv/bin/python studies/helm_selection/run.py \
        --fit slice_mmlu.irt --cache ~/.cache/irtcheck-helm-data/cache_mmlu \
        --holdouts 24 --workers 3 > studies/helm_selection/helm95_mmlu.txt

Share of random draws `select` beat, 95 models, 24 held out (`helm95_*.txt`):

| n   | full suite (3,551) | MMLU, 5 subjects (567) | OpenBookQA (500) | citizenship (1,000) |
| --- | ------------------ | ---------------------- | ---------------- | ------------------- |
| 25  | 84%                | 100%                   | 94%              | 63%                 |
| 50  | 38%                | 96%                    | 94%              | 6%                  |
| 100 | 0%                 | 98%                    | 51%              | 0%                  |
| 200 | 0%                 | 62%                    | 6%               | 0%                  |
| 400 | 0%                 | 29%                    | 0%               | 0%                  |

- **On MMLU and OpenBookQA, choosing wins** where it lost on the full suite:
  98-100% of draws on MMLU up to n=100, 94% on OpenBookQA up to n=50. At n=200
  and 400 these slices are 35-80% of their suite, where a random set is nearly
  the whole thing; synthetic data at 400 of 800 is level too.
- **On citizenship it loses, and does not improve with n**: 12.2-12.5 places
  off from 25 items to 400, while random falls from 13.0 to 3.1. Item-rest
  correlation does better (8.8 to 7.5). This is the scenario with the
  answer-format artefact of `docs/validation.md` §2b — median accuracy 0.54 on a
  two-way question, four models far below chance, three of them otherwise strong — so its full-slice
  ranking is mostly noise plus that artefact, and the most informative items are
  the ones that separate the models the scorer misreads.
- **So "several abilities" is not the whole story.** One subject can defeat
  selection when its signal is an artefact, and a mixed-subject slice (MMLU is
  five subjects) can reward it. Citizenship is 28% of the full suite; the direct
  test is the full suite without it, `slice_no_citizenship`, whose output is
  `helm95_no_citizenship.txt`.

One run per slice, holdout fits not bit-reproducible, and each slice's target is
its own full-slice ranking, not the full suite's.

## Caveats

- One real suite. HELM Lite mixes maths, law, general knowledge and reading;
  a single-subject suite may behave differently, and nothing here says which way.
- Holdout fits are not bit-reproducible across machines or thread counts: two
  runs of the twelve-model n=400 row gave +0.874 and +0.881 on the old
  protocol. The committed tables were produced from one cached set of fits.
- The 95-model run holds out 24 models, not 95.

## Files

- `run.py` — fits holdouts (cached per model), scores all selectors.
- `helm12.txt`, `helm95.txt` — the output quoted in `docs/validation.md` §5.
- `slices.py`, `helm95_*.txt` — the single-subject runs above.
