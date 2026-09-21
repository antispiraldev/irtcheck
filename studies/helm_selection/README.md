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
