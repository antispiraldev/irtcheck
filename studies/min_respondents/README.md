# How many respondents does irtcheck need?

**Not shipped.** A one-time study, excluded from the sdist like `crosscheck/`.
It answers the open question in `docs/spec.md` — the minimum viable respondent
count — by measuring what the tool actually hands a user at each count, against
synthetic ground truth. The write-up, and what it changes, is
[`docs/validation.md` §4](../../docs/validation.md#4-how-many-respondents).

    # ~18 minutes on 14 worker processes; each fit is pinned to one thread
    PYTHONPATH=src .venv/bin/python studies/min_respondents/run.py --out ~/.cache/irtcheck-study --workers 14

    # ~2 minutes; rewrites results.json
    PYTHONPATH=src .venv/bin/python studies/min_respondents/analyse.py \
        --fits ~/.cache/irtcheck-study --json studies/min_respondents/results.json

    # results.txt is generated from results.json, so the two cannot disagree
    PYTHONPATH=src .venv/bin/python studies/min_respondents/analyse.py \
        --from-json studies/min_respondents/results.json > studies/min_respondents/results.txt

## The grid

336 fits: real models ∈ {3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100} ×
items ∈ {200, 800} × prompt variants per model ∈ {1, 3} × six seeds. Every
universe is `synth.make_truth` at its defaults — 15% near-flat items, 8% far
off-range — plus 2% mis-keyed items, because whether a gate keeps those out is
one of the things measured. Complete matrices, default fit settings.

## What gets scored

The anchor set, because that is the deliverable. Each set ranks **2,000 fresh
models** the fit never saw by plain accuracy over its items, and the score is
Spearman against their true ability. Two fresh populations: `same`, drawn like
the fit's models, and `stronger`, a generation one sd ahead.

| set        | how it is chosen                                              |
| ---------- | ------------------------------------------------------------- |
| `tool`     | `select_anchor(fit, n)` — what irtcheck does                  |
| `ungated`  | the same selection with only ceiling/floor excluded           |
| `classic`  | top-n by item-rest correlation, per real model — no IRT       |
| `filtered` | uniform draw from the non-ceiling/floor pool (40 draws)       |
| `random`   | uniform draw from every item (40 draws)                       |
| `oracle`   | top-n by *true* information over N(0, 1)                      |

`classic` and `filtered` are there because beating a uniform draw that includes
items every model gets right is not the bar. Differences are **paired within a
fit** — same universe, same fresh models — which is what makes a 0.002
difference readable at six seeds.

## Files

- `run.py` — fits and saves artifacts; resumable, skips what exists.
- `analyse.py` — reads artifacts, rebuilds each universe, scores everything.
- `results.json` — per-cell means and per-fit rows (one line each).
- `probes.py`, `probes.txt` — the follow-ups §4 quotes: ability spread at few
  models, the sign barrier, which side intervals miss on, and 2,000 vs 8,000
  epochs.
- `results.txt` — every table, including paired differences and where the
  planted mis-keyed items end up.
