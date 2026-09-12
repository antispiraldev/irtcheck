"""Generate the two synthetic datasets and write them in all three input formats.

Run this first. Every estimator reads the files it produces, so all four are
fitting *the same responses* — which is the only reason a disagreement between
them is attributable to the estimator rather than to the draw.

    python crosscheck/export_inputs.py

Writes, per dataset, into crosscheck/work/<dataset>/:

    truth.json    the generating parameters, from irtcheck.synth
    matrix.json   the 0/1/null rectangle plus ids — what our fitter and the
                  scipy reference read
    wide.csv      one row per respondent, one column per item, NA for missing —
                  what mirt reads
    pyirt.jsonl   {"subject_id": ..., "responses": {...}} — what py-irt reads

This module imports irtcheck.synth, which is numpy-only by construction (see
its docstring), so this step needs no torch.
"""

from __future__ import annotations

import numpy as np

from crosscheck.common import DATASETS, work_dir, write_json
from irtcheck.synth import make_truth, responses_from_truth


def export(name: str) -> None:
    spec = dict(DATASETS[name])
    why = spec.pop("why")
    missing = spec.pop("missing")
    seed = spec["seed"]

    truth = make_truth(**spec)
    # `seed + 1` matches synth.synthetic_matrix, so a dataset here is the same
    # draw a test in tests/ would get from the same seed.
    responses = responses_from_truth(truth, missing=missing, seed=seed + 1)

    out = work_dir(name)
    out.mkdir(parents=True, exist_ok=True)

    write_json(
        out / "truth.json",
        {
            "dataset": name,
            "why": why,
            "spec": {**spec, "missing": missing},
            "item_ids": truth.item_ids,
            "respondent_ids": truth.respondent_ids,
            "derives_from": truth.derives_from,
            "a": [float(x) for x in truth.a],
            "b": [float(x) for x in truth.b],
            "theta": [float(x) for x in truth.theta],
        },
    )

    write_json(
        out / "matrix.json",
        {
            "dataset": name,
            "item_ids": truth.item_ids,
            "respondent_ids": truth.respondent_ids,
            "derives_from": truth.derives_from,
            # null, not NaN: JSON has no NaN and a reader that accepts one is a
            # reader that will accept anything.
            "responses": [
                [None if np.isnan(v) else int(v) for v in row] for row in responses
            ],
        },
    )

    header = ["respondent_id", *truth.item_ids]
    lines = [",".join(header)]
    for r, respondent in enumerate(truth.respondent_ids):
        cells = ["NA" if np.isnan(v) else str(int(v)) for v in responses[r]]
        lines.append(",".join([respondent, *cells]))
    (out / "wide.csv").write_text("\n".join(lines) + "\n")

    jsonl = []
    for r, respondent in enumerate(truth.respondent_ids):
        pairs = {
            item: int(responses[r, c])
            for c, item in enumerate(truth.item_ids)
            if not np.isnan(responses[r, c])
        }
        jsonl.append({"subject_id": respondent, "responses": pairs})
    write_jsonl(out / "pyirt.jsonl", jsonl)

    n_obs = int(np.isfinite(responses).sum())
    print(
        f"{name}: {len(truth.respondent_ids)} respondents x {len(truth.item_ids)} items, "
        f"{n_obs} observations ({n_obs / responses.size:.0%} dense) -> {out}"
    )


def write_jsonl(path, rows) -> None:
    import json

    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    for name in DATASETS:
        export(name)


if __name__ == "__main__":
    main()
