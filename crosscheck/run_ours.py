"""Fit both datasets with irtcheck's own 2PL and record the estimates.

    python crosscheck/run_ours.py

This is the only runner that touches torch and pyro, and it does so by calling
`irtcheck.fit.fitter.fit_2pl` rather than by reimplementing anything — the point
of a cross-check is to test the shipped code path, so it goes through the same
entry point `irtcheck fit` uses and keeps the artifact it produces.

Both prior families are run. `--priors hierarchical` is the default and the one
whose `a` pools toward zero (see fit/model.py); `--priors vague` is the one whose
`a` is centred at +1 and does not pool, and so it is the fairer comparison
against mirt's unpenalised marginal maximum likelihood. Running both is what
separates "our estimator disagrees" from "our prior disagrees", and that
distinction is most of what this exercise is for.
"""

from __future__ import annotations

import numpy as np

from crosscheck.common import DATASETS, Estimates, estimates_path, load_matrix, work_dir


def build_matrix(dataset: str):
    """A ResponseMatrix from the exported rectangle, through the real builder.

    Hand-constructing a ResponseMatrix would skip `build_matrix`'s own
    bookkeeping — `derives_from`, the respondent key, the ragged-cell handling —
    and this comparison is worthless if it is not fitting what `irtcheck fit`
    would fit.
    """
    from irtcheck.matrix import build_matrix as build
    from irtcheck.records import ResponseRecord

    payload = load_matrix(dataset)
    responses = payload["responses"]
    records = [
        ResponseRecord(
            model_id=payload["derives_from"][r],
            item_id=item,
            correct=int(responses[r, c]),
        )
        for r in range(responses.shape[0])
        for c, item in enumerate(payload["item_ids"])
        if not np.isnan(responses[r, c])
    ]
    return build(records, respondent_key=("model_id",))


def run(dataset: str, priors: str, epochs: int = 2000, seed: int = 0) -> Estimates:
    from irtcheck.fit.fitter import fit_2pl

    matrix = build_matrix(dataset)
    fit = fit_2pl(matrix, priors=priors, epochs=epochs, seed=seed, device="cpu")

    out = work_dir(dataset)
    fit.save(out / f"ours-{priors}.irt")

    est = Estimates(
        name=f"ours-{priors}",
        dataset=dataset,
        item_ids=list(fit.item_ids),
        a=np.asarray(fit.a.mean, dtype=float),
        b=np.asarray(fit.b.mean, dtype=float),
        respondent_ids=list(fit.respondent_ids),
        theta=np.asarray(fit.theta.mean, dtype=float),
        parameterisation="a * (theta - b)",
        a_positive=False,
        theta_metric="theta ~ N(0, 1) fixed prior",
        detail={
            "estimator": "SVI / Trace_ELBO, mean-field normal guide (posterior means)",
            "priors": priors,
            "epochs": epochs,
            "seed": seed,
            "identification": fit.model["identification"],
            "reflected": fit.diagnostics["reflected"],
            "elbo_final": fit.diagnostics["elbo_final"],
            "elbo_improvement_last_10pct": fit.diagnostics["elbo_improvement_last_10pct"],
            "seconds": fit.diagnostics["seconds"],
            "flag_counts": fit.diagnostics["flag_counts"],
            # The interval half-widths, kept here because the whole
            # insufficient-data argument is about interval width and the grid
            # check in run_gridpost.py needs something to compare against.
            "a_sd": [float(x) for x in fit.a.sd],
            "a_hdi_low": [float(x) for x in fit.a.hdi_low],
            "a_hdi_high": [float(x) for x in fit.a.hdi_high],
            "b_sd": [float(x) for x in fit.b.sd],
        },
    )
    est.to_json(estimates_path(dataset, est.name))
    print(
        f"{dataset}/{est.name}: ELBO {fit.diagnostics['elbo_final']:.1f} in "
        f"{fit.diagnostics['seconds']:.1f}s, reflected={fit.diagnostics['reflected']}, "
        f"flags={fit.diagnostics['flag_counts']}"
    )
    return est


def main() -> None:
    for dataset in DATASETS:
        for priors in ("hierarchical", "vague"):
            run(dataset, priors)


if __name__ == "__main__":
    main()
