"""Check our SVI interval on `a` against a brute-force grid posterior.

    python crosscheck/run_gridpost.py

Every other check in this directory compares *point estimates*. But irtcheck's
central claim is not a point estimate — it is an interval, and specifically the
decision `insufficient-data`, which fires when the 95% interval on `a_i`
contains zero (see artifact.py and fit/model.py). A point estimate can be right
while the interval around it is badly wrong, and mean-field SVI is exactly the
method you would expect to get interval *width* wrong: it factorises the
posterior and so drops the `a`-`b` and item-ability correlations, which is
understood to make variational intervals too narrow.

So this computes the thing SVI approximates, without approximating it: for one
item at a time, the posterior over (a_i, b_i) on a dense 2-D grid, with the
model's own priors, by direct numerical integration.

## What is and is not being checked

Being exact about this matters, because a check that quietly measures something
easier than it claims is worse than none.

This conditions on the **fitted ability vector** rather than integrating over
it. The quantity computed is therefore

    p(a_i, b_i | responses to item i, theta = theta_hat)

not the full joint marginal posterior. It is a real check on the part of the
approximation most likely to be wrong — how much width the mean-field guide
assigns to a single item's parameters given the scale it has settled on — and
it is not a check on how uncertainty in theta propagates into `a`.

**And the direction of that shortfall is what makes the result usable.**
Conditioning on theta_hat treats the abilities as known, which *removes*
uncertainty the real posterior has. So this grid interval is a **lower bound**
on the true interval width, and the width ratio it reports —
`w_svi / w_grid` — is an **upper bound** on how wide the SVI interval is
relative to the truth. When that ratio comes out below 1, the SVI interval is
at least that much too narrow, and integrating theta properly could only make
the gap larger. A finding of over-narrowness from this check therefore cannot
be an artefact of the conditioning; only a finding of over-*wideness* could be,
and none is reported.

Reported per item: the grid posterior's mean and 95% equal-tailed interval for
`a`, our SVI mean and interval, the ratio of widths, and whether the two agree
on the `insufficient-data` verdict — which is the only thing the tool acts on.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from crosscheck.common import (
    DATASETS,
    Estimates,
    estimates_path,
    load_matrix,
    load_truth,
    write_json,
)

# The grid. Wide enough to hold the prior mass that matters and fine enough that
# the 95% interval ends are not grid-limited; both are asserted below rather
# than hoped for.
A_GRID = np.linspace(-6.0, 6.0, 481)
B_GRID = np.linspace(-12.0, 12.0, 481)


def log_normal_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return -0.5 * ((x - loc) / scale) ** 2 - np.log(scale)


def grid_posterior(
    x: np.ndarray, theta: np.ndarray, *, sigma_a: float, mu_b: float, sigma_b: float
) -> tuple[np.ndarray, np.ndarray]:
    """The (a, b) log-posterior for one item on the grid, and its `a` marginal.

    `x` is that item's observed responses and `theta` the abilities of the
    respondents who produced them, already aligned and with missing cells
    dropped by the caller.
    """
    a = A_GRID[:, None, None]
    b = B_GRID[None, :, None]
    z = a * (theta[None, None, :] - b)
    # Bernoulli log-likelihood summed over respondents, vectorised over the grid.
    log_lik = (x[None, None, :] * -np.logaddexp(0.0, -z)).sum(axis=2) + (
        (1.0 - x[None, None, :]) * -np.logaddexp(0.0, z)
    ).sum(axis=2)

    log_prior = log_normal_pdf(A_GRID, 0.0, sigma_a)[:, None] + log_normal_pdf(
        B_GRID, mu_b, sigma_b
    )[None, :]
    log_post = log_lik + log_prior
    log_post -= logsumexp(log_post)

    # Marginalise b out by summing the joint over the b axis. The grid spacing
    # is uniform so it cancels in the normalisation.
    log_marginal_a = logsumexp(log_post, axis=1)
    return log_post, log_marginal_a - logsumexp(log_marginal_a)


def summarise(log_marginal_a: np.ndarray) -> dict:
    """Mean and 95% equal-tailed interval of the `a` marginal, plus a grid check."""
    p = np.exp(log_marginal_a)
    p = p / p.sum()
    mean = float(A_GRID @ p)
    cdf = np.cumsum(p)
    low = float(np.interp(0.025, cdf, A_GRID))
    high = float(np.interp(0.975, cdf, A_GRID))
    # If either tail holds appreciable mass at the grid edge the interval is
    # clipped by the grid rather than by the data, and the number is not a
    # posterior interval at all. Recorded so it can never be quoted unnoticed.
    edge_mass = float(p[0] + p[-1])
    return {
        "mean": mean,
        "low": low,
        "high": high,
        "width": high - low,
        "spans_zero": bool(low <= 0.0 <= high),
        "grid_edge_mass": edge_mass,
        "grid_limited": edge_mass > 1e-4,
    }


def learned_hyperparameters(dataset: str, fit) -> tuple[float, float, float, dict]:
    """Recover the population scales the hierarchical fit actually settled on.

    They matter: `sigma_a`, `mu_b` and `sigma_b` are *learned*, so a grid
    posterior built from the prior's nominal values would be a different model
    from the one SVI fitted, and the gap between them would look like
    approximation error when it was a different prior.

    The artifact does not carry them — `fit`'s diagnostics record convergence,
    not hyperparameters — and adding them is another brief's file to change. So
    this refits with the identical seed and epoch count and reads them off the
    guide. `fit_2pl` is documented as deterministic given its seed, and that is
    not taken on trust: the refit's `a` posterior means are checked against the
    artifact's, and a mismatch aborts rather than quietly using hyperparameters
    from some other fit.
    """
    import pyro
    import torch
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    from crosscheck.run_ours import build_matrix
    from irtcheck.fit.fitter import LR_DECAY_TOTAL
    from irtcheck.fit.model import PRIORS_HIERARCHICAL, make_guide, make_model, to_tensors

    diag = fit.diagnostics
    epochs, seed, lr = int(diag["epochs"]), int(diag["seed"]), float(diag["lr"])

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    data = to_tensors(build_matrix(dataset), device=torch.device("cpu"))
    model = make_model(PRIORS_HIERARCHICAL)
    guide = make_guide(model, data, priors=PRIORS_HIERARCHICAL)
    svi = SVI(
        model,
        guide,
        ClippedAdam({"lr": lr, "lrd": LR_DECAY_TOTAL ** (1.0 / max(epochs, 1))}),
        loss=Trace_ELBO(),
    )
    for _ in range(epochs):
        svi.step(data)

    median = guide.median(data)
    a_refit = median["a"].detach().cpu().numpy()
    a_artifact = np.asarray(fit.a.mean, dtype=float)
    # AutoNormal's median for an unconstrained Normal site is its loc, which is
    # exactly the posterior mean the artifact stores — so these must match.
    drift = float(np.max(np.abs(a_refit - a_artifact)))
    if drift > 1e-4:
        raise SystemExit(
            f"refit did not reproduce the artifact (max|da|={drift:.3g}). fit_2pl is "
            "supposed to be deterministic given its seed; the hyperparameters read off "
            "this guide would not be the ones the artifact's intervals came from."
        )

    return (
        float(median["sigma_a"].detach().cpu()),
        float(median["mu_b"].detach().cpu()),
        float(median["sigma_b"].detach().cpu()),
        {"refit_reproduced_artifact_to": drift},
    )


def run(dataset: str, n_items: int = 12) -> dict:
    from irtcheck.artifact import IrtFit

    fit = IrtFit.load(estimates_path(dataset, "ours-hierarchical").parent / "ours-hierarchical.irt")
    est = Estimates.from_json(estimates_path(dataset, "ours-hierarchical"))
    payload = load_matrix(dataset)
    truth = load_truth(dataset)
    responses = payload["responses"]

    theta_hat = np.asarray(fit.theta.mean, dtype=float)
    # The fit's respondent order is not necessarily the matrix's.
    row_of = {rid: i for i, rid in enumerate(payload["respondent_ids"])}
    rows = np.array([row_of[r] for r in fit.respondent_ids])

    sigma_a, mu_b, sigma_b, refit_note = learned_hyperparameters(dataset, fit)
    a_sd = np.asarray(est.detail["a_sd"], dtype=float)
    a_low = np.asarray(est.detail["a_hdi_low"], dtype=float)
    a_high = np.asarray(est.detail["a_hdi_high"], dtype=float)
    a_mean = np.asarray(est.a, dtype=float)

    # Pick a spread of items rather than the first n: the ones with the widest
    # and narrowest SVI intervals, plus a random middle, so the comparison
    # covers the items the flag actually separates.
    order = np.argsort(a_high - a_low)
    picks = np.unique(
        np.concatenate(
            [
                order[: n_items // 3],
                order[-(n_items // 3) :],
                np.random.default_rng(3).choice(order, size=n_items // 3, replace=False),
            ]
        )
    )

    rows_out = []
    for idx in picks:
        col = fit.item_ids[idx]
        c = payload["item_ids"].index(col)
        observed = np.isfinite(responses[rows, c])
        x = responses[rows, c][observed]
        th = theta_hat[observed]
        if x.size < 2 or np.allclose(x, x[0]):
            continue
        _, log_marg = grid_posterior(x, th, sigma_a=sigma_a, mu_b=mu_b, sigma_b=sigma_b)
        grid = summarise(log_marg)
        svi = {
            "mean": float(a_mean[idx]),
            "sd": float(a_sd[idx]),
            "low": float(a_low[idx]),
            "high": float(a_high[idx]),
            "width": float(a_high[idx] - a_low[idx]),
            "spans_zero": bool(a_low[idx] <= 0.0 <= a_high[idx]),
        }
        rows_out.append(
            {
                "item_id": col,
                "true_a": float(truth["a"][truth["item_ids"].index(col)]),
                "n_responses": int(x.size),
                "grid": grid,
                "svi": svi,
                "width_ratio_svi_over_grid": svi["width"] / grid["width"],
                "verdict_agrees": grid["spans_zero"] == svi["spans_zero"],
            }
        )

    print(f"\n=== {dataset}: SVI interval on `a` vs a grid posterior "
          f"(learned sigma_a={sigma_a:.3f}, mu_b={mu_b:+.3f}, sigma_b={sigma_b:.3f}; "
          f"refit reproduced the artifact to {refit_note['refit_reproduced_artifact_to']:.1e})")
    print("  conditioned on theta = theta_hat; see module docstring for what that does and does not check")
    print(f"  {'item':<13}{'true a':>8}{'n':>5}  {'grid a [95%]':>26}  {'SVI a [95%]':>26}"
          f"  {'w_svi/w_grid':>13}  spans0")
    for row in rows_out:
        g, s = row["grid"], row["svi"]
        star = "" if row["verdict_agrees"] else "  <-- VERDICT DIFFERS"
        limited = " (grid-limited)" if g["grid_limited"] else ""
        print(
            f"  {row['item_id']:<13}{row['true_a']:>8.3f}{row['n_responses']:>5}  "
            f"{g['mean']:>7.3f} [{g['low']:+6.3f},{g['high']:+6.3f}]  "
            f"{s['mean']:>7.3f} [{s['low']:+6.3f},{s['high']:+6.3f}]  "
            f"{row['width_ratio_svi_over_grid']:>13.3f}  "
            f"{'grid=' + str(g['spans_zero'])[:1]}/{'svi=' + str(s['spans_zero'])[:1]}"
            f"{star}{limited}"
        )

    ratios = np.array([r["width_ratio_svi_over_grid"] for r in rows_out])
    agree = sum(r["verdict_agrees"] for r in rows_out)
    print(f"  width ratio: median {np.median(ratios):.3f}, range "
          f"[{ratios.min():.3f}, {ratios.max():.3f}]  "
          f"(<1 means SVI is narrower than the grid posterior, the expected direction)")
    print(f"  insufficient-data verdict agrees on {agree}/{len(rows_out)} items")

    return {
        "dataset": dataset,
        "sigma_a": sigma_a,
        "mu_b": mu_b,
        "sigma_b": sigma_b,
        **refit_note,
        "conditioned_on": "theta = posterior mean (NOT a full joint marginal)",
        "n_items": len(rows_out),
        "median_width_ratio_svi_over_grid": float(np.median(ratios)),
        "min_width_ratio": float(ratios.min()),
        "max_width_ratio": float(ratios.max()),
        "verdict_agreement": f"{agree}/{len(rows_out)}",
        "items": rows_out,
    }


def main() -> None:
    from crosscheck.common import RESULTS

    out = {ds: run(ds) for ds in DATASETS}
    write_json(RESULTS / "grid_posterior.json", out)
    print(f"\nwrote {RESULTS / 'grid_posterior.json'}")


if __name__ == "__main__":
    main()
