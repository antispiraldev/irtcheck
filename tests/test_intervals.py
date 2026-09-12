"""The interval on `a` is a claim, so it is measured rather than asserted.

`insufficient-data` fires when the 95% interval on `a_i` contains zero, and it
is the flag the whole "refusal is a feature" argument rests on. That makes the
*width* of the interval load-bearing in a way a point estimate never is, and it
makes too-narrow the dangerous direction: narrowing the interval makes the tool
refuse less, claiming a discrimination it has not earned.

For the whole of waves 0-2 the fitter's docstring asserted the opposite — that
mean-field's narrow intervals made the flag "conservative, never over-eager" —
and nothing measured it. The wave-2 cross-check found the claim inverted. This
file is the check that should have existed: it measures coverage against
synth.py's ground truth, so the next person to change the guide, the priors or
the summaries finds out from a failing test rather than from a cross-check two
waves later.

## Why coverage, and not a comparison against a better interval

crosscheck/run_gridpost.py compares our interval to a brute-force grid
posterior, which is a good check but an expensive one, and it conditions on
`theta_hat` so it only ever bounds the width from below. Coverage needs neither
concession: the true `a` is known, so "does a nominal 95% interval contain it
95% of the time" has a direct answer. The two agree on the magnitude of the old
error — the grid put the variational width at 0.56x truth at 300 respondents,
and 80.6% coverage implies 0.66x — which is why both are worth having.

## The tolerances

Coverage is a proportion over a few hundred items and three seeds, so it has
sampling error of its own; the bands below are wide enough not to flake and
narrow enough to catch a regression to the variational width, which sat 5-11
points lower in every regime. The floor is what matters. The ceiling is there
because an interval that over-covers is also wrong, and because a guide change
that made every interval enormous would otherwise pass.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyro", reason="the fitter and its tests need torch and pyro")

from irtcheck.artifact import FLAG_INSUFFICIENT_DATA  # noqa: E402
from irtcheck.fit.fitter import fit_2pl  # noqa: E402
from irtcheck.synth import synthetic_matrix  # noqa: E402

SEEDS = (0, 1, 2)

# Measured with the conditional-information interval this module tests, over the
# three seeds above. The variational marginals it replaced gave 87.5 / 84.3 /
# 80.6 in the same three regimes, so the floors sit below what was measured and
# above what the old width could reach.
#
#   regime      variational   conditional information   floor asserted here
#   15 x 500        87.5%              92.7%                   0.89
#   60 x 500        84.3%              93.3%                   0.89
#   300 x 60        80.6%              91.1%                   0.88
REGIMES = {
    "small": dict(n_models=15, n_items=500, floor=0.89, ceiling=0.99),
    "medium": dict(n_models=60, n_items=500, floor=0.89, ceiling=0.99),
    "dense": dict(n_models=300, n_items=60, floor=0.88, ceiling=0.99),
}


def coverage_of(*, n_models: int, n_items: int, seed: int) -> tuple[float, float]:
    """(coverage of the true `a`, median interval width) for one fit."""
    matrix, truth = synthetic_matrix(n_models=n_models, n_items=n_items, seed=seed)
    fit = fit_2pl(matrix, seed=seed)
    # The fit's item order is its own; truth is indexed by item id.
    order = [list(truth.item_ids).index(item) for item in fit.item_ids]
    true_a = np.asarray(truth.a, dtype=float)[order]
    low = np.asarray(fit.a.hdi_low, dtype=float)
    high = np.asarray(fit.a.hdi_high, dtype=float)
    covered = (low <= true_a) & (true_a <= high)
    return float(covered.mean()), float(np.median(high - low))


@pytest.mark.slow
@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_the_interval_on_a_covers_the_truth(regime):
    """A nominal 95% interval must cover the true `a` close to 95% of the time.

    Averaged over seeds rather than asserted per seed: one fit's coverage is a
    proportion over a few hundred correlated items and moves by a couple of
    points between seeds for reasons that are not the fitter.
    """
    spec = REGIMES[regime]
    measured = [
        coverage_of(n_models=spec["n_models"], n_items=spec["n_items"], seed=seed)
        for seed in SEEDS
    ]
    coverage = float(np.mean([c for c, _ in measured]))
    assert coverage >= spec["floor"], (
        f"{regime}: the 95% interval on `a` covered the true value "
        f"{coverage:.1%} of the time, below the {spec['floor']:.0%} floor. "
        "Too narrow means `insufficient-data` fires too rarely, so the tool "
        "claims discriminations it has not earned. See fit/intervals.py."
    )
    assert coverage <= spec["ceiling"], (
        f"{regime}: coverage came out {coverage:.1%}, above the "
        f"{spec['ceiling']:.0%} ceiling. An interval that always contains the "
        "answer is not evidence, and it would make `insufficient-data` fire on "
        "items the data can in fact separate."
    )


@pytest.mark.slow
def test_a_wider_interval_refuses_more_not_less():
    """The direction of the whole finding, asserted as a property.

    Not a tolerance but a monotonicity: `insufficient-data` is "the interval
    spans zero", so scaling every interval up must refuse at least as many
    items and scaling it down at most as many. This is the sentence the old
    fitter docstring got backwards, and it is cheap to make untellable-wrong.
    """
    from irtcheck.artifact import compute_flags

    matrix, _ = synthetic_matrix(n_models=15, n_items=200, seed=0)
    fit = fit_2pl(matrix, seed=0)

    def refused(scale: float) -> int:
        widened = type(fit.a)(
            mean=list(fit.a.mean),
            sd=[s * scale for s in fit.a.sd],
            hdi_low=[m - (m - lo) * scale for m, lo in zip(fit.a.mean, fit.a.hdi_low, strict=True)],
            hdi_high=[m + (hi - m) * scale for m, hi in zip(fit.a.mean, fit.a.hdi_high, strict=True)],
        )
        flags = compute_flags(widened, fit.b, fit.p_correct, fit.theta.mean)
        return sum(FLAG_INSUFFICIENT_DATA in item for item in flags)

    narrow, base, wide = refused(0.5), refused(1.0), refused(2.0)
    assert narrow <= base <= wide, (
        f"refusals did not increase with interval width: {narrow} at half width, "
        f"{base} as fitted, {wide} at double. `insufficient-data` is defined as "
        "the interval spanning zero, so this is arithmetic, not a tolerance."
    )
    assert narrow < wide, (
        "widening the interval by 4x changed nothing, which means this suite has "
        "no items near the boundary and the test is not measuring anything"
    )


# Measured with the information-plus-scale interval this module tests, over the
# three seeds above. The variational marginals it replaced gave 64.4 / 78.9 /
# 93.9, and information alone gave 64.4 / 80.0 / 93.6 -- the scale term is the
# whole of the improvement at small n.
#
#   regime      variational   information only   information + scale   floor
#   15 x 500        64.4%          64.4%               88.9%            0.84
#   60 x 500        78.9%          80.0%               88.9%            0.84
#   300 x 60        93.9%          93.6%               93.7%            0.88
THETA_REGIMES = {
    "small": dict(n_models=15, n_items=500, floor=0.84),
    "medium": dict(n_models=60, n_items=500, floor=0.84),
    "dense": dict(n_models=300, n_items=60, floor=0.88),
}


def theta_coverage(*, n_models: int, n_items: int, seed: int) -> float:
    matrix, truth = synthetic_matrix(n_models=n_models, n_items=n_items, seed=seed)
    fit = fit_2pl(matrix, seed=seed)
    order = [list(truth.respondent_ids).index(r) for r in fit.respondent_ids]
    true_theta = np.asarray(truth.theta, dtype=float)[order]
    low = np.asarray(fit.theta.hdi_low, dtype=float)
    high = np.asarray(fit.theta.hdi_high, dtype=float)
    return float(((low <= true_theta) & (true_theta <= high)).mean())


@pytest.mark.slow
@pytest.mark.parametrize("regime", sorted(THETA_REGIMES))
def test_the_interval_on_theta_covers_the_truth(regime):
    """`theta`'s interval has to carry the uncertainty in its own ruler.

    `theta ~ N(0, 1)` is a fixed prior, so a fit standardises abilities to its
    own sample — and `n` draws from `N(0, 1)` have a sample sd that is not 1.
    Every ability therefore comes out stretched by a shared factor, which no
    per-respondent width covers: at 15 respondents the variational interval
    covered the truth 64.4% of the time against a nominal 95%, and computing it
    from information instead changed nothing. Adding the scale term takes it to
    88.9%. See fit/intervals.py.
    """
    spec = THETA_REGIMES[regime]
    coverage = float(
        np.mean(
            [
                theta_coverage(n_models=spec["n_models"], n_items=spec["n_items"], seed=seed)
                for seed in SEEDS
            ]
        )
    )
    assert coverage >= spec["floor"], (
        f"{regime}: the 95% interval on `theta` covered the true ability "
        f"{coverage:.1%} of the time, below the {spec['floor']:.0%} floor."
    )


@pytest.mark.slow
def test_the_scale_correction_widens_the_extremes_and_not_the_middle():
    """The scale term is proportional to |theta|, and that is the point.

    A mis-estimated ruler moves the models furthest from zero the most and the
    ones near zero not at all, so a correction that widened every interval
    equally would be the wrong shape however well it scored on coverage.
    """
    matrix, _ = synthetic_matrix(n_models=40, n_items=300, seed=0)
    fit = fit_2pl(matrix, seed=0)
    theta = np.abs(np.asarray(fit.theta.mean, dtype=float))
    sd = np.asarray(fit.theta.sd, dtype=float)
    middle = theta < np.quantile(theta, 0.25)
    extreme = theta > np.quantile(theta, 0.75)
    assert sd[extreme].mean() > sd[middle].mean(), (
        f"the extreme models' intervals ({sd[extreme].mean():.4f}) are not wider "
        f"than the middle of the pack's ({sd[middle].mean():.4f}), so the scale "
        "correction is not proportional to |theta| as intended"
    )


@pytest.mark.slow
def test_a_respondent_who_answered_nothing_gets_the_prior_back():
    """theta ~ N(0, 1) contributes precision 1, so an empty row has sd 1.

    Not a division by zero, and not an arbitrary fallback: the prior *is* the
    answer for a respondent there is no evidence about.
    """
    matrix, _ = synthetic_matrix(n_models=15, n_items=60, seed=0)
    keep = np.asarray(matrix.rows) != 0
    stripped = type(matrix)(
        respondent_ids=list(matrix.respondent_ids),
        item_ids=list(matrix.item_ids),
        derives_from=list(matrix.derives_from),
        rows=np.asarray(matrix.rows)[keep],
        cols=np.asarray(matrix.cols)[keep],
        obs=np.asarray(matrix.obs)[keep],
        respondent_key=tuple(matrix.respondent_key),
    )
    dropped = matrix.respondent_ids[0]
    fit = fit_2pl(stripped, seed=0)
    index = fit.respondent_ids.index(dropped)
    # The within-respondent term is exactly 1/sqrt(1) = 1; the scale term adds
    # (theta * relative)^2 on top, and theta for an unanswered row sits near 0.
    assert 1.0 <= fit.theta.sd[index] < 1.05, fit.theta.sd[index]


@pytest.mark.slow
def test_an_unanswered_item_gets_the_population_prior_back():
    """With no responses the information is all prior, so sd should be sigma_a.

    The honest answer for an item nobody attempted is "we know what this suite
    looks like and nothing about this item", and that is what the conditional
    information gives for free — the data term is empty and the prior precision
    is all that is left. Worth pinning because it is the one case where the
    interval is not driven by data at all.
    """
    matrix, _ = synthetic_matrix(n_models=15, n_items=60, seed=0)
    # Keep the item in the suite, drop every response to it.
    dropped = 0
    keep = np.asarray(matrix.cols) != dropped
    stripped = type(matrix)(
        respondent_ids=list(matrix.respondent_ids),
        item_ids=list(matrix.item_ids),
        derives_from=list(matrix.derives_from),
        rows=np.asarray(matrix.rows)[keep],
        cols=np.asarray(matrix.cols)[keep],
        obs=np.asarray(matrix.obs)[keep],
        respondent_key=tuple(matrix.respondent_key),
    )
    dropped_id = matrix.item_ids[dropped]
    fit = fit_2pl(stripped, seed=0)
    index = fit.item_ids.index(dropped_id)
    sigma_a = fit.diagnostics["hyperparameters"]["sigma_a"]
    assert fit.n_resp[index] == 0
    assert fit.a.sd[index] == pytest.approx(sigma_a, rel=1e-6), (
        "an item with no responses should fall back to exactly the population "
        f"prior sd {sigma_a:.4f}, got {fit.a.sd[index]:.4f}"
    )
    assert FLAG_INSUFFICIENT_DATA in fit.flags[index]
