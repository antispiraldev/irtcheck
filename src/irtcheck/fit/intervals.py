"""Intervals on the item parameters, from the conditional information matrix.

The variational marginals that come out of SVI are point estimates with a width
attached, and the width is the part mean-field gets wrong. This module computes
the width a different way — from the curvature of the log-posterior — and it is
the width the artifact reports.

## Why the variational width is not good enough

`insufficient-data`, the flag that decides whether an item is ranked at all, is
the question of whether the 95% interval on `a_i` contains zero. So the width of
that interval is not a diagnostic here, it is the claim.

A mean-field guide factorises the posterior, and for a Gaussian target that
makes each marginal variance the *conditional* one, `1/precision_ii`, rather
than the marginal `Sigma_ii`. The two differ by `sqrt(1 - R^2)` where `R` is the
coordinate's multiple correlation with everything else, so a mean-field interval
is too narrow by exactly as much as the parameter is correlated with the rest of
the model. For a 2PL that correlation is large and structural: `a_i` and `b_i`
enter the likelihood only through `a_i * (theta_j - b_i)`, and their posterior
correlation runs past 0.85 on ordinary data.

Too narrow is the dangerous direction. `insufficient-data` fires when the
interval *spans zero*, so narrowing it makes the tool refuse **less** — it
claims a discrimination it has not earned. Measured against synthetic ground
truth over three seeds, the nominal 95% variational interval on `a` covered the
true value 87.5% of the time at 15 respondents x 500 items, 84.3% at 60 x 500
and 80.6% at 300 x 60. See tests/test_intervals.py, which measures this rather
than trusting this paragraph.

## What is computed instead

For one item, hold the abilities and the population scales at their fitted
values and take the 2x2 expected information of `(a_i, b_i)`:

    I_aa = sum_j w_ij (theta_j - b_i)^2  +  1 / sigma_a^2
    I_ab = -a_i sum_j w_ij (theta_j - b_i)
    I_bb = a_i^2 sum_j w_ij             +  1 / sigma_b^2

with `w_ij = p_ij (1 - p_ij)` the Bernoulli variance. Inverting the block gives
`var(a_i) = I_bb / det` and `var(b_i) = I_aa / det`, which is the Laplace
approximation to the posterior of that item's two parameters *jointly* — so the
`a`-`b` correlation is carried rather than dropped.

Three properties make this the right shape for this tool:

  - **It is always defined.** The data term is a sum of outer products of
    `[theta_j - b_i, -a_i]` weighted by `w_ij >= 0`, hence positive
    semi-definite, and the prior precisions on the diagonal are strictly
    positive; the sum is positive definite for every item, including one with
    no responses at all. *Expected* information is what buys this. The observed
    information — the actual Hessian — carries a score term that vanishes only
    at the conditional mode, and evaluated at the variational mean instead it
    made the block indefinite for 2% of items at 15 respondents. Coverage of the
    two is otherwise indistinguishable (93.1% vs 92.7%), so there is no accuracy
    being traded for the guarantee.

  - **An item nobody answered gets its prior back.** With no responses the data
    term is zero and `var(a_i) = sigma_a^2` exactly, which is the honest answer:
    we know what the suite looks like and nothing about this item.

  - **It is closed form and deterministic.** One pass over the response
    triplets, no sampling, so a fit is still reproducible to the last digit
    given its seed.

## What it does not fix, stated plainly

Conditioning on `theta_hat` treats the abilities as known, so this drops the
uncertainty in them. The remaining shortfall is measurable and is not zero:
coverage comes out at 92.7% / 93.3% / 91.1% in the three regimes above against
a nominal 95%. That is the direction to be wrong in — still slightly too narrow,
so still slightly slow to refuse — but it is a 5-11 point improvement on the
variational width, and the flag it feeds is the point of the tool.

## A knock-on effect on `dead`, measured

Widening every interval makes `dead` rarer, because `dead` needs the interval
to sit wholly inside `(0, DEAD_THRESHOLD)` — it must clear zero, or the item is
`insufficient-data` instead. That caps the width at 0.35 and so the sd at
`0.35 / (2 * 1.96) = 0.089`, which is a lot of evidence about one item.

Measured on synth matrices with these intervals: at 300 respondents the
smallest sd on any item is 0.088 and **no item reaches `dead`**; at 1000 it is
0.050 and 18 items do; at 3000, 13 of 100. So `dead` is reachable, but it needs
roughly a *thousand* respondents rather than the hundred the docs used to claim.
Below that, an item that does not discriminate comes back
`insufficient-data` — "we cannot tell" rather than "we are sure it is flat" —
which is the weaker claim and the right one to make from a wider interval.

## `theta`, where the missing uncertainty is the ruler itself

`theta` needs a different correction, and applying the item treatment to it
does nothing at all: 64.4% coverage before and 64.4% after at 15 x 500. The
reason is that its error is not width but **scale**.

`theta ~ N(0, 1)` is a fixed prior, and that is deliberate — it is the ruler
that makes `a` and `b` identified at all (see model.py, "Identification"). But
it is a claim about the *population*, and a fit sees a sample of `n` of them.
Fifteen draws from `N(0, 1)` have a sample sd of 0.82 as readily as 1.0, and
the fit standardises to its own sample, so every ability comes out stretched by
the ratio. Measured at 15 x 500: sd of the estimates 0.94 against 0.82 for the
truth, and regressing truth on estimate gives a slope of 0.85 with an RMSE 2.18
times the reported sd. A shared multiplicative factor is not noise, so no
amount of per-respondent width covers it.

What does cover it is admitting the ruler is uncertain. The sample sd of `n`
draws has relative standard error `1/sqrt(2(n-1))`, and an error in the scale
moves `theta_j` in proportion to `theta_j` itself, so it enters in quadrature:

    var(theta_j) = 1 / (1 + I(theta_j))  +  (theta_j / sqrt(2(n-1)))^2

The first term is the within-respondent information, `I(theta_j) =
sum_i a_i^2 w_ij` with the prior's precision of 1 added — the same
Fisher-information reasoning as for the items, and already what `select` and the
HTML report use for the ability standard errors they show. The second is the
ruler. Measured coverage, three seeds:

    regime      variational   information only   information + scale
    15 x 500        64.4%          64.4%               88.9%
    60 x 500        78.9%          80.0%               88.9%
    300 x 60        93.9%          93.6%               93.7%

Two properties worth noting. The correction vanishes as `n` grows, which is
right: at 300 respondents the ruler is pinned down and all three agree. And it
is proportional to `|theta_j|`, so it widens the intervals on the extreme
models and leaves the middle of the pack alone — which is also right, since a
scale error is exactly what you cannot detect near zero.

It is still short of 95%, and the shortfall has the same cause as the items':
these are Gaussian intervals on a posterior that is not Gaussian, and the
scale correction is itself a first-order approximation.
"""

from __future__ import annotations

import math

import torch

from irtcheck.artifact import Posterior
from irtcheck.fit.model import (
    PRIORS_VAGUE,
    VAGUE_A_SCALE,
    VAGUE_B_SCALE,
    ResponseTensors,
)
from irtcheck.fit.summaries import Z95

# The information matrix is accumulated in float64 regardless of the dtype the
# fit ran in. `w_ij` underflows for a confident prediction, and the determinant
# is a difference of products of sums — the one place in this file where float32
# would cost real digits.
DTYPE = torch.float64


def prior_precisions(
    hyperparameters: dict[str, float], *, priors: str
) -> tuple[float, float]:
    """1/sigma^2 for `a` and `b`, from whichever prior family the fit used.

    The hierarchical scales are learned, so they come from the fitted guide; the
    vague ones are the fixed constants the model declares. Either way these are
    the precisions the population prior contributes to every item, and they are
    what an item with no responses falls back to.
    """
    if priors == PRIORS_VAGUE:
        return 1.0 / VAGUE_A_SCALE**2, 1.0 / VAGUE_B_SCALE**2
    sigma_a = float(hyperparameters["sigma_a"])
    sigma_b = float(hyperparameters["sigma_b"])
    return 1.0 / sigma_a**2, 1.0 / sigma_b**2


def respondent_posterior(
    data: ResponseTensors,
    *,
    a_mean: list[float],
    b_mean: list[float],
    theta_mean: list[float],
) -> Posterior:
    """A posterior for `theta`, from its information plus the scale uncertainty.

    See the module docstring for the derivation. The mean is the variational
    one; this replaces the width only.
    """
    a = torch.as_tensor(a_mean, dtype=DTYPE)
    b = torch.as_tensor(b_mean, dtype=DTYPE)
    theta = torch.as_tensor(theta_mean, dtype=DTYPE)

    rows = data.rows.to(torch.long)
    slope = a[data.cols.to(torch.long)]
    p = torch.sigmoid(slope * (theta[rows] - b[data.cols.to(torch.long)]))
    w = p * (1.0 - p)
    # The prior contributes precision 1, because theta ~ N(0, 1) is fixed. So a
    # respondent who answered nothing gets sd 1 -- the prior -- rather than a
    # division by zero.
    information = torch.zeros(data.n_respondents, dtype=DTYPE).index_add_(
        0, rows, slope * slope * w
    ) + 1.0

    # Relative standard error of the sample sd of n draws. Clamped at n = 2:
    # with a single respondent the scale is not estimated at all, and 1/sqrt(2)
    # is the widest this term is ever allowed to be rather than infinite.
    n = max(int(data.n_respondents) - 1, 1)
    relative = 1.0 / math.sqrt(2.0 * n)
    variance = 1.0 / information + (theta * relative) ** 2
    return _posterior(theta, variance.sqrt())


def item_posteriors(
    data: ResponseTensors,
    *,
    a_mean: list[float],
    b_mean: list[float],
    theta_mean: list[float],
    precision_a: float,
    precision_b: float,
) -> tuple[Posterior, Posterior]:
    """Posteriors for `a` and `b`, keeping the fitted means and recomputing the width.

    The means are the variational ones — this changes how wide the interval is,
    not where it sits. `Posterior.hdi_*` are `mean +/- 1.96 sd`, which for a
    Normal is the highest-density interval exactly, matching the convention
    `summaries.normal_posterior` and `synth.py` both use.
    """
    a = torch.as_tensor(a_mean, dtype=DTYPE)
    b = torch.as_tensor(b_mean, dtype=DTYPE)
    theta = torch.as_tensor(theta_mean, dtype=DTYPE)

    cols = data.cols.to(torch.long)
    ability = theta[data.rows.to(torch.long)]
    centred = ability - b[cols]
    slope = a[cols]
    p = torch.sigmoid(slope * centred)
    w = p * (1.0 - p)

    def accumulate(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(data.n_items, dtype=DTYPE).index_add_(0, cols, values)

    i_aa = accumulate(w * centred * centred) + precision_a
    i_ab = -accumulate(slope * w * centred)
    i_bb = accumulate(w * slope * slope) + precision_b
    # Positive definite for every item, by the argument in the module docstring.
    # The clamp is a floor against float64 underflow, not a correction.
    det = (i_aa * i_bb - i_ab * i_ab).clamp(min=torch.finfo(DTYPE).tiny)

    return (
        _posterior(a, (i_bb / det).sqrt()),
        _posterior(b, (i_aa / det).sqrt()),
    )


def _posterior(mean: torch.Tensor, sd: torch.Tensor) -> Posterior:
    low, high = mean - Z95 * sd, mean + Z95 * sd
    return Posterior(
        mean=[float(x) for x in mean],
        sd=[float(x) for x in sd],
        hdi_low=[float(x) for x in low],
        hdi_high=[float(x) for x in high],
    )
