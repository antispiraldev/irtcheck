"""The 2PL, as a Pyro model and a mean-field guide.

    P(correct) = sigmoid(a_i * (theta_j - b_i))

Written against Pyro directly rather than through py-irt; see
docs/build-plan.html finding F-1 for why that dependency could not survive.

## Identification

The likelihood is invariant under two transformations, and a fit that pins
neither down returns numbers that are not comparable between runs, let alone
against `mirt`:

  - **scale and location**: (a, b, theta) -> (a/c, c*b + d, c*theta + d) leaves
    every probability unchanged. Fixed by giving theta a *fixed* standard
    normal prior — not a hierarchical one with a learned mean and variance.
    theta ~ N(0, 1) is the ruler; a and b are measured against it.

  - **reflection**: negating a, theta and b together also leaves every
    probability unchanged, so the sign of every discrimination can flip as a
    block with the ability scale. Fixed twice over: the guide starts every item
    at a = +1, which puts mode-seeking SVI in the positive basin, and
    `canonical_sign()` in fitter.py checks the fitted mean afterwards and flips
    the whole solution if it somehow ended up in the mirror one.

`IrtFit.model["identification"]` records the convention on every artifact, for
the wave-2 cross-check against mirt and py-irt.

## Why `a` is not hard-constrained positive, and why it pools toward zero

The obvious way to break the reflection is a LogNormal prior on `a`, and it is
wrong *for this tool*. With a strictly positive `a` the 95% interval can never
contain zero, and `insufficient-data` — the flag carrying the spec's entire
"refusal is a feature" argument, and the one CLAUDE.md says most low-information
items should land in at 5-15 respondents — becomes unreachable. An item nobody
has enough evidence about would come back with a cheerful interval of
[0.05, 2.4] and be ranked as though it were fine.

The same argument rules out the textbook hierarchical prior
`a_i ~ Normal(mu_a, sigma_a)` with a learned positive `mu_a`. Measured on
synthetic 15 x 500 matrices, that prior pools every item to a ~ 1.2 +/- 0.3:
truly dead items came back at 0.95, no item's interval spanned zero, and
frequentist coverage of the true `a` fell to 60%. It is pooling working exactly
as designed and destroying the product — the estimate for an item we know
nothing about was the suite average, stated confidently.

So the population prior for `a` is centred at **zero** with a learned suite-level
scale:

    sigma_a ~ LogNormal(0, 0.5)        how sharply this suite's items separate
    a_i     ~ Normal(0, sigma_a)

That is still partial pooling — a sparse item is pulled toward the suite-level
distribution rather than fitted on its own thin evidence, which is what the spec
asks for — but the thing it is pulled *toward* is "does not discriminate". An
item earns a discrimination from its data or it does not get one, and when it
does not, its interval spans zero and the tool says so. On the same synthetic
matrices this recovers coverage of 91% against a nominal 95%, flags 93% of the
genuinely dead items as insufficient-data at 15 respondents against 53% of the
live ones, and separates them further as respondents are added (84% vs 11% at
60). Difficulty has no such tension and keeps the conventional form,
`b_i ~ Normal(mu_b, sigma_b)` with both learned.

The cost is a downward bias on `a`, and it is the right direction to be wrong
in: it makes the tool slower to claim an item discriminates, never quicker.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import pyro
import pyro.distributions as dist
import torch

# The two prior families `--priors` selects between.
PRIORS_HIERARCHICAL = "hierarchical"
PRIORS_VAGUE = "vague"
ALL_PRIORS = (PRIORS_HIERARCHICAL, PRIORS_VAGUE)

# Where the guide starts. `a` at +1 is half of the reflection fix above; `b`
# and `theta` come from the data via classical_start().
INIT_A = 1.0
INIT_SIGMA_A = 1.0
INIT_SIGMA_B = 1.0
INIT_SCALE = 0.1
# Observed rates are clamped away from 0 and 1 before being turned into
# starting values, because logit(0) is not a number an optimiser can start at.
RATE_CLAMP = 0.02
START_B_LIMIT = 6.0

# Vague priors: fixed, wide, and no pooling whatsoever, so a thin item is
# estimated on its own evidence and can come back extreme. That is the
# comparison `--priors vague` exists to make, not a second-best default;
# centring `a` at +1 keeps the reflection broken the same way.
VAGUE_A_LOC, VAGUE_A_SCALE = 1.0, 2.0
VAGUE_B_LOC, VAGUE_B_SCALE = 0.0, 3.0

# Hyperpriors. LogNormal rather than HalfNormal for the scales: HalfNormal's
# mode is at zero, which invites a mean-field guide to collapse the population
# spread to nothing and pool every item into a single point.
SIGMA_A_PRIOR = (0.0, 0.5)  # LogNormal: median 1.0, 95% within [0.37, 2.7]
SIGMA_B_PRIOR = (0.0, 0.5)
MU_B_PRIOR = (0.0, 1.0)


class ModelError(ValueError):
    pass


def check_priors(priors: str) -> str:
    if priors not in ALL_PRIORS:
        raise ModelError(
            f"unknown --priors {priors!r}. Choose one of: {', '.join(ALL_PRIORS)}. "
            "'hierarchical' pools item parameters toward the suite-level "
            "distribution, which is what keeps a sparse item from producing an "
            "extreme point estimate; 'vague' estimates every item on its own."
        )
    return priors


@dataclass(slots=True)
class ResponseTensors:
    """A ResponseMatrix as the three tensors the model plates over.

    Sparse triplets throughout. Real eval matrices are ragged — models get
    re-run on subsets, harnesses drop failures — so there is no rectangle to
    densify into and no missing-value mask to get wrong. One observation is one
    entry in `obs`, and its (respondent, item) coordinates are `rows[k]` and
    `cols[k]`.
    """

    rows: torch.Tensor  # long, index into respondents
    cols: torch.Tensor  # long, index into items
    obs: torch.Tensor  # float, 0.0 or 1.0
    n_respondents: int
    n_items: int

    @property
    def n_responses(self) -> int:
        return int(self.obs.numel())


def to_tensors(matrix, *, device: str | torch.device = "cpu") -> ResponseTensors:
    """Move a ResponseMatrix onto a device without densifying it."""
    device = torch.device(device)
    return ResponseTensors(
        rows=torch.as_tensor(matrix.rows, dtype=torch.long, device=device),
        cols=torch.as_tensor(matrix.cols, dtype=torch.long, device=device),
        obs=torch.as_tensor(matrix.obs, dtype=torch.float32, device=device),
        n_respondents=matrix.n_respondents,
        n_items=matrix.n_items,
    )


def classical_start(data: ResponseTensors) -> tuple[torch.Tensor, torch.Tensor]:
    """Starting values for `b` and `theta` from the observed rates alone.

    b_i = -logit(p_i) and theta_j = standardised logit(accuracy_j): the
    classical estimates every piece of IRT software starts from, and the
    starting point mirt uses.

    This is not a speed optimisation, it is what makes the hierarchical fit
    work on a ragged matrix. Pooling `a` toward zero puts a second optimum in
    reach: every a_i drawn from its prior with random signs, every b_i at the
    population mean, and abilities collapsed — a solution that predicts 0.5 for
    everything and, once responses thin out, has a *higher* ELBO than the true
    structure, because it pays no KL for 2 x n_items item parameters. Measured
    on a 15 x 500 matrix with 30% of cells missing, starting from zeros landed
    there every time: recovery correlation 0.005 for `a`, 0.09 for `b`,
    abilities uncorrelated with accuracy. Starting from the classical estimates
    instead: 0.43 and 0.76, abilities at 0.96.

    It also reinforces the reflection convention, since theta starts positively
    correlated with the score a user would compute by hand.
    """
    counts_i = torch.bincount(data.cols, minlength=data.n_items).clamp(min=1)
    correct_i = torch.bincount(data.cols, weights=data.obs, minlength=data.n_items)
    p_item = (correct_i / counts_i).clamp(RATE_CLAMP, 1.0 - RATE_CLAMP)
    b0 = (-torch.log(p_item / (1.0 - p_item))).clamp(-START_B_LIMIT, START_B_LIMIT)

    counts_r = torch.bincount(data.rows, minlength=data.n_respondents).clamp(min=1)
    correct_r = torch.bincount(data.rows, weights=data.obs, minlength=data.n_respondents)
    accuracy = (correct_r / counts_r).clamp(RATE_CLAMP, 1.0 - RATE_CLAMP)
    theta0 = torch.log(accuracy / (1.0 - accuracy))
    # Standardised, because theta ~ N(0, 1) is the scale the whole fit is
    # measured against; starting anywhere else asks SVI to undo a translation
    # before it can learn anything.
    spread = theta0.std(unbiased=False) if theta0.numel() > 1 else torch.tensor(0.0)
    theta0 = (theta0 - theta0.mean()) / spread.clamp(min=1e-6)
    return b0.to(data.obs.dtype), theta0.to(data.obs.dtype)


def two_pl(data: ResponseTensors, *, priors: str = PRIORS_HIERARCHICAL) -> None:
    """The generative model.

    Item parameters are drawn per item, abilities per respondent, and the
    Bernoulli likelihood plates over *observations* rather than over a
    respondent x item rectangle — which is what makes a ragged matrix a
    non-event here.
    """
    check_priors(priors)
    zero = torch.tensor(0.0, device=data.obs.device)

    if priors == PRIORS_HIERARCHICAL:
        # The pooling. See the module docstring for why `a` pools toward zero
        # and `b` toward a learned mean.
        sigma_a = pyro.sample("sigma_a", dist.LogNormal(zero + SIGMA_A_PRIOR[0], SIGMA_A_PRIOR[1]))
        mu_b = pyro.sample("mu_b", dist.Normal(zero + MU_B_PRIOR[0], MU_B_PRIOR[1]))
        sigma_b = pyro.sample("sigma_b", dist.LogNormal(zero + SIGMA_B_PRIOR[0], SIGMA_B_PRIOR[1]))
        a_loc, a_scale = zero, sigma_a
        b_loc, b_scale = mu_b, sigma_b
    else:
        a_loc, a_scale = zero + VAGUE_A_LOC, VAGUE_A_SCALE
        b_loc, b_scale = zero + VAGUE_B_LOC, VAGUE_B_SCALE

    with pyro.plate("items", data.n_items):
        a = pyro.sample("a", dist.Normal(a_loc, a_scale))
        b = pyro.sample("b", dist.Normal(b_loc, b_scale))

    # Fixed, not learned. This is the ruler: see "Identification" above.
    with pyro.plate("respondents", data.n_respondents):
        theta = pyro.sample("theta", dist.Normal(zero, torch.ones_like(zero)))

    # `[..., idx]` rather than `[idx]` so a vectorised-particle ELBO, which adds
    # a leading sample dimension, indexes items rather than particles.
    logits = a[..., data.cols] * (theta[..., data.rows] - b[..., data.cols])
    with pyro.plate("responses", data.n_responses):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=data.obs)


def make_model(priors: str = PRIORS_HIERARCHICAL):
    """The model as a one-argument callable, so SVI hands it and the guide the
    same `data` and nothing else."""
    return partial(two_pl, priors=check_priors(priors))


def make_guide(model_fn, data: ResponseTensors, *, priors: str = PRIORS_HIERARCHICAL):
    """A mean-field normal guide, initialised in the positive-`a` mode.

    AutoNormal rather than a hand-rolled guide: the sites this model has are
    exactly the ones it handles well, and a hand-written guide is where sign and
    scale bugs hide — the build plan's stated reason for keeping one pair of
    hands inside this package.

    `init_to_value` is load-bearing, not cosmetic, for two separate reasons.
    The hierarchical prior on `a` is symmetric about zero, so the reflected
    solution has identical posterior density and starting every a_i at +1 is
    what decides which of the two SVI walks into — fitter.canonical_sign()
    checks the result rather than trusting it. And `b` and `theta` start at the
    classical estimates rather than at zero, without which a thin matrix
    converges to a degenerate no-structure optimum; see classical_start().
    """
    from pyro.infer.autoguide import AutoNormal, init_to_value

    device = data.obs.device
    b0, theta0 = classical_start(data)
    values = {
        "a": torch.full((data.n_items,), INIT_A, device=device),
        "b": b0,
        "theta": theta0,
    }
    if priors == PRIORS_HIERARCHICAL:
        values |= {
            "sigma_a": torch.tensor(INIT_SIGMA_A, device=device),
            "mu_b": torch.tensor(0.0, device=device),
            "sigma_b": torch.tensor(INIT_SIGMA_B, device=device),
        }
    return AutoNormal(
        model_fn,
        init_loc_fn=init_to_value(values=values),
        init_scale=INIT_SCALE,
    )


def model_metadata(priors: str) -> dict[str, str]:
    """What goes in `IrtFit.model`.

    `identification` is read by the wave-2 cross-check against mirt and py-irt.
    An unrecorded convention difference there is an afternoon of nobody being
    able to tell a bug from a parameterisation.
    """
    pooling = (
        "a_i ~ N(0, sigma_a) with sigma_a learned; b_i ~ N(mu_b, sigma_b) with both learned"
        if priors == PRIORS_HIERARCHICAL
        else f"a_i ~ N({VAGUE_A_LOC}, {VAGUE_A_SCALE}); b_i ~ N({VAGUE_B_LOC}, {VAGUE_B_SCALE})"
    )
    return {
        "kind": "2pl",
        "priors": priors,
        "population": pooling,
        # Deliberately more specific than the placeholder synth.py writes: `a`
        # is not constrained positive, which is what leaves `insufficient-data`
        # reachable. See the module docstring.
        "identification": (
            "theta ~ N(0, 1) fixed, which sets location and scale; a is estimated "
            "on the real line (not constrained positive), so a 95% interval on a "
            "may legitimately contain zero; the reflection (a, b, theta) -> "
            "(-a, -b, -theta) is resolved toward positive mean a"
        ),
        "inference": "svi/trace-elbo, mean-field normal guide",
    }
