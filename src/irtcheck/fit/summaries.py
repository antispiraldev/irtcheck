"""Turning a variational posterior into the `Posterior` the artifact carries.

Mean, sd and a 95% **highest-density interval** for every `a`, `b` and `theta`.
The intervals are not garnish: `insufficient-data` — the flag that decides
whether an item is ranked at all — is precisely the question of whether the
interval on `a_i` contains zero. Everything downstream of `fit` reads these
four lists and nothing else about the posterior.

Two paths, and which one is taken is a property of the guide rather than a
setting:

  - a site whose guide marginal is Normal on an unconstrained support has an
    exact answer, and for a Normal the HDI *is* the equal-tailed interval:
    mean +/- 1.96 sd. That is the path AutoNormal takes for `a`, `b` and
    `theta` here, and it makes a fit deterministic given its seed — no Monte
    Carlo jitter in the numbers a report prints or a flag turns on.

  - anything else is summarised from samples, with the HDI computed as the
    narrowest interval covering 95% of them. A LogNormal or otherwise
    transformed marginal is skewed, and exponentiating an equal-tailed normal
    interval would silently report something that is not an HDI at all.

synth.py fabricates its posteriors with the same Z95 convention, so a
fabricated artifact and a real one describe intervals that mean the same thing.
"""

from __future__ import annotations

import math

import torch

from irtcheck.artifact import Posterior

# 95%, two-sided. Matches synth.Z95 deliberately: the fabricated fits the other
# wave-1 agents build against have to mean the same thing as a real one.
Z95 = 1.959963985
HDI_PROB = 0.95


def normal_posterior(loc: torch.Tensor, scale: torch.Tensor) -> Posterior:
    """Exact summaries for a Normal marginal on an unconstrained support."""
    loc = loc.detach().reshape(-1).to("cpu", torch.float64)
    scale = scale.detach().reshape(-1).to("cpu", torch.float64).abs()
    low, high = loc - Z95 * scale, loc + Z95 * scale
    return Posterior(
        mean=[float(x) for x in loc],
        sd=[float(x) for x in scale],
        hdi_low=[float(x) for x in low],
        hdi_high=[float(x) for x in high],
    )


def hdi(samples: torch.Tensor, prob: float = HDI_PROB) -> tuple[float, float]:
    """The narrowest interval containing `prob` of `samples`.

    The textbook sorted-window scan. For a symmetric marginal it agrees with
    the equal-tailed interval; for a skewed one it does not, and the difference
    is the whole reason this exists rather than a pair of quantiles.
    """
    ordered, _ = torch.sort(samples.detach().reshape(-1).to("cpu", torch.float64))
    n = ordered.numel()
    if n == 0:
        raise ValueError("cannot summarise an empty sample")
    if n == 1:
        return float(ordered[0]), float(ordered[0])
    # Number of points the interval must span, at least two so it has ends.
    window = max(2, int(math.ceil(prob * n)))
    if window >= n:
        return float(ordered[0]), float(ordered[-1])
    widths = ordered[window - 1 :] - ordered[: n - window + 1]
    start = int(torch.argmin(widths))
    return float(ordered[start]), float(ordered[start + window - 1])


def sampled_posterior(samples: torch.Tensor, prob: float = HDI_PROB) -> Posterior:
    """Summaries from draws shaped (n_samples, n_parameters)."""
    samples = samples.detach().to("cpu", torch.float64)
    if samples.dim() == 1:
        samples = samples.unsqueeze(1)
    samples = samples.reshape(samples.shape[0], -1)
    mean = samples.mean(dim=0)
    # Sample sd, not population sd: these are draws, not the whole posterior.
    sd = samples.std(dim=0, unbiased=True)
    bounds = [hdi(samples[:, i], prob) for i in range(samples.shape[1])]
    return Posterior(
        mean=[float(x) for x in mean],
        sd=[float(x) for x in sd],
        hdi_low=[lo for lo, _ in bounds],
        hdi_high=[hi for _, hi in bounds],
    )


def posterior_for(
    guide,
    site: str,
    *,
    n_samples: int = 2000,
    data=None,
) -> Posterior:
    """Summarise one latent site of an AutoNormal-style guide.

    Takes the exact path when the guide holds a Normal marginal on an
    unconstrained support for `site`, and falls back to sampling otherwise —
    so changing the guide changes the cost of this function, never its
    correctness.
    """
    loc, scale = _normal_params(guide, site)
    if loc is not None and scale is not None:
        return normal_posterior(loc, scale)
    return sampled_posterior(_draw(guide, site, n_samples=n_samples, data=data))


def _normal_params(guide, site: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(loc, scale) if this site's guide marginal is an untransformed Normal.

    Returns (None, None) — meaning "summarise this one from samples" — when the
    site is constrained, because then a transform sits between the guide's
    Normal and the value the artifact reports, and an exponentiated
    equal-tailed interval is not an HDI. Structural, not a setting: change the
    guide and this changes what it answers.
    """
    from torch.distributions import constraints

    locs, scales = getattr(guide, "locs", None), getattr(guide, "scales", None)
    if locs is None or scales is None or not hasattr(locs, site):
        return None, None
    trace = getattr(guide, "prototype_trace", None)
    if trace is None or trace.nodes[site]["fn"].support is not constraints.real:
        return None, None
    return getattr(locs, site), getattr(scales, site)


def _draw(guide, site: str, *, n_samples: int, data=None) -> torch.Tensor:
    """`n_samples` draws of one site from the guide, shaped (n_samples, size)."""
    draws = []
    with torch.no_grad():
        for _ in range(n_samples):
            sample = guide(data) if data is not None else guide()
            if site not in sample:
                raise KeyError(f"the guide has no site {site!r}; got {sorted(sample)}")
            draws.append(sample[site].detach().reshape(-1))
    return torch.stack(draws)
