"""Posterior summaries: mean, sd, and the interval everything downstream reads.

`insufficient-data` is decided entirely by whether the interval on `a_i`
contains zero, so an HDI that is subtly wrong is not a cosmetic problem — it is
the difference between an item being ranked and being refused.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("pyro", reason="the fitter and its tests need torch and pyro")

import torch  # noqa: E402

from irtcheck.fit.summaries import (  # noqa: E402
    Z95,
    hdi,
    normal_posterior,
    posterior_for,
    sampled_posterior,
)
from irtcheck.synth import Z95 as SYNTH_Z95  # noqa: E402

pytestmark = pytest.mark.slow


def test_the_interval_convention_matches_synth():
    """synth.py fabricates the artifacts the other wave-1 agents build against.
    If its intervals and ours mean different things, everything they tested is
    testing a fiction."""
    assert Z95 == SYNTH_Z95


# -- closed form -------------------------------------------------------------


def test_a_normal_posterior_is_mean_plus_minus_1_96_sd():
    post = normal_posterior(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 0.5]))
    assert post.mean == pytest.approx([0.0, 2.0])
    assert post.sd == pytest.approx([1.0, 0.5])
    assert post.hdi_low == pytest.approx([-Z95, 2.0 - 0.5 * Z95])
    assert post.hdi_high == pytest.approx([Z95, 2.0 + 0.5 * Z95])


def test_a_normal_posterior_centred_on_zero_spans_zero():
    """The refusal case, stated as a property rather than a number."""
    post = normal_posterior(torch.tensor([0.1, 1.5]), torch.tensor([0.4, 0.2]))
    assert post.spans_zero() == [True, False]


def test_interval_ends_stay_ordered():
    post = normal_posterior(torch.tensor([1.0]), torch.tensor([-0.5]))
    assert post.hdi_low[0] < post.hdi_high[0]
    post.validate(1, "a")


# -- empirical HDI -----------------------------------------------------------


def test_hdi_of_a_symmetric_sample_is_the_equal_tailed_interval():
    samples = torch.distributions.Normal(0.0, 1.0).sample((40_000,))
    low, high = hdi(samples)
    assert low == pytest.approx(-1.96, abs=0.1)
    assert high == pytest.approx(1.96, abs=0.1)


def test_hdi_of_a_skewed_sample_is_narrower_than_the_quantile_interval():
    """The whole reason to compute an HDI rather than two quantiles: on a
    lognormal they differ, and only one of them is the *highest density*
    interval."""
    samples = torch.distributions.LogNormal(0.0, 1.0).sample((40_000,))
    low, high = hdi(samples)
    q_low = float(torch.quantile(samples, 0.025))
    q_high = float(torch.quantile(samples, 0.975))
    assert high - low < q_high - q_low
    assert low < q_low  # mass is pushed toward the short left tail


def test_hdi_covers_the_requested_share_of_the_sample():
    samples = torch.distributions.Normal(0.0, 1.0).sample((5_000,))
    low, high = hdi(samples, 0.5)
    covered = ((samples >= low) & (samples <= high)).float().mean()
    assert covered == pytest.approx(0.5, abs=0.01)


def test_hdi_of_a_degenerate_sample():
    assert hdi(torch.tensor([2.0])) == (2.0, 2.0)
    assert hdi(torch.full((100,), 3.0)) == (3.0, 3.0)


def test_hdi_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        hdi(torch.tensor([]))


def test_sampled_posterior_recovers_the_distribution_it_was_drawn_from():
    torch.manual_seed(0)
    samples = torch.distributions.Normal(
        torch.tensor([0.0, 5.0]), torch.tensor([1.0, 2.0])
    ).sample((20_000,))
    post = sampled_posterior(samples)
    assert post.mean == pytest.approx([0.0, 5.0], abs=0.05)
    assert post.sd == pytest.approx([1.0, 2.0], abs=0.05)
    assert post.hdi_low == pytest.approx([-1.96, 5.0 - 3.92], abs=0.15)


# -- against a real guide ----------------------------------------------------


def test_posterior_for_reads_an_autonormal_guide_exactly():
    """AutoNormal keeps a Normal marginal on every unconstrained site, so the
    summaries are closed-form and a fit is deterministic given its seed —
    no Monte Carlo jitter in the numbers a flag turns on."""
    import pyro

    from irtcheck.fit.model import make_guide, make_model, to_tensors
    from irtcheck.synth import synthetic_matrix

    matrix, _ = synthetic_matrix(n_models=5, n_items=20, seed=1)
    data = to_tensors(matrix)
    pyro.clear_param_store()
    model = make_model("hierarchical")
    guide = make_guide(model, data, priors="hierarchical")
    guide(data)

    post = posterior_for(guide, "a", data=data)
    assert len(post) == matrix.n_items
    locs = guide.locs.a.detach()
    scales = guide.scales.a.detach()
    assert post.mean == pytest.approx(locs.tolist(), abs=1e-6)
    assert post.sd == pytest.approx(scales.tolist(), abs=1e-6)
    for i in range(len(post)):
        assert post.hdi_high[i] - post.hdi_low[i] == pytest.approx(2 * Z95 * post.sd[i], abs=1e-6)


def test_a_constrained_site_is_summarised_from_samples_instead():
    """`sigma_a` is positive, so its guide marginal is a transformed normal and
    exponentiating an equal-tailed interval would not be an HDI. The fallback is
    structural — it depends on the site's support, not on a flag."""
    import pyro

    from irtcheck.fit.model import make_guide, make_model, to_tensors
    from irtcheck.synth import synthetic_matrix

    matrix, _ = synthetic_matrix(n_models=5, n_items=20, seed=1)
    data = to_tensors(matrix)
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    model = make_model("hierarchical")
    guide = make_guide(model, data, priors="hierarchical")
    guide(data)

    post = posterior_for(guide, "sigma_a", n_samples=400, data=data)
    assert len(post) == 1
    assert post.hdi_low[0] > 0.0  # a scale cannot be negative
    assert post.hdi_low[0] < post.mean[0] < post.hdi_high[0]
    assert math.isfinite(post.sd[0])
