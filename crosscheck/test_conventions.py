"""Tests for the convention reconciliation, and for the vacuity guards.

    .venv/bin/python -m pytest crosscheck/test_conventions.py

Not part of the project's suite — `pyproject.toml` sets `testpaths = ["tests"]`,
so these run only when pointed at explicitly, which is right for a directory
excluded from the distribution.

They matter anyway. Every number this directory reports passes through
`conventions.py`, so a silent bug there would corrupt the whole comparison
while leaving it looking clean. Two kinds of test:

  - each invariance is checked against a **likelihood that must not move**. A
    transformation claimed to be the identity either leaves every response
    probability unchanged or it is not the identity, and that is checkable
    without reference to any fitted model.
  - the vacuity guards in `compare.py` are checked by feeding them the exact
    degenerate inputs they exist to catch. A guard nobody has ever seen fire is
    indistinguishable from a guard that cannot fire — which is precisely how
    `test_no_two_commands_disagree_about_a_flag` passed vacuously through the
    whole of wave 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from crosscheck.compare import (
    CANARY_PAIR,
    VacuousComparison,
    assert_comparable,
    check_canary,
    checked_agreement,
)
from crosscheck.conventions import (
    agreement,
    align,
    probabilities,
    reflect,
    reflection_agrees,
    slope_intercept_to_difficulty,
    standardise_theta_metric,
)

RNG = np.random.default_rng(20260912)
N_ITEMS, N_RESP = 25, 40


def a_b_theta() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = RNG.lognormal(0.0, 0.4, size=N_ITEMS)
    b = RNG.normal(0.0, 1.2, size=N_ITEMS)
    theta = RNG.normal(0.0, 1.0, size=N_RESP)
    return a, b, theta


# --- the invariances, checked on the likelihood ------------------------------


def test_reflection_leaves_every_probability_unchanged():
    a, b, theta = a_b_theta()
    base = probabilities(a, b, theta)
    mirrored = probabilities(*reflect(a, b, theta))
    assert np.allclose(base, mirrored, atol=1e-12)


def test_rescaling_leaves_every_probability_unchanged():
    """The scale/location invariance `standardise_theta_metric` relies on."""
    a, b, theta = a_b_theta()
    base = probabilities(a, b, theta)
    # Put theta on a deliberately wrong metric, as py-irt's hierarchical model
    # does, then standardise it back.
    c, d = 2.37, -0.84
    a2, b2, theta2, scale, loc = standardise_theta_metric(a / c, b * c + d, theta * c + d)
    assert np.allclose(probabilities(a2, b2, theta2), base, atol=1e-10)
    # And it reports what it did, which the report prints.
    assert scale == pytest.approx(c * theta.std(ddof=0), rel=1e-9)
    assert loc == pytest.approx(theta.mean() * c + d, rel=1e-9, abs=1e-12)


def test_standardise_actually_puts_theta_on_mean_zero_sd_one():
    a, b, theta = a_b_theta()
    _, _, theta2, _, _ = standardise_theta_metric(a, b, theta * 3.0 + 5.0)
    assert theta2.mean() == pytest.approx(0.0, abs=1e-12)
    assert theta2.std(ddof=0) == pytest.approx(1.0, rel=1e-12)


def test_slope_intercept_conversion_is_the_identity_on_the_likelihood():
    """b = -d/a1 is a notation change, so the probabilities must be identical.

    This is the trap docs/spec.md names, and run_mirt.py additionally checks the
    conversion against mirt's own IRTpars output on every real run.
    """
    a, b, theta = a_b_theta()
    d = -a * b  # the slope-intercept form of the same items
    a_back, b_back = slope_intercept_to_difficulty(a, d)
    assert np.allclose(a_back, a, atol=1e-12)
    assert np.allclose(b_back, b, atol=1e-12)
    direct = 1.0 / (1.0 + np.exp(-(a[None, :] * theta[:, None] + d[None, :])))
    assert np.allclose(direct, probabilities(a_back, b_back, theta), atol=1e-12)


def test_slope_intercept_refuses_to_invent_a_difficulty_for_a_zero_slope():
    """A dead item has no difficulty; NaN says so rather than an inf."""
    a, b = slope_intercept_to_difficulty(np.array([0.0, 1.0]), np.array([0.5, -2.0]))
    assert np.isnan(b[0])
    assert b[1] == pytest.approx(2.0)
    assert a[0] == 0.0


def test_reflection_agrees_detects_the_mirror():
    _, _, theta = a_b_theta()
    assert reflection_agrees(theta, theta * 1.4)
    assert not reflection_agrees(theta, -theta * 1.4)


# --- alignment ---------------------------------------------------------------


def test_align_reorders_and_nans_what_is_missing():
    ref = ["c", "a", "b", "zz"]
    other = ["a", "b", "c"]
    (out,) = align(ref, other, np.array([10.0, 20.0, 30.0]))
    assert out[0] == 30.0 and out[1] == 10.0 and out[2] == 20.0
    assert np.isnan(out[3])


def test_agreement_drops_positions_either_side_left_nan():
    ref = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    other = ref * 2.0
    other[1] = np.nan
    result = agreement(ref, other)
    assert result.n == 9
    assert result.pearson == pytest.approx(1.0)
    assert result.slope == pytest.approx(2.0)


def test_agreement_through_origin_recovers_a_pure_scale_factor():
    ref = RNG.normal(0.0, 1.0, size=50)
    result = agreement(ref, ref * 0.6, through_origin=True)
    assert result.slope == pytest.approx(0.6, rel=1e-9)
    assert result.intercept == 0.0
    assert result.rmse_after_slope == pytest.approx(0.0, abs=1e-9)


# --- the guards against a comparison that compares nothing -------------------


def test_assert_comparable_accepts_a_real_comparison():
    ref = RNG.normal(size=30)
    assert assert_comparable("ok", ref, ref + RNG.normal(scale=0.1, size=30)) == 30


@pytest.mark.parametrize(
    ("why", "ref", "other"),
    [
        ("mismatched shapes", np.zeros(5), np.zeros(6)),
        ("too few overlapping", np.array([1.0, 2.0, np.nan]), np.array([1.0, np.nan, 3.0])),
        ("constant reference", np.ones(30), RNG.normal(size=30)),
        ("constant subject", RNG.normal(size=30), np.full(30, 2.5)),
        ("all-zero subject", RNG.normal(size=30), np.zeros(30)),
        ("all NaN", np.full(30, np.nan), np.full(30, np.nan)),
    ],
)
def test_assert_comparable_refuses_a_vacuous_comparison(why, ref, other):
    with pytest.raises(VacuousComparison):
        assert_comparable(why, np.asarray(ref, dtype=float), np.asarray(other, dtype=float))


def test_assert_comparable_refuses_bit_identical_sides():
    """The shape a no-op conversion takes: the output is the input."""
    ref = RNG.normal(size=30)
    with pytest.raises(VacuousComparison, match="bit-identical"):
        assert_comparable("identity", ref, ref.copy())


def test_the_shuffled_control_collapses_on_a_real_comparison():
    """A genuine agreement must beat every permutation of its own reference.

    This is the check on the check. If a comparison scored the same against the
    reference and against a shuffle of it, the harness would not be reading the
    data.
    """
    ref = RNG.normal(size=60)
    checked = checked_agreement("real", ref, ref * 1.3 + RNG.normal(scale=0.05, size=60))
    assert checked.real.pearson > 0.99
    assert checked.shuffled_pearson < 0.30
    assert checked.beats_control


def test_an_unrelated_pair_does_not_beat_its_control():
    """Two independent vectors score no better than a permutation, at small n.

    Reported as NO SIGNAL rather than as a harness fault — see
    `Checked.beats_control` for why that distinction is load-bearing.
    """
    ref = RNG.normal(size=14)
    checked = checked_agreement("noise", ref, RNG.normal(size=14))
    assert not checked.beats_control, (
        f"unrelated vectors should not beat their control, got r={checked.real.pearson:+.3f} "
        f"against a worst control of {checked.shuffled_worst:.3f}"
    )


def test_the_canary_fails_when_a_comparison_is_missing():
    """A report with no known-answer comparison in it must not pass.

    The failure mode the coordinator flagged: a harness that stopped comparing
    would report agreement everywhere and look healthy. The canary is the only
    defence against that, so it has to fail loudly when it cannot run.
    """
    assert check_canary({"dense": {"comparisons": {}}}) == 1
    assert check_canary({}) == 1


def test_the_canary_fails_on_a_reference_that_stopped_agreeing():
    subject, reference = CANARY_PAIR
    entry = {
        "n": 59,
        "pearson": 0.81,  # two implementations of one estimator cannot score this
        "rmse": 0.4,
        "shuffled_pearson_worst": 0.2,
        "beats_shuffled_control": True,
    }
    summary = {"dense": {"comparisons": {f"a:{subject}-vs-{reference}": entry}}}
    assert check_canary(summary) == 1


def test_the_canary_passes_on_a_healthy_report():
    subject, reference = CANARY_PAIR
    entry = {
        "n": 59,
        "pearson": 0.9999999,
        "rmse": 1e-6,
        "shuffled_pearson_worst": 0.21,
        "beats_shuffled_control": True,
    }
    summary = {
        "dense": {
            "comparisons": {
                f"{kind}:{subject}-vs-{reference}": entry for kind in ("a", "b", "theta")
            }
        }
    }
    assert check_canary(summary) == 0
