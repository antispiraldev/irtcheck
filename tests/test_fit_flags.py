"""Item flags come from artifact.compute_flags(), and only from there.

synth.py flags its fabricated fits with the same function. Three wave-1 briefs
build against those fabrications, so a second implementation here — however
obviously equivalent when written — would mean they had been testing a fiction,
and nothing would fail loudly on the day the two drifted apart.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyro", reason="the fitter and its tests need torch and pyro")

from irtcheck.artifact import (  # noqa: E402
    ALL_FLAGS,
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
    Posterior,
    compute_flags,
)
from irtcheck.fit.flags import flag_counts, item_flags  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def posteriors():
    #                undetermined   dead      solid     solid
    a = Posterior(
        mean=[0.05, 0.10, 1.40, 1.40],
        sd=[0.50, 0.05, 0.20, 0.20],
        hdi_low=[-0.93, 0.01, 1.01, 1.01],
        hdi_high=[1.03, 0.19, 1.79, 1.79],
    )
    b = Posterior(
        mean=[0.0, 0.0, 0.0, 9.0],
        sd=[0.3, 0.3, 0.3, 0.3],
        hdi_low=[-0.6, -0.6, -0.6, 8.4],
        hdi_high=[0.6, 0.6, 0.6, 9.6],
    )
    return a, b


def test_the_fitter_delegates_flagging_wholesale(posteriors):
    """Byte-for-byte the same answer as the shared implementation, on inputs
    that exercise every branch."""
    a, b = posteriors
    p_correct = [0.5, 0.5, 1.0, 0.0]
    theta = [-1.2, 0.0, 1.1]
    assert item_flags(a, b, p_correct, theta) == compute_flags(a, b, p_correct, theta)


def test_the_two_uncertainty_flags_are_never_both_set(posteriors):
    """`dead` is a finding about the suite, `insufficient-data` a finding about
    the user's data. Claiming both about one item is incoherent, and
    IrtFit.validate() rejects it."""
    a, b = posteriors
    flags = item_flags(a, b, [0.5, 0.5, 1.0, 0.0], [-1.0, 0.0, 1.0])
    for item in flags:
        assert not (FLAG_DEAD in item and FLAG_INSUFFICIENT_DATA in item)


def test_an_interval_spanning_zero_is_refused_not_ranked(posteriors):
    a, b = posteriors
    flags = item_flags(a, b, [0.5, 0.5, 1.0, 0.0], [-1.0, 0.0, 1.0])
    assert FLAG_INSUFFICIENT_DATA in flags[0]
    assert FLAG_DEAD in flags[1]
    assert FLAG_CEILING in flags[2]
    assert FLAG_FLOOR in flags[3]


def test_flag_counts_report_every_flag_including_the_zeros(posteriors):
    """A caller reading diagnostics should not have to tell "no dead items"
    apart from "this fitter never sets that flag"."""
    a, b = posteriors
    counts = flag_counts(item_flags(a, b, [0.5, 0.5, 1.0, 0.0], [-1.0, 0.0, 1.0]))
    assert set(counts) == set(ALL_FLAGS)
    assert counts[FLAG_INSUFFICIENT_DATA] == 1
    assert counts[FLAG_DEAD] == 1
    assert counts[FLAG_CEILING] == 1
    assert counts[FLAG_FLOOR] == 1


def test_flag_counts_of_an_unflagged_suite_are_all_zero():
    assert flag_counts([[], [], []]) == dict.fromkeys(ALL_FLAGS, 0)


def test_a_real_fit_and_a_fabricated_one_flag_by_the_same_rules():
    """The property the other three wave-1 briefs depend on."""
    import numpy as np

    from irtcheck.fit.fitter import fit_2pl
    from irtcheck.synth import synthetic_matrix

    matrix, _ = synthetic_matrix(n_models=8, n_items=60, seed=9)
    fit = fit_2pl(matrix, epochs=400, seed=0)
    recomputed = compute_flags(
        fit.a, fit.b, fit.p_correct, fit.theta.mean
    )
    assert fit.flags == recomputed
    assert fit.p_correct == [float(p) for p in np.nan_to_num(matrix.item_p_correct(), nan=0.0)]
