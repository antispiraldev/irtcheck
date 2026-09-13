"""`inverted`: an item that discriminates backwards is not dead weight.

Split out of `dead` in schema 2. Under schema 1 the rule was
`hdi_high < DEAD_THRESHOLD`, which is satisfied by `[0.05, 0.30]` and by
`[-1.17, -0.22]` alike — so an item that reliably separates respondents *the
wrong way* was reported as "confidently does not discriminate" and offered up
as dead weight to drop.

That was not a corner case. On the real HELM matrix every single item flagged
`dead` reached it by the negative route — 32 of 3,551 at 95 respondents, 3 at
twelve — and none by the small-positive one, which needs `sd(a) < 0.089`. The
flag fired 35 times across two real fits and was describing the wrong thing
every time.

The rule is now three mutually exclusive readings of one interval:

    spans zero                  -> insufficient-data  (we cannot tell)
    inside (-0.35, +0.35)       -> dead               (confidently negligible)
    wholly below zero, wider    -> inverted           (confidently backwards)

No torch: `compute_flags` is in artifact.py and takes posteriors, so all of
this runs from fabricated intervals except the end-to-end test at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

from irtcheck.artifact import (
    DEAD_THRESHOLD,
    FLAG_DEAD,
    FLAG_INSUFFICIENT_DATA,
    FLAG_INVERTED,
    ArtifactError,
    Posterior,
    compute_flags,
)
from irtcheck.synth import synthetic_fit

THETA = [-1.0, 0.0, 1.0]


def interval(low: float, high: float) -> Posterior:
    mid = (low + high) / 2
    return Posterior(mean=[mid], sd=[(high - low) / 3.92], hdi_low=[low], hdi_high=[high])


def flags_for(low: float, high: float, *, p_correct: float = 0.5) -> list[str]:
    return compute_flags(interval(low, high), interval(-0.1, 0.1), [p_correct], THETA)[0]


# -- the three readings ------------------------------------------------------


@pytest.mark.parametrize(
    "low,high,expected",
    [
        # Confidently negligible, either side of zero.
        (0.05, 0.30, FLAG_DEAD),
        (-0.30, -0.10, FLAG_DEAD),
        (0.01, 0.34, FLAG_DEAD),
        # Confidently backwards, and not negligible.
        (-1.17, -0.22, FLAG_INVERTED),
        (-0.50, -0.10, FLAG_INVERTED),
        (-3.00, -1.00, FLAG_INVERTED),
        # Cannot tell.
        (-0.20, 0.40, FLAG_INSUFFICIENT_DATA),
        (-2.00, 2.00, FLAG_INSUFFICIENT_DATA),
        (0.00, 0.50, FLAG_INSUFFICIENT_DATA),
    ],
)
def test_each_interval_gets_the_reading_it_should(low, high, expected):
    assert expected in flags_for(low, high)


@pytest.mark.parametrize("low,high", [(0.40, 0.90), (0.36, 2.00), (1.00, 3.00)])
def test_a_healthy_item_carries_none_of_the_three(low, high):
    flags = set(flags_for(low, high))
    assert not flags & {FLAG_DEAD, FLAG_INVERTED, FLAG_INSUFFICIENT_DATA}


def test_the_three_readings_are_mutually_exclusive_across_the_whole_line():
    """Swept rather than spot-checked: every interval gets exactly one reading,
    or none, and never two. A gap would be as bad as an overlap — an item that
    falls through would be silently treated as healthy."""
    grid = np.linspace(-3.0, 3.0, 61)
    exclusive = {FLAG_DEAD, FLAG_INVERTED, FLAG_INSUFFICIENT_DATA}
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            got = exclusive.intersection(flags_for(float(low), float(high)))
            assert len(got) <= 1, f"[{low:.2f}, {high:.2f}] got {got}"


def test_the_boundary_between_dead_and_inverted_is_the_threshold():
    """An interval reaching just past -DEAD_THRESHOLD stops being negligible."""
    assert FLAG_DEAD in flags_for(-DEAD_THRESHOLD + 0.01, -0.05)
    assert FLAG_INVERTED in flags_for(-DEAD_THRESHOLD - 0.01, -0.05)


def test_an_inverted_item_is_not_reported_as_dead():
    """The whole point. This interval was `dead` under schema 1."""
    flags = flags_for(-1.17, -0.22)
    assert FLAG_INVERTED in flags
    assert FLAG_DEAD not in flags


# -- the artifact's invariants -----------------------------------------------


def test_an_artifact_claiming_two_of_the_three_is_rejected():
    fit, _ = synthetic_fit(n_models=6, n_items=12, seed=0)
    fit.flags[0] = [FLAG_DEAD, FLAG_INVERTED]
    with pytest.raises(ArtifactError, match="mutually exclusive"):
        fit.validate()


def test_the_error_names_both_claims_so_it_can_be_acted_on():
    fit, _ = synthetic_fit(n_models=6, n_items=12, seed=0)
    fit.flags[0] = [FLAG_INVERTED, FLAG_INSUFFICIENT_DATA]
    with pytest.raises(ArtifactError) as exc:
        fit.validate()
    assert "discriminates backwards" in str(exc.value)
    assert "cannot tell" in str(exc.value)


def test_inverted_items_are_not_eligible_for_selection():
    """An anchor set is scored by plain accuracy over its items, so an item a
    stronger model reliably gets *wrong* subtracts from the signal.

    `dead` items stay eligible — their `a` is confidently near zero, and
    selection maximises `a^2`, so they are never picked anyway. That argument
    does not extend to an inverted item, whose `|a|` is large.
    """
    fit, _ = synthetic_fit(n_models=8, n_items=20, seed=0)
    fit.flags[3] = [FLAG_INVERTED]
    fit.flags[4] = [FLAG_DEAD]
    usable = fit.usable_items()
    assert 3 not in usable, "an inverted item is eligible for an anchor set"
    assert 4 in usable, "a dead item should stay eligible"


# -- end to end --------------------------------------------------------------


@pytest.mark.slow
def test_the_fitter_recovers_and_flags_a_planted_inverted_item():
    """synth draws `a` from a lognormal and so cannot produce one by accident;
    `inverted_fraction` asks for them explicitly."""
    from irtcheck.fit.fitter import fit_2pl
    from irtcheck.synth import synthetic_matrix

    matrix, truth = synthetic_matrix(
        n_models=120, n_items=200, inverted_fraction=0.05, seed=0
    )
    fit = fit_2pl(matrix, seed=0)
    order = [list(truth.item_ids).index(item) for item in fit.item_ids]
    true_a = np.asarray(truth.a, dtype=float)[order]
    planted = true_a < -0.5
    flagged = np.array([FLAG_INVERTED in f for f in fit.flags])

    assert planted.sum() == 10, planted.sum()
    assert flagged[planted].mean() >= 0.7, (
        f"only {flagged[planted].mean():.0%} of planted inverted items were "
        f"flagged inverted; recovered a = {np.asarray(fit.a.mean)[planted]}"
    )
    # And nothing with a genuinely positive discrimination is called inverted.
    assert flagged[true_a > 0.5].sum() == 0
