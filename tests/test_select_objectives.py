"""The experiment that chose the default selection objective.

`max-information` is the spec's criterion: pick the items whose summed test
information over the observed ability distribution is highest. It is linear in
the item set, so greedy forward selection on it is exactly a sort, and it puts
the whole anchor set wherever the ability density is highest. `min-variance`
minimises the mean posterior variance of the respondents' abilities instead; it
is concave, so it spreads to cover the range.

The argument for switching the default to `min-variance` is a good one. The
measurement does not support it, so the default stayed with the spec. This file
is that measurement, kept runnable so nobody has to take the number on trust:

    .venv/bin/python tests/test_select_objectives.py

The assertions below are the load-bearing part — they fail if the conclusion
stops holding, e.g. after the real fitter replaces the ground-truth one, which
is the moment to run this again. They are stated as inequalities with slack
rather than as fixed numbers, because the point is which objective wins, not
what tau was on a particular seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_validate import truth_fit_fn  # noqa: E402

from irtcheck.select import (  # noqa: E402
    DEFAULT_OBJECTIVE,
    OBJECTIVE_MAX_INFORMATION,
    OBJECTIVE_MIN_VARIANCE,
    select_anchor,
)
from irtcheck.synth import synthetic_fit, synthetic_matrix  # noqa: E402
from irtcheck.validate import leave_one_model_out  # noqa: E402

# The five seeds the PR's curve was first measured on, plus twenty fresh ones.
# Kendall tau over eight models moves in steps of 1/28 = 0.036, so five seeds
# cannot separate objectives that differ by less than a swapped pair; the wide
# set is what gives the comparison any power at all.
HEADLINE_SEEDS = (3, 100, 101, 102, 103)
WIDE_SEEDS = tuple(range(200, 220))
SIZES = (10, 25, 50, 100, 200, 400)
SMALL_N = (25, 50)  # where the objectives should diverge, and where users live


def _selector(objective: str):
    def select_fn(fit, n: int) -> list[str]:
        return select_anchor(fit, n, objective=objective).item_ids

    return select_fn


def rank_recovery(seed: int, sizes=SIZES, **universe) -> dict[str, dict[int, float]]:
    """Kendall tau per objective per size, on one synthetic universe."""
    matrix, truth = synthetic_matrix(seed=seed, **universe)
    out: dict[str, dict[int, float]] = {}
    for objective in (OBJECTIVE_MAX_INFORMATION, OBJECTIVE_MIN_VARIANCE):
        report = leave_one_model_out(
            matrix,
            fit_fn=truth_fit_fn(truth),
            sizes=sizes,
            select_fn=_selector(objective),
        )
        out[objective] = {r.size: r.kendall for r in report.results}
    return out


def paired(seeds, sizes=SIZES, **universe):
    """(mean tau per objective, win/loss counts) at each size, paired by seed."""
    per = {seed: rank_recovery(seed, sizes, **universe) for seed in seeds}
    rows = {}
    for n in sizes:
        mi = np.array([per[s][OBJECTIVE_MAX_INFORMATION][n] for s in seeds], dtype=float)
        mv = np.array([per[s][OBJECTIVE_MIN_VARIANCE][n] for s in seeds], dtype=float)
        rows[n] = {
            "max-information": float(mi.mean()),
            "min-variance": float(mv.mean()),
            "min_variance_wins": int(np.sum(mv > mi)),
            "min_variance_losses": int(np.sum(mv < mi)),
            "ties": int(np.sum(mv == mi)),
        }
    return rows


def precision(seed: int, sizes=SIZES, **universe):
    """Mean and worst-case ability standard error. No holdout, no noise."""
    fit, _ = synthetic_fit(seed=seed, **universe)
    rows = {}
    for n in sizes:
        rows[n] = {}
        for objective in (OBJECTIVE_MAX_INFORMATION, OBJECTIVE_MIN_VARIANCE):
            anchor = select_anchor(fit, n, objective=objective)
            rows[n][objective] = (anchor.mean_theta_se, max(anchor.theta_se))
    return rows


# -- the conclusions ---------------------------------------------------------


def test_the_default_is_the_specs_criterion():
    """docs/spec.md: "Selection maximizes test information over the observed
    ability distribution." Deviating from that needs a measured reason, and the
    measurement below did not produce one."""
    assert DEFAULT_OBJECTIVE == OBJECTIVE_MAX_INFORMATION


def test_min_variance_does_measure_more_precisely():
    """It is not that the concave objective fails at what it optimises."""
    rows = precision(3, sizes=(10, 25, 50), n_models=8, n_items=400)
    for n in (10, 25, 50):
        mean_mi, worst_mi = rows[n][OBJECTIVE_MAX_INFORMATION]
        mean_mv, worst_mv = rows[n][OBJECTIVE_MIN_VARIANCE]
        assert mean_mv < mean_mi
        assert worst_mv < worst_mi
    # The worst-served model is where the gap is wide enough to matter.
    assert rows[10][OBJECTIVE_MIN_VARIANCE][1] < 0.6 * rows[10][OBJECTIVE_MAX_INFORMATION][1]


@pytest.mark.parametrize("n", SMALL_N)
def test_measuring_more_precisely_does_not_recover_ranks_better(n):
    """The finding: precision at the edges of the range is not what ranks models.

    Spreading information helps the models furthest from the centre, and those
    are the ones whose rank was never in doubt. The pairs that actually swap are
    bunched in the middle, which is where the linear objective already was.
    """
    rows = paired(WIDE_SEEDS, sizes=(n,), n_models=8, n_items=400)[n]
    assert rows["min_variance_wins"] <= rows["min_variance_losses"]
    assert rows["min-variance"] <= rows["max-information"] + 0.01


def test_neither_objective_is_ahead_by_a_swapped_pair_anywhere():
    """1/28 is the smallest tau difference eight models can express."""
    quantum = 1.0 / 28
    for seeds in (HEADLINE_SEEDS, WIDE_SEEDS):
        rows = paired(seeds, sizes=SMALL_N, n_models=8, n_items=400)
        for n in SMALL_N:
            gap = abs(rows[n]["min-variance"] - rows[n]["max-information"])
            assert gap < quantum, f"{n=} {seeds[0]=} {gap=}"


def test_the_objectives_converge_once_n_covers_the_usable_suite():
    """Both end up selecting the same set, so the sweep has to agree at the top."""
    rows = paired(HEADLINE_SEEDS, sizes=(400,), n_models=8, n_items=400)[400]
    assert rows["min-variance"] == pytest.approx(rows["max-information"])


# -- the table, for a human --------------------------------------------------


def _print(label: str, rows: dict) -> None:
    print(f"\n=== {label}")
    print(f"{'n':>5} | {'max-information':>15} | {'min-variance':>13} | min-variance record")
    for n, row in rows.items():
        print(
            f"{n:5d} | {row['max-information']:+15.3f} | {row['min-variance']:+13.3f} | "
            f"win {row['min_variance_wins']} / lose {row['min_variance_losses']} / "
            f"tie {row['ties']}"
        )


def main() -> None:
    print("Mean Kendall tau against the full-suite ranking, paired by seed.")
    _print(
        f"8 models, 400 items, {len(HEADLINE_SEEDS)} seeds (the PR's curve)",
        paired(HEADLINE_SEEDS, n_models=8, n_items=400),
    )
    _print(
        f"8 models x 2 variants, 400 items, {len(HEADLINE_SEEDS)} seeds",
        paired(HEADLINE_SEEDS, n_models=8, n_items=400, variants_per_model=2),
    )
    _print(
        f"8 models, 400 items, {len(WIDE_SEEDS)} fresh seeds (power)",
        paired(WIDE_SEEDS, sizes=(10, 25, 50, 100), n_models=8, n_items=400),
    )
    _print(
        f"12 models x 3 variants, 1000 items, {len(HEADLINE_SEEDS)} seeds",
        paired(HEADLINE_SEEDS, n_models=12, n_items=1000, variants_per_model=3),
    )

    print("\n=== ability standard error, mean / worst model (no holdout, no noise)")
    rows = precision(3, n_models=8, n_items=400)
    print(f"{'n':>5} | {'max-information':>17} | {'min-variance':>17}")
    for n, row in rows.items():
        mi = row[OBJECTIVE_MAX_INFORMATION]
        mv = row[OBJECTIVE_MIN_VARIANCE]
        print(f"{n:5d} | {mi[0]:7.4f} / {mi[1]:7.4f} | {mv[0]:7.4f} / {mv[1]:7.4f}")

    print(
        "\nmin-variance measures more precisely and ranks no better, so the default "
        f"stays with the spec: {DEFAULT_OBJECTIVE}."
    )


if __name__ == "__main__":
    main()
