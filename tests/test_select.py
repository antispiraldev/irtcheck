"""Anchor selection: the information maths, and what the greedy loop does with it."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import typer

from irtcheck.artifact import FLAG_CEILING, FLAG_FLOOR, FLAG_INSUFFICIENT_DATA
from irtcheck.commands import select as command
from irtcheck.select import (
    DEFAULT_OBJECTIVE,
    OBJECTIVE_MAX_INFORMATION,
    OBJECTIVE_MIN_VARIANCE,
    SelectError,
    ability_distribution,
    item_information,
    select_anchor,
    select_item_ids,
)
from irtcheck.synth import synthetic_fit

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI.sub("", text)


# -- the information function ------------------------------------------------


def test_information_peaks_at_the_difficulty():
    """I_i(theta) = a^2 P (1-P) is maximal at theta = b, where P = 0.5."""
    theta = np.linspace(-3, 3, 601)
    info = item_information([1.4], [0.7], theta)[0]
    assert theta[int(np.argmax(info))] == pytest.approx(0.7, abs=0.01)
    assert info.max() == pytest.approx(1.4**2 * 0.25)


def test_information_grows_with_the_square_of_discrimination():
    peak = item_information([0.5, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0])[:, 0]
    assert peak[1] / peak[0] == pytest.approx(4.0)
    assert peak[2] / peak[1] == pytest.approx(4.0)


def test_an_item_far_from_every_model_carries_almost_nothing():
    """The finding the tool exists for: precise measurement where nobody sits."""
    at_home = item_information([2.0], [0.0], [0.0])[0, 0]
    far_away = item_information([2.0], [5.0], [0.0])[0, 0]
    assert far_away < at_home / 1000


# -- the ability distribution ------------------------------------------------


def test_ability_points_are_the_respondents_not_a_grid():
    fit, _ = synthetic_fit(n_models=6, n_items=40, seed=1)
    theta, weights = ability_distribution(fit)
    assert theta.tolist() == fit.theta.mean
    assert weights.sum() == pytest.approx(1.0)


def test_pseudo_respondents_share_one_models_weight():
    """Ten temperature samples of one model must not outvote a model run once."""
    fit, _ = synthetic_fit(n_models=4, n_items=40, variants_per_model=3, seed=2)
    _, weights = ability_distribution(fit)
    assert fit.n_respondents == 12
    per_model: dict[str, float] = {}
    for source, w in zip(fit.derives_from, weights, strict=True):
        per_model[source] = per_model.get(source, 0.0) + float(w)
    assert set(per_model) == set(fit.derives_from)
    for total in per_model.values():
        assert total == pytest.approx(0.25)


# -- selection ---------------------------------------------------------------


def test_selection_never_returns_an_item_we_said_we_could_not_read():
    fit, _ = synthetic_fit(n_models=6, n_items=300, seed=4)
    anchor = select_anchor(fit, 40)
    excluded = {FLAG_INSUFFICIENT_DATA, FLAG_CEILING, FLAG_FLOOR}
    for index in anchor.item_indices:
        assert not excluded.intersection(fit.flags[index])
    assert set(anchor.item_indices) <= set(fit.usable_items())


def test_selection_is_deterministic():
    """A regression suite needs the same items every run or scores drift."""
    fit, _ = synthetic_fit(n_models=6, n_items=200, seed=5)
    assert select_item_ids(fit, 30) == select_item_ids(fit, 30)


def test_greedy_selection_is_nested():
    """The 30-item set is the 60-item set's prefix, so n is a dial and not a fork."""
    fit, _ = synthetic_fit(n_models=6, n_items=200, seed=6)
    assert select_item_ids(fit, 60)[:30] == select_item_ids(fit, 30)


def test_selection_beats_picking_usable_items_at_random():
    """The whole claim: choosing on information measures these models better."""
    fit, _ = synthetic_fit(n_models=8, n_items=400, seed=7)
    anchor = select_anchor(fit, 25)
    theta, weights = ability_distribution(fit)
    a = np.asarray(fit.a.mean)
    b = np.asarray(fit.b.mean)
    usable = fit.usable_items()

    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(20):
        draw = rng.choice(usable, size=25, replace=False)
        info = item_information(a[draw], b[draw], theta).sum(axis=0)
        se = float(np.dot(weights, np.sqrt(1.0 / (1.0 + info))))
        worse += se > anchor.mean_theta_se
    assert worse == 20


def test_selection_prefers_items_near_where_the_models_sit():
    fit, _ = synthetic_fit(n_models=8, n_items=400, off_range_fraction=0.2, seed=8)
    anchor = select_anchor(fit, 40)
    theta = np.asarray(fit.theta.mean)
    b = np.asarray(fit.b.mean)
    chosen = np.abs(b[anchor.item_indices] - theta.mean()).mean()
    pool = np.abs(b[fit.usable_items()] - theta.mean()).mean()
    assert chosen < pool


def test_dead_items_are_eligible_and_never_worth_picking():
    """Dead is a finding about the suite, not a filter — it falls out of the maths."""
    fit, _ = synthetic_fit(n_models=15, n_items=300, precision=6.0, seed=9)
    dead = set(fit.flagged("dead"))
    assert dead, "the fixture is meant to contain confidently dead items"
    assert dead <= set(fit.usable_items())
    assert not dead.intersection(select_anchor(fit, 100).item_indices)


def test_a_short_anchor_set_is_reported_rather_than_padded():
    fit, _ = synthetic_fit(n_models=6, n_items=60, seed=10)
    usable = len(fit.usable_items())
    anchor = select_anchor(fit, usable + 25)
    assert anchor.n_selected == usable
    assert anchor.shortfall == 25
    assert len(set(anchor.item_ids)) == usable  # no duplicates padding it out


def test_gains_decay_so_a_user_can_see_where_n_stops_buying_anything():
    fit, _ = synthetic_fit(n_models=8, n_items=300, seed=11)
    anchor = select_anchor(fit, 60)
    assert anchor.gains[0] > anchor.gains[-1]
    assert all(g >= 0 for g in anchor.gains)
    assert anchor.mean_theta_se < 1.0  # 1.0 is the prior, i.e. no items at all


def test_bad_sizes_and_objectives_raise():
    fit, _ = synthetic_fit(n_models=4, n_items=30, seed=12)
    with pytest.raises(SelectError, match="must be positive"):
        select_anchor(fit, 0)
    with pytest.raises(SelectError, match="unknown objective"):
        select_anchor(fit, 5, objective="vibes")
    with pytest.raises(SelectError, match="out of range"):
        select_anchor(fit, 5, candidates=[0, 9999])


def test_no_eligible_items_points_at_respondent_count():
    """The refusal path: with too few respondents nothing can be read at all."""
    fit, _ = synthetic_fit(n_models=4, n_items=30, seed=13)
    fit.flags = [[FLAG_INSUFFICIENT_DATA] for _ in fit.item_ids]
    with pytest.raises(SelectError, match="respondent-key"):
        select_anchor(fit, 5)


# -- items nobody answered ---------------------------------------------------


def _make_irresistible(fit, index: int, *, n_resp: int) -> None:
    """Give one item a posterior good enough to be selected, and `n_resp` responses.

    Discrimination well above the suite, difficulty in the middle of the ability
    range, and a tight interval far from zero — so nothing about the posterior
    excludes it and the information ranking wants it. Whether it belongs in an
    anchor set then turns entirely on whether anybody answered it.
    """
    theta = float(np.mean(fit.theta.mean))
    fit.a.mean[index], fit.a.sd[index] = 2.0, 0.01
    fit.a.hdi_low[index], fit.a.hdi_high[index] = 1.9, 2.1
    fit.b.mean[index], fit.b.sd[index] = theta, 0.01
    fit.b.hdi_low[index], fit.b.hdi_high[index] = theta - 0.05, theta + 0.05
    fit.n_resp[index] = n_resp
    fit.p_correct[index] = 0.5  # what the flag fix will leave behind: no claim
    fit.flags[index] = []
    fit.validate()


def test_an_item_nobody_answered_is_never_selected():
    """Leave-one-model-out manufactures this: a ragged matrix where only the
    held-out model answered an item leaves it with zero responses in the fit
    that then picks the anchor set.

    The exclusion must not be inherited from flag semantics. Today such an item
    also lands as `floor` because p_correct comes through as 0.0, but that is a
    bug being fixed — "everyone got it wrong" is not a claim about an item
    nobody answered — and after the fix the flag path would not catch it. So
    this is an A/B on the response count alone: same posterior, same flags, same
    everything, one response versus none.
    """
    fit, _ = synthetic_fit(n_models=8, n_items=120, seed=19)
    ghost = fit.usable_items()[0]

    _make_irresistible(fit, ghost, n_resp=1)
    assert ghost in select_anchor(fit, 10).item_indices, "the fixture must be selectable"

    _make_irresistible(fit, ghost, n_resp=0)
    # Nothing in the flag path is doing the work here.
    assert ghost in fit.usable_items()
    assert not set(fit.flags[ghost]).intersection(
        {FLAG_INSUFFICIENT_DATA, FLAG_CEILING, FLAG_FLOOR}
    )
    # It still tops the information ranking; it is simply not about anyone.
    theta, weights = ability_distribution(fit)
    score = item_information(fit.a.mean, fit.b.mean, theta) @ weights
    assert score[ghost] > np.median(score)

    anchor = select_anchor(fit, 10)
    assert ghost not in anchor.item_indices
    assert anchor.n_usable == len(fit.usable_items()) - 1


def test_a_zero_response_item_is_excluded_even_when_named_explicitly():
    """Not a filter the caller can opt out of by passing candidates directly."""
    fit, _ = synthetic_fit(n_models=6, n_items=60, seed=20)
    ghost = fit.usable_items()[0]
    _make_irresistible(fit, ghost, n_resp=0)
    others = [i for i in fit.usable_items() if i != ghost][:5]

    anchor = select_anchor(fit, 6, candidates=[ghost, *others])
    assert ghost not in anchor.item_indices
    assert anchor.n_selected == 5


def test_a_fit_where_nothing_was_answered_says_so():
    fit, _ = synthetic_fit(n_models=4, n_items=30, seed=21)
    fit.n_resp = [0] * fit.n_items
    with pytest.raises(SelectError, match="not have a single response|single response"):
        select_anchor(fit, 5)


# -- the two objectives ------------------------------------------------------


def test_max_information_is_a_sort_and_min_variance_is_not():
    """Total information is linear in the set, so greedy on it is exactly top-k."""
    fit, _ = synthetic_fit(n_models=8, n_items=300, seed=14)
    theta, weights = ability_distribution(fit)
    a = np.asarray(fit.a.mean)
    b = np.asarray(fit.b.mean)
    usable = fit.usable_items()
    score = item_information(a[usable], b[usable], theta) @ weights
    top = [usable[i] for i in np.argsort(-score, kind="stable")[:20]]

    linear = select_anchor(fit, 20, objective=OBJECTIVE_MAX_INFORMATION)
    assert linear.item_indices == top

    concave = select_anchor(fit, 20, objective=OBJECTIVE_MIN_VARIANCE)
    assert concave.item_indices != top


def test_min_variance_covers_the_worst_served_model_better():
    """Why it is the default: nothing in a linear objective ever says 'enough here'."""
    fit, _ = synthetic_fit(n_models=8, n_items=300, seed=15)
    spread = select_anchor(fit, 20, objective=OBJECTIVE_MIN_VARIANCE)
    stacked = select_anchor(fit, 20, objective=OBJECTIVE_MAX_INFORMATION)
    assert max(spread.theta_se) < max(stacked.theta_se)


# -- the command -------------------------------------------------------------


def _artifact(tmp_path: Path, **kwargs) -> Path:
    fit, _ = synthetic_fit(n_models=6, n_items=120, seed=16, **kwargs)
    return fit.save(tmp_path / "suite.irt")


def test_command_writes_json_and_summarises_to_stderr(tmp_path: Path, capsys):
    path = _artifact(tmp_path)
    out = tmp_path / "anchor.json"
    command.run(artifact=path, count=20, output=out, adaptive=False)

    payload = json.loads(out.read_text())
    assert len(payload["item_ids"]) == 20
    assert payload["requested"] == 20
    assert payload["objective"] == DEFAULT_OBJECTIVE
    assert payload["source"] == str(path)

    captured = capsys.readouterr()
    assert captured.out == ""  # everything human went to stderr
    assert "20" in plain(captured.err)
    assert "6 models" in plain(captured.err)


def test_command_prints_ids_to_stdout_when_there_is_no_output_file(tmp_path: Path, capsys):
    path = _artifact(tmp_path)
    command.run(artifact=path, count=8, output=None, adaptive=False)
    captured = capsys.readouterr()
    ids = captured.out.split()
    assert len(ids) == 8
    assert all(i.startswith("item_") for i in ids)


def test_command_refuses_adaptive_rather_than_quietly_doing_something_else(
    tmp_path: Path, capsys
):
    path = _artifact(tmp_path)
    with pytest.raises(typer.Exit) as exc:
        command.run(artifact=path, count=5, output=None, adaptive=True)
    assert exc.value.exit_code == 2
    message = plain(capsys.readouterr().err)
    assert "--adaptive is not implemented" in message
    assert "comparable" in message


def test_command_reports_a_short_set(tmp_path: Path, capsys):
    path = _artifact(tmp_path)
    command.run(artifact=path, count=10_000, output=None, adaptive=False)
    assert "Asked for" in plain(capsys.readouterr().err)


def test_command_names_the_pseudo_respondent_key_in_its_header(tmp_path: Path, capsys):
    fit, _ = synthetic_fit(n_models=5, n_items=60, variants_per_model=3, seed=17)
    path = fit.save(tmp_path / "pseudo.irt")
    command.run(artifact=path, count=5, output=None, adaptive=False)
    err = plain(capsys.readouterr().err)
    assert "5 models" in err  # not 15 respondents
    assert "model_id+prompt_variant" in err


def test_command_exits_cleanly_on_a_bad_artifact(tmp_path: Path, capsys):
    broken = tmp_path / "broken.irt"
    broken.write_bytes(b"not a gzip file")
    with pytest.raises(typer.Exit) as exc:
        command.run(artifact=broken, count=5, output=None, adaptive=False)
    assert exc.value.exit_code == 2
    assert "broken.irt" in plain(capsys.readouterr().err)


def test_command_surfaces_a_selection_refusal(tmp_path: Path, capsys):
    fit, _ = synthetic_fit(n_models=4, n_items=30, seed=18)
    fit.flags = [[FLAG_INSUFFICIENT_DATA] for _ in fit.item_ids]
    path = fit.save(tmp_path / "thin.irt")
    with pytest.raises(typer.Exit) as exc:
        command.run(artifact=path, count=5, output=None, adaptive=False)
    assert exc.value.exit_code == 2
    assert "respondent-key" in plain(capsys.readouterr().err)


def test_the_cli_reaches_this_command(tmp_path: Path):
    """The wiring in the frozen cli.py, exercised end to end."""
    from typer.testing import CliRunner

    from irtcheck.cli import app

    path = _artifact(tmp_path)
    out = tmp_path / "anchor.json"
    result = CliRunner().invoke(app, ["select", str(path), "-n", "12", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(out.read_text())["item_ids"]) == 12
