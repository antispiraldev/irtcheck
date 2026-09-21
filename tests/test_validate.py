"""Leave-one-model-out, built against an injected fitter.

The fake fitter here returns the *generating* parameters of a synthetic
universe. That is what makes this suite possible before a real fitter exists,
and it is also the right thing to test against: with ground-truth item
parameters, anything the correlation curve does is a property of the selection
and holdout logic rather than of SVI convergence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from irtcheck import __version__
from irtcheck.artifact import (
    EmbeddedResponses,
    IrtFit,
    Posterior,
    compute_flags,
    utc_now,
)
from irtcheck.matrix import ResponseMatrix
from irtcheck.select import select_item_ids
from irtcheck.synth import SyntheticTruth, synthetic_fit, synthetic_matrix
from irtcheck.validate import (
    DEFAULT_SIZES,
    AccuracyTable,
    ValidateError,
    full_suite_accuracy,
    leave_one_model_out,
    parse_sizes,
    placement,
)

Z95 = 1.959963985

# rich decides for itself whether the capture buffer is a terminal, so assert
# against text rather than against whatever it decided this run.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI.sub("", text)


# -- the fake fitter ---------------------------------------------------------


def _posterior(values, sd: float) -> Posterior:
    mean = np.asarray(values, dtype=float)
    spread = np.full(mean.shape, sd, dtype=float)
    return Posterior(
        mean=[float(x) for x in mean],
        sd=[float(x) for x in spread],
        hdi_low=[float(x) for x in mean - Z95 * spread],
        hdi_high=[float(x) for x in mean + Z95 * spread],
    )


def truth_fit_fn(truth: SyntheticTruth, *, sd: float = 0.005, log: list | None = None):
    """A `fit_fn` that returns the parameters the matrix was generated from.

    Intervals are narrow on purpose: a fake fit that flagged half the suite
    insufficient-data would be testing `usable_items()`, not the holdout loop.
    Genuinely flat items still come back flagged `dead` — correctly, since the
    interval is both narrow and under the threshold — and are left eligible,
    which is what lets the tests assert that selection ignores them on
    information alone.
    """
    theta_of = dict(zip(truth.respondent_ids, truth.theta, strict=True))
    a_of = dict(zip(truth.item_ids, truth.a, strict=True))
    b_of = dict(zip(truth.item_ids, truth.b, strict=True))

    def fit_fn(matrix: ResponseMatrix) -> IrtFit:
        if log is not None:
            log.append(matrix)
        p_correct = matrix.item_p_correct()
        theta = _posterior([theta_of[r] for r in matrix.respondent_ids], sd)
        a = _posterior([a_of[i] for i in matrix.item_ids], sd)
        b = _posterior([b_of[i] for i in matrix.item_ids], sd)
        fit = IrtFit(
            irtcheck_version=__version__,
            created=utc_now(),
            model={"kind": "2pl", "identification": "a > 0, theta ~ N(0, 1)", "fake": True},
            respondent_key=list(matrix.respondent_key),
            respondent_ids=list(matrix.respondent_ids),
            derives_from=list(matrix.derives_from),
            theta=theta,
            item_ids=list(matrix.item_ids),
            a=a,
            b=b,
            n_resp=[int(x) for x in matrix.item_response_counts()],
            p_correct=[float(x) for x in np.nan_to_num(p_correct, nan=0.0)],
            flags=compute_flags(a, b, [float(p) for p in p_correct], theta.mean),
            responses=EmbeddedResponses(
                rows=[int(x) for x in matrix.rows],
                cols=[int(x) for x in matrix.cols],
                obs=[int(x) for x in matrix.obs],
            ),
        )
        fit.validate()
        return fit

    return fit_fn


# -- ground truth ------------------------------------------------------------


def test_full_suite_accuracy_is_respondent_accuracy_per_model():
    matrix, _ = synthetic_matrix(n_models=4, n_items=60, seed=5)
    truth = full_suite_accuracy(matrix)
    expected = dict(zip(matrix.respondent_ids, matrix.respondent_accuracy(), strict=True))
    assert set(truth) == set(matrix.derives_from)
    for model_id, value in truth.items():
        assert value == pytest.approx(expected[model_id])


def test_full_suite_accuracy_averages_over_pseudo_respondents():
    matrix, _ = synthetic_matrix(n_models=3, n_items=40, variants_per_model=3, seed=6)
    per_respondent = matrix.respondent_accuracy()
    truth = full_suite_accuracy(matrix)
    assert len(truth) == 3
    assert matrix.n_respondents == 9
    rows = [i for i, src in enumerate(matrix.derives_from) if src == "model-01"]
    assert truth["model-01"] == pytest.approx(float(np.mean(per_respondent[rows])))


# -- the trap: holding out a model, not a respondent -------------------------


def test_holdout_removes_every_pseudo_respondent_of_the_model():
    """The failure this whole module is arranged around.

    With --respondent-key model_id,prompt_variant one model is several
    respondents. If the holdout drops one row, the model's other variants stay
    in the fit that chooses the anchor set — the set is then picked with
    knowledge of the model it is about to be scored on, and the headline
    correlation is inflated with nothing to catch it.
    """
    matrix, truth = synthetic_matrix(n_models=4, n_items=40, variants_per_model=3, seed=7)
    assert matrix.n_respondents == 12
    assert matrix.respondent_key == ("model_id", "prompt_variant")

    seen: list[ResponseMatrix] = []
    leave_one_model_out(matrix, fit_fn=truth_fit_fn(truth, log=seen), sizes=(10,))

    assert len(seen) == 4  # one fit per model, not per respondent
    held_out = sorted(set(matrix.derives_from))
    for model_id, fitted in zip(held_out, seen, strict=True):
        assert model_id not in fitted.derives_from
        assert not [r for r in fitted.respondent_ids if r.startswith(f"{model_id}|")]
        assert fitted.n_respondents == 9  # all three variants gone together
        assert fitted.n_real_models == 3
        assert sorted(set(fitted.derives_from)) == [m for m in held_out if m != model_id]


def test_the_fit_that_chooses_the_anchor_set_never_saw_the_held_out_model():
    matrix, truth = synthetic_matrix(n_models=5, n_items=40, variants_per_model=2, seed=8)
    seen: list[str] = []

    def select_fn(fit: IrtFit, n: int) -> list[str]:
        seen.append(",".join(sorted(set(fit.derives_from))))
        return select_item_ids(fit, n)

    report = leave_one_model_out(
        matrix, fit_fn=truth_fit_fn(truth), sizes=(5, 10), select_fn=select_fn
    )
    assert len(seen) == 5 * 2  # a selection per (holdout, size)
    for sources, model_id in zip(seen[::2], report.model_ids, strict=True):
        assert model_id not in sources.split(",")


def test_one_fit_per_holdout_not_one_per_size():
    """Fits are the entire cost of this command; sizes must not multiply them."""
    matrix, truth = synthetic_matrix(n_models=4, n_items=40, seed=9)
    seen: list[ResponseMatrix] = []
    leave_one_model_out(matrix, fit_fn=truth_fit_fn(truth, log=seen), sizes=(5, 10, 20, 30))
    assert len(seen) == 4


# -- scoring every model on one set ------------------------------------------


def test_accuracy_table_matches_full_suite_accuracy_on_every_item():
    matrix, _ = synthetic_matrix(n_models=5, n_items=60, variants_per_model=3, missing=0.2, seed=10)
    table = AccuracyTable(matrix)
    every = np.arange(matrix.n_items)
    truth = full_suite_accuracy(matrix)
    assert table.accuracy(every) == pytest.approx([truth[m] for m in table.model_ids])


def test_accuracy_table_uses_only_the_set():
    matrix, _ = synthetic_matrix(n_models=4, n_items=50, seed=10)
    table = AccuracyTable(matrix)
    cols = table.columns(matrix.item_ids[:12])
    row = matrix.respondent_ids.index("model-02")
    mask = (matrix.rows == row) & np.isin(matrix.cols, np.arange(12))
    assert table.accuracy(cols)[table.model_ids.index("model-02")] == pytest.approx(
        float(matrix.obs[mask].mean())
    )


def test_a_model_that_answered_none_of_the_set_scores_nan_and_drops_out_of_placement():
    scores = np.array([0.9, np.nan, 0.5, 0.7])
    truth = np.array([0.8, 0.95, 0.4, 0.6])
    assert all(np.isnan(placement(scores, truth, 1)))
    # model 2 is last of the three that have a score, on both rankings
    assert placement(scores, truth, 2) == (3.0, 3.0)


def test_placement_is_one_for_the_best_and_shares_ties():
    truth = np.array([0.9, 0.8, 0.7])
    assert placement(np.array([0.2, 0.6, 0.6]), truth, 0) == (3.0, 1.0)
    assert placement(np.array([0.2, 0.6, 0.6]), truth, 1) == (1.5, 2.0)


def test_every_model_is_scored_on_the_held_out_models_set():
    """The correction: a set that is hard for everyone moves nobody's place."""
    matrix, truth = synthetic_matrix(n_models=6, n_items=120, seed=12)
    hardest = [i for _, i in sorted(zip(truth.b, matrix.item_ids, strict=True))][-40:]

    report = leave_one_model_out(
        matrix, fit_fn=truth_fit_fn(truth), sizes=(40,), select_fn=lambda fit, n: hardest
    )
    [result] = report.results
    for score in result.models:
        assert score.anchor_accuracy < score.truth  # the set skews hard for the held-out model
    # ...and yet places are compared among models all scored on that same set.
    assert result.mean_error <= 1.0


def test_select_fn_must_return_items_from_the_fit():
    matrix, truth = synthetic_matrix(n_models=4, n_items=30, seed=11)
    with pytest.raises(ValidateError, match="does not know"):
        leave_one_model_out(
            matrix,
            fit_fn=truth_fit_fn(truth),
            sizes=(2,),
            select_fn=lambda fit, n: ["item_00000", "not_an_item"],
            random_draws=0,
        )


# -- the random baseline -----------------------------------------------------


def test_the_random_baseline_is_seeded():
    matrix, truth = synthetic_matrix(n_models=6, n_items=120, seed=18)
    run = lambda seed: leave_one_model_out(  # noqa: E731
        matrix, fit_fn=truth_fit_fn(truth), sizes=(10,), random_draws=30, seed=seed
    ).results[0]
    assert run(0).random_errors == run(0).random_errors
    assert run(0).random_errors != run(1).random_errors


def test_random_sets_match_the_size_the_anchor_set_came_out():
    """A short anchor set is compared with an equally short random one."""
    matrix, truth = synthetic_matrix(n_models=5, n_items=80, seed=19)
    sizes_drawn: list[int] = []
    original = AccuracyTable.accuracy

    def spy(self, cols):
        sizes_drawn.append(len(cols))
        return original(self, cols)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AccuracyTable, "accuracy", spy)
        leave_one_model_out(
            matrix,
            fit_fn=truth_fit_fn(truth),
            sizes=(30,),
            select_fn=lambda fit, n: fit.item_ids[:7],
            random_draws=4,
        )
    assert set(sizes_drawn) == {7}


def test_random_sets_never_include_items_the_held_out_fit_has_no_responses_for():
    matrix, truth = synthetic_matrix(n_models=4, n_items=40, seed=20)
    only_model_00 = matrix.item_ids[0]
    col = matrix.item_ids.index(only_model_00)
    owner = np.array([matrix.derives_from[r] for r in matrix.rows])
    keep = (matrix.cols != col) | (owner == "model-00")
    matrix.rows, matrix.cols, matrix.obs = matrix.rows[keep], matrix.cols[keep], matrix.obs[keep]

    drawn: list[np.ndarray] = []
    original = AccuracyTable.accuracy

    def spy(self, cols):
        drawn.append(np.asarray(cols))
        return original(self, cols)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AccuracyTable, "accuracy", spy)
        leave_one_model_out(
            matrix,
            fit_fn=truth_fit_fn(truth),
            sizes=(39,),
            select_fn=lambda fit, n: [i for i in fit.item_ids if i != only_model_00][:n],
            holdouts=["model-00"],
            random_draws=10,
        )
    assert all(col not in cols for cols in drawn[1:])


def test_beats_random_counts_ties_as_half():
    from irtcheck.validate import ModelScore, SizeResult

    model = ModelScore("m", 0.5, 1.0, 0.5, 1.0, 10, 1, 0.0)
    result = SizeResult(size=10, models=[model], random_errors=[0.0, 0.0, 1.0, 1.0])
    assert result.mean_error == 0.0
    assert result.beats_random == pytest.approx(0.75)


def test_holdouts_restrict_fits_but_every_model_is_still_scored():
    matrix, truth = synthetic_matrix(n_models=6, n_items=60, seed=21)
    seen: list[ResponseMatrix] = []
    report = leave_one_model_out(
        matrix,
        fit_fn=truth_fit_fn(truth, log=seen),
        sizes=(10,),
        holdouts=["model-01", "model-04"],
        random_draws=5,
    )
    assert len(seen) == 2
    assert report.n_models == 6
    assert [m.model_id for m in report.results[0].models] == ["model-01", "model-04"]
    with pytest.raises(ValidateError, match="does not have"):
        leave_one_model_out(matrix, fit_fn=truth_fit_fn(truth), holdouts=["model-99"])


# -- the headline curve ------------------------------------------------------


@pytest.fixture(scope="module")
def curve():
    """The full sweep, computed once: tau against n on a synthetic universe."""
    matrix, truth = synthetic_matrix(n_models=8, n_items=400, seed=3)
    report = leave_one_model_out(matrix, fit_fn=truth_fit_fn(truth), sizes=DEFAULT_SIZES)
    return report


def test_tau_approaches_one_as_n_approaches_the_whole_suite(curve):
    """The done-when condition for this brief.

    With a fitter returning ground-truth parameters, an anchor set the size of
    the suite must reproduce the full-suite ranking exactly — anything else
    would mean the holdout scoring itself is wrong. The interesting part is
    that it gets there long before n reaches the whole suite.
    """
    last = curve.results[-1]
    assert last.size == 400
    # 400 of 400 requested, but ceiling and floor items are never eligible, so the
    # set is not quite the suite: one near-tie may swap half a place.
    assert last.mean_error < 0.25
    assert last.kendall == pytest.approx(1.0)
    assert last.spearman == pytest.approx(1.0)


def test_the_curve_is_reported_at_every_requested_size(curve):
    assert [r.size for r in curve.results] == list(DEFAULT_SIZES)
    for result in curve.results:
        assert len(result.models) == 8
        assert np.isfinite(result.kendall)
        assert np.isfinite(result.random_error)
        assert len(result.random_errors) == 200


def test_small_anchor_sets_are_not_worse_than_chance(curve):
    """Even 25 items chosen where the models sit carry most of the ranking."""
    assert curve.results[0].kendall > 0.5
    assert curve.results[0].mean_error < 1.5


def test_the_curve_is_broadly_monotone_in_n(curve):
    taus = [r.kendall for r in curve.results]
    assert taus[-1] >= taus[0]


def test_report_serialises_to_json(curve):
    payload = json.loads(json.dumps(curve.to_dict()))
    assert payload["n_models"] == 8
    assert payload["headline"] == "place_error"
    assert payload["random_draws"] == 200
    assert {"place_error", "random_place_error", "beats_random"} <= set(payload["sizes"][0])
    assert [s["size"] for s in payload["sizes"]] == list(DEFAULT_SIZES)
    assert payload["sizes"][0]["models"][0]["model_id"] == "model-00"


# -- refusals ----------------------------------------------------------------


def test_two_models_is_refused_with_the_reason():
    matrix, truth = synthetic_matrix(n_models=2, n_items=30, variants_per_model=4, seed=13)
    with pytest.raises(ValidateError, match="at least 3 real models"):
        leave_one_model_out(matrix, fit_fn=truth_fit_fn(truth), sizes=(5,))


def test_parse_sizes_sorts_deduplicates_and_rejects_nonsense():
    assert parse_sizes("100,25,25,50") == (25, 50, 100)
    assert parse_sizes([10, 5]) == (5, 10)
    with pytest.raises(ValidateError, match="comma-separated integers"):
        parse_sizes("25,fifty")
    with pytest.raises(ValidateError, match="must be positive"):
        parse_sizes("0,10")
    with pytest.raises(ValidateError, match="at least one anchor size"):
        parse_sizes("")


# -- the command -------------------------------------------------------------


def test_command_explains_a_missing_response_matrix(tmp_path: Path, capsys):
    """--no-embed-responses is an opt-out that has to fail legibly."""
    import typer

    from irtcheck.commands import validate as impl

    fit, _ = synthetic_fit(n_models=4, n_items=20, seed=14)
    fit.responses = None
    path = fit.save(tmp_path / "no-responses.irt")

    with pytest.raises(typer.Exit) as exc:
        impl.run(artifact=path, sizes="10", as_json=False, seed=0, epochs=1)
    assert exc.value.exit_code == 2
    assert "--no-embed-responses" in plain(capsys.readouterr().err)


def test_command_rejects_bad_sizes(tmp_path: Path, capsys):
    import typer

    from irtcheck.commands import validate as impl

    fit, _ = synthetic_fit(n_models=4, n_items=20, seed=15)
    path = fit.save(tmp_path / "suite.irt")
    with pytest.raises(typer.Exit) as exc:
        impl.run(artifact=path, sizes="-5", as_json=False, seed=0, epochs=1)
    assert exc.value.exit_code == 2
    assert "positive" in plain(capsys.readouterr().err)


def test_command_says_where_the_fitter_is_when_it_is_missing(tmp_path: Path, capsys, monkeypatch):
    """Until the wiring commit lands, this is the CLI's honest answer.

    The fitter is made unimportable rather than assumed absent: once wave1/fit
    merges, `irtcheck.fit.fitter` exists, and a test that reached it would
    quietly start running five real SVI fits in the fast lane.
    """
    import sys

    import typer

    from irtcheck.commands import validate as impl

    monkeypatch.setitem(sys.modules, "irtcheck.fit.fitter", None)
    fit, _ = synthetic_fit(n_models=4, n_items=20, seed=16)
    path = fit.save(tmp_path / "suite.irt")
    with pytest.raises(typer.Exit) as exc:
        impl.run(artifact=path, sizes="5", as_json=False, seed=0, epochs=1)
    assert exc.value.exit_code == 2
    message = plain(capsys.readouterr().err)
    assert "fitter" in message
    assert "leave_one_model_out" in message


def test_command_renders_and_serialises(tmp_path: Path, capsys, monkeypatch):
    from irtcheck.commands import validate as impl

    matrix, truth = synthetic_matrix(n_models=5, n_items=60, seed=17)
    fit = truth_fit_fn(truth)(matrix)
    path = fit.save(tmp_path / "suite.irt")
    monkeypatch.setattr(impl, "default_fit_fn", lambda **_: truth_fit_fn(truth))

    impl.run(artifact=path, sizes="10,30", as_json=False, seed=0, epochs=1)
    rendered = " ".join(plain(capsys.readouterr().out).split())
    assert "Leave-one-model-out" in rendered
    assert "5 models" in rendered
    assert "beats random" in rendered

    impl.run(artifact=path, sizes="10,30", as_json=True, seed=0, epochs=1)
    payload = json.loads(capsys.readouterr().out)
    assert [s["size"] for s in payload["sizes"]] == [10, 30]
    assert payload["source"] == str(path)


def test_the_reason_random_wins_depends_on_the_model_count():
    """Below the measured count the estimates are to blame, not the suite.

    The first version of this message blamed the suite's subject mix, and it
    printed that on a one-dimensional synthetic matrix, where there is no mix.
    """
    from irtcheck.commands.validate import SELECTION_NEEDS_MODELS, why_random_wins

    few = why_random_wins(SELECTION_NEEDS_MODELS - 1)
    many = why_random_wins(SELECTION_NEEDS_MODELS)
    assert "estimates" in few and "at random" in few
    assert "mix" not in few
    assert "this suite" in many and "estimates" not in many
