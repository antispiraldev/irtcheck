"""synth.py is what decouples wave 1. If a fabricated fit is not a faithful
stand-in for a real one, three briefs are building against a fiction."""

from __future__ import annotations

import numpy as np
import pytest

from irtcheck.artifact import FLAG_INSUFFICIENT_DATA, IrtFit
from irtcheck.synth import make_truth, responses_from_truth, synthetic_fit, synthetic_matrix


def test_synthetic_fit_is_a_valid_artifact(tmp_path):
    fit, truth = synthetic_fit(n_models=8, n_items=100, seed=0)
    fit.validate()
    assert isinstance(IrtFit.load(fit.save(tmp_path / "s.irt")), IrtFit)
    assert fit.n_items == truth.n_items


def test_synthetic_fit_needs_no_fitter():
    """The whole point: report/select/validate/html are unblocked on day one."""
    import sys

    synthetic_fit(n_models=5, n_items=20, seed=0)
    assert "torch" not in sys.modules, "synth pulled in torch; the lazy boundary is broken"


def test_generation_is_reproducible():
    a, _ = synthetic_fit(n_models=6, n_items=50, seed=42)
    b, _ = synthetic_fit(n_models=6, n_items=50, seed=42)
    assert a.a.mean == b.a.mean
    assert a.flags == b.flags


def test_different_seeds_differ():
    a, _ = synthetic_fit(n_models=6, n_items=50, seed=1)
    b, _ = synthetic_fit(n_models=6, n_items=50, seed=2)
    assert a.a.mean != b.a.mean


def test_posterior_means_are_near_but_not_equal_to_truth():
    """A fabricated artifact whose means are exactly the generating parameters
    would let a consumer depend on precision no real fit provides."""
    fit, truth = synthetic_fit(n_models=12, n_items=200, seed=5, precision=4.0)
    recovered = np.asarray(fit.a.mean)
    assert not np.allclose(recovered, truth.a)
    assert np.corrcoef(recovered, truth.a)[0, 1] > 0.9


def test_the_universe_contains_items_worth_reporting_on():
    """A suite where every item discriminates is one this tool has nothing to
    say about, so the generator must produce dead and off-range populations."""
    truth = make_truth(n_models=10, n_items=400, seed=0)
    assert (truth.a < 0.25).sum() > 20, "no near-flat items generated"
    assert (np.abs(truth.b) > 3.0).sum() > 10, "no off-range items generated"


def test_small_n_produces_undecidable_items():
    """At realistic respondent counts most low-information items land in
    insufficient-data, which is the true state of knowledge."""
    fit, _ = synthetic_fit(n_models=8, n_items=300, seed=0)
    assert len(fit.flagged(FLAG_INSUFFICIENT_DATA)) > 0


# -- matrices ----------------------------------------------------------------


def test_missing_cells_produce_a_ragged_matrix():
    matrix, truth = synthetic_matrix(n_models=6, n_items=50, missing=0.2, seed=0)
    assert matrix.n_responses < truth.n_respondents * truth.n_items
    assert matrix.density == pytest.approx(0.8, abs=0.06)


def test_pseudo_respondents_carry_provenance():
    matrix, _ = synthetic_matrix(n_models=4, n_items=30, variants_per_model=3, seed=0)
    assert matrix.n_respondents == 12
    assert matrix.n_real_models == 4
    assert matrix.respondent_key == ("model_id", "prompt_variant")
    assert len(matrix.drop_model("model-00").respondent_ids) == 9


def test_responses_follow_the_2pl():
    """Higher ability answers a given item correctly more often. If this fails,
    every recovery test downstream is meaningless."""
    truth = make_truth(n_models=200, n_items=1, dead_fraction=0.0, off_range_fraction=0.0, seed=0)
    responses = responses_from_truth(truth, seed=1)[:, 0]
    strong = responses[truth.theta > 0.5].mean()
    weak = responses[truth.theta < -0.5].mean()
    assert strong > weak


def test_embedded_responses_rebuild_the_matrix():
    """validate refits from the artifact alone, so this path is load-bearing."""
    fit, _ = synthetic_fit(n_models=6, n_items=40, seed=0)
    rebuilt = fit.matrix()
    assert rebuilt.n_items == fit.n_items
    assert rebuilt.respondent_ids == fit.respondent_ids
    np.testing.assert_array_equal(rebuilt.item_response_counts(), fit.n_resp)
