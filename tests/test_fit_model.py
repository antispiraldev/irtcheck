"""The model and guide: shapes, priors, starting values, ragged input.

Every module here imports torch, so every module here is skipped when torch is
absent (the `tests-light` CI job) and marked `slow` so that the `tests-fit` job,
which selects `-m slow`, actually runs it. A unit test of a tensor shape is not
slow in wall-clock terms; it is in the only sense CI cares about, which is
"needs the heavy dependency".
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyro", reason="the fitter and its tests need torch and pyro")

import torch  # noqa: E402
from pyro.infer import Trace_ELBO  # noqa: E402

from irtcheck.fit.model import (  # noqa: E402
    PRIORS_HIERARCHICAL,
    PRIORS_VAGUE,
    ModelError,
    check_priors,
    classical_start,
    make_guide,
    make_model,
    model_metadata,
    to_tensors,
    two_pl,
)
from irtcheck.synth import synthetic_matrix  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def small():
    matrix, truth = synthetic_matrix(n_models=6, n_items=40, seed=3)
    return matrix, truth


@pytest.fixture(scope="module")
def ragged():
    matrix, truth = synthetic_matrix(n_models=6, n_items=40, seed=3, missing=0.35)
    return matrix, truth


# -- priors ------------------------------------------------------------------


@pytest.mark.parametrize("priors", [PRIORS_HIERARCHICAL, PRIORS_VAGUE])
def test_both_prior_families_are_accepted(priors):
    assert check_priors(priors) == priors


def test_an_unknown_prior_family_names_the_alternatives():
    with pytest.raises(ModelError, match="hierarchical"):
        check_priors("flat")


def test_metadata_records_the_identification_choice():
    """The wave-2 cross-check reads this to reconcile conventions with mirt.
    An artifact that does not say how `a` was identified is an afternoon lost."""
    meta = model_metadata(PRIORS_HIERARCHICAL)
    assert meta["kind"] == "2pl"
    assert meta["priors"] == PRIORS_HIERARCHICAL
    identification = meta["identification"]
    assert "theta ~ N(0, 1)" in identification
    # The claim that matters: `a` is *not* truncated at zero, which is what
    # leaves `insufficient-data` reachable.
    assert "not constrained positive" in identification
    assert model_metadata(PRIORS_VAGUE)["priors"] == PRIORS_VAGUE


# -- tensors -----------------------------------------------------------------


def test_tensors_stay_sparse_and_keep_every_response(ragged):
    matrix, _ = ragged
    data = to_tensors(matrix)
    assert data.n_responses == matrix.n_responses < matrix.n_items * matrix.n_respondents
    assert data.rows.dtype == torch.long and data.cols.dtype == torch.long
    assert int(data.rows.max()) == matrix.n_respondents - 1
    assert int(data.cols.max()) == matrix.n_items - 1
    assert set(data.obs.unique().tolist()) <= {0.0, 1.0}


def test_tensors_do_not_densify(ragged):
    """A ragged matrix has no rectangle to fill in, and inventing one would
    turn "this model never answered that item" into "it got it wrong"."""
    matrix, _ = ragged
    data = to_tensors(matrix)
    assert data.obs.shape == (matrix.n_responses,)
    assert data.obs.numel() < matrix.n_respondents * matrix.n_items


# -- classical starting values ----------------------------------------------


def test_classical_start_puts_theta_on_the_prior_scale(small):
    matrix, truth = small
    _, theta0 = classical_start(to_tensors(matrix))
    assert theta0.shape == (matrix.n_respondents,)
    assert float(theta0.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(theta0.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


def test_classical_start_orders_respondents_by_score(small):
    """theta starts positively correlated with the score a user would compute
    by hand — which is also what keeps the fit out of the mirrored solution."""
    matrix, _ = small
    _, theta0 = classical_start(to_tensors(matrix))
    accuracy = torch.as_tensor(matrix.respondent_accuracy(), dtype=torch.float32)
    assert float(torch.corrcoef(torch.stack([theta0, accuracy]))[0, 1]) > 0.99


def test_classical_start_puts_hard_items_at_a_high_difficulty(small):
    matrix, _ = small
    b0, _ = classical_start(to_tensors(matrix))
    p = torch.as_tensor(matrix.item_p_correct(), dtype=torch.float32)
    # Not -1.0: the logit is non-linear and the rates are clamped away from
    # the ends, so the relation is monotone rather than affine.
    assert float(torch.corrcoef(torch.stack([b0, p]))[0, 1]) < -0.95


def test_classical_start_survives_an_item_nobody_got_right(small):
    """logit(0) is not a number to start an optimiser at."""
    matrix, _ = small
    data = to_tensors(matrix)
    data.obs = torch.zeros_like(data.obs)
    b0, theta0 = classical_start(data)
    assert torch.isfinite(b0).all() and torch.isfinite(theta0).all()


def test_classical_start_survives_a_single_respondent():
    from irtcheck.matrix import build_matrix
    from irtcheck.records import ResponseRecord

    matrix = build_matrix(
        ResponseRecord(model_id="m", item_id=f"i{i}", correct=i % 2) for i in range(6)
    )
    b0, theta0 = classical_start(to_tensors(matrix))
    assert torch.isfinite(b0).all() and torch.isfinite(theta0).all()


# -- the model itself --------------------------------------------------------


@pytest.mark.parametrize("priors", [PRIORS_HIERARCHICAL, PRIORS_VAGUE])
def test_the_model_runs_and_the_guide_covers_every_latent(small, priors):
    import pyro

    matrix, _ = small
    data = to_tensors(matrix)
    pyro.clear_param_store()
    model = make_model(priors)
    guide = make_guide(model, data, priors=priors)

    sample = guide(data)
    assert sample["a"].shape == (matrix.n_items,)
    assert sample["b"].shape == (matrix.n_items,)
    assert sample["theta"].shape == (matrix.n_respondents,)
    if priors == PRIORS_HIERARCHICAL:
        assert {"sigma_a", "mu_b", "sigma_b"} <= set(sample)
    else:
        assert not {"sigma_a", "mu_b", "sigma_b"} & set(sample)

    # A finite ELBO is the cheapest proof that model and guide agree about
    # every site's shape and support.
    loss = Trace_ELBO().loss(model, guide, data)
    assert torch.isfinite(torch.tensor(loss))


def test_the_likelihood_indexes_items_not_particles(small):
    """A vectorised-particle ELBO adds a leading sample dimension. Indexing with
    `a[cols]` instead of `a[..., cols]` silently indexes *that* dimension, and
    the model then fits particle 375 of 4."""
    import pyro

    matrix, _ = small
    data = to_tensors(matrix)
    pyro.clear_param_store()
    model = make_model(PRIORS_HIERARCHICAL)
    guide = make_guide(model, data, priors=PRIORS_HIERARCHICAL)
    loss = Trace_ELBO(num_particles=4, vectorize_particles=True).loss(model, guide, data)
    assert torch.isfinite(torch.tensor(loss))


def test_the_model_rejects_an_unknown_prior_family(small):
    matrix, _ = small
    with pytest.raises(ModelError):
        two_pl(to_tensors(matrix), priors="wishful")
