"""Does the fitter get the right answer back out of data it knows the truth of?

Ground truth exists only in synth.py, so every claim here is checked against
parameters generated there. The fits are seeded with `pyro.set_rng_seed` and run
on CPU, and each assertion is a tolerance on recovery rather than a value —
SVI is stochastic, and a flaky test across four concurrent agents costs more
than it catches. See CLAUDE.md → Testing a stochastic fit.

## How much recovery is possible at 15 respondents

The brief for this work asked for correlation above 0.9 between recovered and
generating `a` at 15 respondents x 500 items. That is not reachable, and the
limit is the data rather than the fitter. Fifteen binary responses per item is
about three units of Fisher information about a_i, so a standard error near 0.5
against a generating spread of 0.53.

Measured directly: integrating the exact per-item posterior on a 121 x 181
grid over (a, b), given the *true* abilities and the *true* generating prior —
the Bayes-optimal estimator, which no fitter can beat — recovers a at 0.52 and
b at 0.87 on this matrix. This fitter reaches 0.50-0.59 and 0.83-0.85 across
data seeds, i.e. within a few points of the ceiling.

So the 0.9 bar is asserted where it is achievable, at 300 respondents, and the
15-respondent case asserts closeness to the statistical ceiling instead. The
recovery curve for `a` on this generator, at 500 items and the default 2000
epochs, is roughly:

    15 respondents  0.50      60 respondents  0.75
    150 respondents 0.87      300 respondents 0.93
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyro", reason="the fitter and its tests need torch and pyro")

from irtcheck.artifact import (  # noqa: E402
    FLAG_DEAD,
    FLAG_INSUFFICIENT_DATA,
    ArtifactError,
    IrtFit,
    Posterior,
)
from irtcheck.fit.fitter import FitError, canonical_sign, fit_matrix  # noqa: E402
from irtcheck.synth import synthetic_matrix  # noqa: E402

pytestmark = pytest.mark.slow

# The bar the brief set, asserted where the data can support it.
STRONG_CORRELATION = 0.9


def align(fit: IrtFit, truth):
    """Truth in the fit's own item and respondent order.

    A ragged matrix's items are in first-seen order, which is *not* the order
    synth generated them in — comparing the two positionally is a test that
    passes on dense matrices and quietly means nothing on sparse ones.
    """
    item_at = {item: i for i, item in enumerate(truth.item_ids)}
    resp_at = {r: i for i, r in enumerate(truth.respondent_ids)}
    items = np.array([item_at[i] for i in fit.item_ids])
    respondents = np.array([resp_at[r] for r in fit.respondent_ids])
    return truth.a[items], truth.b[items], truth.theta[respondents]


def correlation(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


@pytest.fixture(scope="module")
def thin_fit():
    """15 respondents x 500 items — the size the tool is actually aimed at."""
    matrix, truth = synthetic_matrix(n_models=15, n_items=500, seed=0)
    return fit_matrix(matrix, epochs=2000, seed=0), truth


@pytest.fixture(scope="module")
def deep_fit():
    """300 respondents: the regime where psychometrics is comfortable, and the
    only one where a discrimination can be pinned down to a few percent."""
    matrix, truth = synthetic_matrix(n_models=300, n_items=500, seed=0)
    return fit_matrix(matrix, epochs=2000, seed=0), truth


# -- recovery ----------------------------------------------------------------


def test_recovers_item_parameters_at_fifteen_respondents(thin_fit):
    """Within a few points of the Bayes-optimal ceiling (0.52 for a, 0.87 for
    b on this generator). See the module docstring for why 0.9 is not on the
    table here."""
    fit, truth = thin_fit
    true_a, true_b, _ = align(fit, truth)
    assert correlation(fit.a.mean, true_a) > 0.45
    assert correlation(fit.b.mean, true_b) > 0.80


def test_recovers_abilities_at_fifteen_respondents(thin_fit):
    """Abilities are the easy half: every respondent has 500 responses."""
    fit, truth = thin_fit
    _, _, true_theta = align(fit, truth)
    assert correlation(fit.theta.mean, true_theta) > 0.95


def test_recovery_reaches_the_strong_bar_when_respondents_allow(deep_fit):
    """The brief's 0.9, asserted at the respondent count that can support it."""
    fit, truth = deep_fit
    true_a, true_b, _ = align(fit, truth)
    assert correlation(fit.a.mean, true_a) > STRONG_CORRELATION
    assert correlation(fit.b.mean, true_b) > STRONG_CORRELATION


def test_more_respondents_recover_more(thin_fit, deep_fit):
    """The number that should move when a user follows the tool's own advice."""
    thin, thin_truth = thin_fit
    deep, deep_truth = deep_fit
    assert correlation(deep.a.mean, align(deep, deep_truth)[0]) > correlation(
        thin.a.mean, align(thin, thin_truth)[0]
    )


def test_recovers_from_a_ragged_matrix():
    """Not every model answers every item. The fit works in sparse triplets, so
    a third of the grid missing costs accuracy and nothing else."""
    matrix, truth = synthetic_matrix(n_models=15, n_items=500, seed=0, missing=0.3)
    assert matrix.n_responses < matrix.n_items * matrix.n_respondents
    fit = fit_matrix(matrix, epochs=2000, seed=0)
    true_a, true_b, true_theta = align(fit, truth)
    assert correlation(fit.a.mean, true_a) > 0.35
    assert correlation(fit.b.mean, true_b) > 0.65
    assert correlation(fit.theta.mean, true_theta) > 0.90


def test_vague_priors_also_recover(thin_fit):
    """`--priors vague` is a real alternative, not a trap door: it recovers
    comparably and its interval on `a` is wider, because nothing pools."""
    matrix, truth = synthetic_matrix(n_models=15, n_items=500, seed=0)
    fit = fit_matrix(matrix, epochs=2000, seed=0, priors="vague")
    true_a, true_b, _ = align(fit, truth)
    assert correlation(fit.a.mean, true_a) > 0.45
    assert correlation(fit.b.mean, true_b) > 0.80
    assert fit.model["priors"] == "vague"

    pooled, _ = thin_fit
    widths = np.array(fit.a.hdi_high) - np.array(fit.a.hdi_low)
    pooled_widths = np.array(pooled.a.hdi_high) - np.array(pooled.a.hdi_low)
    assert widths.mean() > pooled_widths.mean()


# -- identification ----------------------------------------------------------


def test_the_fit_is_not_mirrored(thin_fit):
    """The reflection: negating a, b and theta together fits identically. A fit
    that returned the mirror would invert every ability with nothing to see."""
    fit, truth = thin_fit
    _, _, true_theta = align(fit, truth)
    assert correlation(fit.theta.mean, true_theta) > 0
    assert np.mean(fit.a.mean) > 0
    assert fit.diagnostics["reflected"] is False


def test_canonical_sign_detects_a_mirrored_solution():
    """The check itself, on a posterior built to be in the wrong mode — the
    fitted path should never reach it, which is exactly why it needs its own
    test."""
    positive = Posterior(mean=[0.4, 1.2], sd=[0.1, 0.1], hdi_low=[0.2, 1.0], hdi_high=[0.6, 1.4])
    mirrored = Posterior(
        mean=[-0.4, -1.2], sd=[0.1, 0.1], hdi_low=[-0.6, -1.4], hdi_high=[-0.2, -1.0]
    )
    assert canonical_sign(positive) == 1
    assert canonical_sign(mirrored) == -1


def test_abilities_stay_on_the_prior_scale(thin_fit):
    """theta ~ N(0, 1) is the ruler that fixes location and scale. Posterior
    means shrink toward the prior, so the spread is below one rather than at
    it — but the centre has to hold, or `a` and `b` are on a scale nothing else
    knows about."""
    fit, _ = thin_fit
    theta = np.array(fit.theta.mean)
    assert abs(theta.mean()) < 0.35
    assert 0.4 < theta.std() < 1.4


# -- flags -------------------------------------------------------------------


def test_dead_items_are_the_ones_refused_at_fifteen_respondents(thin_fit):
    """The refusal path, checked against which items are *actually* flat.

    At 15 respondents most low-information items land in insufficient-data
    rather than dead, and this asserts the flag lands on the right items far
    more often than on the wrong ones — not that it is decisive, which at this
    respondent count it cannot honestly be."""
    fit, truth = thin_fit
    true_a, _, _ = align(fit, truth)
    refused = np.array([FLAG_INSUFFICIENT_DATA in flags for flags in fit.flags])
    flat = true_a < 0.3
    assert refused[flat].mean() > 0.8
    assert refused[flat].mean() > refused[~flat].mean() + 0.2


def test_refusal_recedes_as_respondents_are_added(thin_fit, deep_fit):
    """The tool's own advice, checked: add respondents and it stops refusing."""
    thin, _ = thin_fit
    deep, _ = deep_fit
    assert len(deep.flagged(FLAG_INSUFFICIENT_DATA)) < len(thin.flagged(FLAG_INSUFFICIENT_DATA))


def test_no_item_is_both_dead_and_undetermined(thin_fit, deep_fit):
    """'we are confident it does not discriminate' and 'we cannot tell' are
    mutually exclusive claims. IrtFit.validate() enforces it; this says the
    fitter never produces one."""
    for fit, _ in (thin_fit, deep_fit):
        for flags in fit.flags:
            assert not (FLAG_DEAD in flags and FLAG_INSUFFICIENT_DATA in flags)


def test_usable_items_exclude_the_undetermined(thin_fit):
    fit, _ = thin_fit
    usable = set(fit.usable_items())
    assert usable
    assert not usable & set(fit.flagged(FLAG_INSUFFICIENT_DATA))


# -- the artifact ------------------------------------------------------------


def test_the_artifact_is_complete_and_round_trips(tmp_path, thin_fit):
    fit, _ = thin_fit
    path = fit.save(tmp_path / "suite.irt")
    reloaded = IrtFit.load(path)
    assert reloaded.item_ids == fit.item_ids
    assert reloaded.a.mean == fit.a.mean
    assert reloaded.flags == fit.flags
    assert reloaded.model["kind"] == "2pl"
    # Byte-identical, which is what the contracts job asserts about synth's
    # artifacts and has to hold for real ones too.
    assert reloaded.save(tmp_path / "again.irt").read_bytes() == path.read_bytes()


def test_the_artifact_carries_its_responses_so_validate_can_refit(thin_fit):
    """`irtcheck validate suite.irt` takes one argument, and leave-one-model-out
    refits k times — so the matrix has to travel with the fit."""
    fit, _ = thin_fit
    assert fit.responses is not None
    rebuilt = fit.matrix()
    assert rebuilt.n_responses == len(fit.responses)
    assert rebuilt.item_ids == fit.item_ids
    assert rebuilt.derives_from == fit.derives_from
    assert rebuilt.respondent_key == tuple(fit.respondent_key)


def test_no_embed_responses_leaves_a_fit_that_explains_itself():
    matrix, _ = synthetic_matrix(n_models=5, n_items=30, seed=2)
    fit = fit_matrix(matrix, epochs=50, seed=0, embed_responses=False)
    assert fit.responses is None
    with pytest.raises(ArtifactError, match="no-embed-responses"):
        fit.matrix()


def test_diagnostics_carry_the_fit_history(thin_fit):
    fit, _ = thin_fit
    diagnostics = fit.diagnostics
    assert diagnostics["epochs"] == 2000
    assert diagnostics["seed"] == 0
    assert diagnostics["priors"] == "hierarchical"
    assert diagnostics["device"] == "cpu"
    history = diagnostics["elbo_history"]
    assert len(history) == 2000
    assert history[-1] == diagnostics["elbo_final"]
    # It is a fit, so it should have improved on where it started.
    assert history[-1] > history[0]
    assert diagnostics["n_real_models"] == 15
    assert diagnostics["flag_counts"][FLAG_INSUFFICIENT_DATA] == len(
        fit.flagged(FLAG_INSUFFICIENT_DATA)
    )


def test_a_long_run_thins_its_elbo_history():
    """A 50k-epoch fit should not put 50k floats in the artifact."""
    matrix, _ = synthetic_matrix(n_models=5, n_items=20, seed=2)
    fit = fit_matrix(matrix, epochs=2500, seed=0)
    history = fit.diagnostics["elbo_history"]
    assert len(history) < 2500
    assert fit.diagnostics["elbo_history_stride"] > 1
    assert history[-1] == fit.diagnostics["elbo_final"]


def test_pseudo_respondents_are_fitted_but_reported_as_their_models():
    """Prompt variants are extra respondents for estimation and *not* extra
    models for the header. Getting this wrong misrepresents the one thing the
    tool exists to be honest about."""
    matrix, _ = synthetic_matrix(n_models=5, n_items=40, variants_per_model=3, seed=4)
    fit = fit_matrix(matrix, epochs=200, seed=0)
    assert fit.n_respondents == 15
    assert fit.n_real_models == 5
    assert fit.has_pseudo_respondents
    assert fit.respondent_key == ["model_id", "prompt_variant"]
    assert fit.diagnostics["n_real_models"] == 5


# -- reproducibility and argument handling -----------------------------------


def test_the_same_seed_gives_the_same_numbers():
    """Two runs at one seed must agree exactly, and not by luck: the param store
    is global, so a fit that did not clear it would inherit the previous one.

    Exactly the posteriors, not exactly the file: `created` and the elapsed
    seconds differ between any two runs by design, so the artifacts are not
    byte-identical and should not be."""
    matrix, _ = synthetic_matrix(n_models=6, n_items=30, seed=5)
    first = fit_matrix(matrix, epochs=300, seed=11)
    second = fit_matrix(matrix, epochs=300, seed=11)
    assert first.a.mean == second.a.mean
    assert first.a.hdi_low == second.a.hdi_low
    assert first.b.mean == second.b.mean
    assert first.theta.mean == second.theta.mean
    assert first.flags == second.flags
    assert first.diagnostics["elbo_history"] == second.diagnostics["elbo_history"]


def test_a_different_seed_gives_a_different_fit(tmp_path):
    matrix, _ = synthetic_matrix(n_models=6, n_items=30, seed=5)
    one = fit_matrix(matrix, epochs=300, seed=11)
    two = fit_matrix(matrix, epochs=300, seed=12)
    assert one.a.mean != two.a.mean


def test_a_fit_after_another_fit_is_unaffected_by_it():
    """`validate` refits k times in one process. If the param store leaked
    between them, every holdout after the first would start from the previous
    one's parameters and the headline number would be quietly wrong."""
    matrix, _ = synthetic_matrix(n_models=6, n_items=30, seed=5)
    other, _ = synthetic_matrix(n_models=8, n_items=25, seed=6)
    alone = fit_matrix(matrix, epochs=300, seed=1)
    fit_matrix(other, epochs=300, seed=99)
    after = fit_matrix(matrix, epochs=300, seed=1)
    assert alone.a.mean == after.a.mean


def test_epochs_must_be_positive():
    matrix, _ = synthetic_matrix(n_models=5, n_items=10, seed=0)
    with pytest.raises(FitError, match="at least 1"):
        fit_matrix(matrix, epochs=0)


def test_an_unknown_prior_family_is_refused():
    matrix, _ = synthetic_matrix(n_models=5, n_items=10, seed=0)
    with pytest.raises(FitError, match="hierarchical"):
        fit_matrix(matrix, epochs=10, priors="uniform")


def test_cuda_is_refused_clearly_when_there_is_no_cuda():
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA, so there is no error to report")
    matrix, _ = synthetic_matrix(n_models=5, n_items=10, seed=0)
    with pytest.raises(FitError, match="cpu"):
        fit_matrix(matrix, epochs=10, device="cuda")
