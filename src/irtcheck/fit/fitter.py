"""SVI over the 2PL, and the artifact that comes out of it.

`fit_2pl(matrix) -> IrtFit` is the entry point everything else uses:
`irtcheck fit` calls it once, and `irtcheck validate` calls it k times for the
leave-one-model-out refits. It takes a ResponseMatrix and returns a fully
formed, already-validated artifact — no half-populated intermediate object that
a caller has to know how to finish.

Stochastic variational inference rather than MCMC: a 4000-item suite is ~8000
latent parameters, NUTS on that is minutes-to-hours, and the intervals this
tool acts on are coarse decisions ("does the interval on a_i contain zero")
rather than tail probabilities.

Mean-field SVI understates posterior correlations, so its marginals are too
**narrow** — and for this tool that is the dangerous direction, not the safe
one. `insufficient-data` fires when the interval on `a_i` spans zero, so a
narrower interval makes the tool refuse *less*: it claims a discrimination it
has not earned. An earlier version of this docstring asserted the opposite,
that narrow intervals made the flag "conservative, never over-eager". That was
wrong, and measurement is what settled it: the nominal 95% variational interval
on `a` covered the true value 87.5% of the time at 15 respondents, 84.3% at 60
and 80.6% at 300, against a nominal 95%.

So the point estimates come from SVI and the item **intervals do not**. They
are computed from the conditional information matrix instead, which carries the
`a`-`b` correlation the guide drops and lifts coverage to 91-93%. See
fit/intervals.py for the derivation and what it still does not fix, and
tests/test_intervals.py, which measures the claim rather than restating it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from irtcheck import __version__
from irtcheck.artifact import EmbeddedResponses, IrtFit, Posterior, utc_now
from irtcheck.fit.flags import flag_counts, item_flags
from irtcheck.fit.intervals import item_posteriors, prior_precisions, respondent_posterior
from irtcheck.fit.model import (
    PRIORS_HIERARCHICAL,
    ModelError,
    check_priors,
    make_guide,
    make_model,
    model_metadata,
    to_tensors,
)
from irtcheck.fit.summaries import posterior_for

DEFAULT_EPOCHS = 2000
# Tuned on synthetic 15 x 500 matrices: high enough to converge inside the
# default epoch budget, low enough that the ELBO trace is monotone rather than
# a sawtooth. `lrd` decays it to ~1/10 by the last step, which is what stops
# the item parameters from rattling around their optimum at the end of the run.
DEFAULT_LR = 0.05
LR_DECAY_TOTAL = 0.1
# Draws used only when a guide site is *not* an unconstrained Normal, which the
# current guide never produces. See summaries.posterior_for.
POSTERIOR_SAMPLES = 2000
# The ELBO trace is a diagnostic, not a dataset: past this many points it is
# thinned, so a 50k-epoch fit does not put 50k floats in every artifact.
MAX_ELBO_HISTORY = 2000
# Recorded in diagnostics. Artifacts written before the item intervals moved off
# the variational marginals carry no such key, which is how they are told apart.
INTERVAL_METHOD = "conditional-information"


class FitError(RuntimeError):
    pass


def fit_2pl(
    matrix,
    *,
    priors: str = PRIORS_HIERARCHICAL,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 0,
    device: str = "cpu",
    lr: float = DEFAULT_LR,
    embed_responses: bool = True,
    on_step: Callable[[int, float], None] | None = None,
) -> IrtFit:
    """Fit a 2PL to `matrix` and return the artifact.

    Deterministic given `seed`: the RNG is seeded, the param store is cleared,
    and the posterior summaries are closed-form rather than sampled, so two runs
    at one seed produce the same posteriors to the last digit. Not the same
    *file* — `created` and the elapsed seconds are properties of the run, not of
    the fit.
    """
    try:
        check_priors(priors)
    except ModelError as exc:
        raise FitError(str(exc)) from exc
    if epochs < 1:
        raise FitError(f"--epochs must be at least 1, got {epochs}")
    torch_device = _resolve_device(device)

    # Both of these, every time. Seeding alone is not reproducibility: Pyro's
    # param store is global, so `validate`'s second refit would otherwise start
    # from the first refit's fitted parameters and quietly converge somewhere a
    # standalone fit never would.
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    data = to_tensors(matrix, device=torch_device)
    model = make_model(priors)
    guide = make_guide(model, data, priors=priors)

    optimizer = ClippedAdam({"lr": lr, "lrd": LR_DECAY_TOTAL ** (1.0 / max(epochs, 1))})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    started = time.perf_counter()
    elbo_history: list[float] = []
    for step in range(epochs):
        # SVI reports the loss, which is the *negative* ELBO. Store the ELBO:
        # a diagnostic someone reads should go up as the fit improves.
        elbo = -float(svi.step(data))
        if not np.isfinite(elbo):
            raise FitError(
                f"the ELBO went non-finite at step {step}. That is a diverged "
                "optimisation, not a bad matrix: retry with a lower learning rate "
                "(fit_2pl(..., lr=...)) and report it, because the default is "
                "meant to be safe on any matrix this tool accepts."
            )
        elbo_history.append(elbo)
        if on_step is not None:
            on_step(step, elbo)
    elapsed = time.perf_counter() - started

    # posterior_for supplies the means for every site. The widths are then
    # recomputed — from curvature for the items, and from information plus the
    # scale uncertainty for the abilities. See fit/intervals.py; both are too
    # narrow coming out of a mean-field guide, for two different reasons.
    theta_svi = posterior_for(guide, "theta", n_samples=POSTERIOR_SAMPLES, data=data)
    a_svi = posterior_for(guide, "a", n_samples=POSTERIOR_SAMPLES, data=data)
    b_svi = posterior_for(guide, "b", n_samples=POSTERIOR_SAMPLES, data=data)

    hyperparameters = _hyperparameters(guide, data, priors)
    precision_a, precision_b = prior_precisions(hyperparameters, priors=priors)
    a, b = item_posteriors(
        data,
        a_mean=a_svi.mean,
        b_mean=b_svi.mean,
        theta_mean=theta_svi.mean,
        precision_a=precision_a,
        precision_b=precision_b,
    )
    theta = respondent_posterior(
        data, a_mean=a.mean, b_mean=b.mean, theta_mean=theta_svi.mean
    )

    # The reflection check. Initialising the guide at a = +1 should always land
    # in the positive mode, but "should" is not a property anyone can read off
    # an artifact, and a fit that silently returned the mirror solution would
    # invert every ability and every difficulty while fitting the data exactly
    # as well.
    reflected = canonical_sign(a) < 0
    if reflected:
        a, b, theta = _reflect(a), _reflect(b), _reflect(theta)

    n_resp = matrix.item_response_counts()
    p_correct = np.nan_to_num(matrix.item_p_correct(), nan=0.0)
    flags = item_flags(a, b, [float(p) for p in p_correct], theta.mean)

    fit = IrtFit(
        irtcheck_version=__version__,
        created=utc_now(),
        model=model_metadata(priors),
        respondent_key=list(matrix.respondent_key),
        respondent_ids=list(matrix.respondent_ids),
        derives_from=list(matrix.derives_from),
        theta=theta,
        item_ids=list(matrix.item_ids),
        a=a,
        b=b,
        n_resp=[int(x) for x in n_resp],
        p_correct=[float(p) for p in p_correct],
        flags=flags,
        diagnostics=_diagnostics(
            elbo_history=elbo_history,
            epochs=epochs,
            seed=seed,
            priors=priors,
            device=str(torch_device),
            lr=lr,
            elapsed=elapsed,
            matrix=matrix,
            flags=flags,
            reflected=reflected,
            hyperparameters=hyperparameters,
        ),
        passthrough={k: dict(v) for k, v in matrix.passthrough.items()},
        responses=(
            EmbeddedResponses(
                rows=[int(x) for x in matrix.rows],
                cols=[int(x) for x in matrix.cols],
                obs=[int(x) for x in matrix.obs],
            )
            if embed_responses
            else None
        ),
    )
    fit.validate()
    return fit


def _hyperparameters(guide, data, priors: str) -> dict[str, float]:
    """The population scales the fit settled on, as plain floats.

    Recorded in diagnostics and read by intervals.prior_precisions, which needs
    the prior precision every item falls back to. They were previously nowhere
    in the artifact, which cost the wave-2 cross-check a whole second fit just
    to recover them: crosscheck/run_gridpost.py refits with the same seed and
    reads them off the guide, checking that the refit reproduced the artifact
    because it has no other way to know.

    Empty under vague priors, where nothing is learned — the scales are the
    constants model.py declares, and prior_precisions reads them from there.
    """
    if priors != PRIORS_HIERARCHICAL:
        return {}
    median = guide.median(data)
    return {
        name: float(median[name].detach().cpu())
        for name in ("sigma_a", "mu_b", "sigma_b")
    }


def canonical_sign(a: Posterior) -> int:
    """+1 if a fit sits in the positive-discrimination mode, -1 if mirrored.

    Negating a, b and theta together leaves every response probability
    unchanged, so both are equally good fits and only a convention separates
    them. Ours is "most items discriminate positively", which is what the whole
    scale means: higher theta answers more items correctly.
    """
    return 1 if sum(a.mean) >= 0 else -1


def _reflect(p: Posterior) -> Posterior:
    """Negate a posterior, swapping the interval ends so it stays ordered."""
    return Posterior(
        mean=[-m for m in p.mean],
        sd=list(p.sd),
        hdi_low=[-h for h in p.hdi_high],
        hdi_high=[-lo for lo in p.hdi_low],
    )


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise FitError(
            "--device cuda was asked for but this torch build sees no CUDA device. "
            "Fitting is small enough that --device cpu is usually the right answer."
        )
    return resolved


def _thin(history: list[float], limit: int = MAX_ELBO_HISTORY) -> tuple[list[float], int]:
    """Subsample an ELBO trace, always keeping the last point."""
    if len(history) <= limit:
        return list(history), 1
    stride = (len(history) + limit - 1) // limit
    kept = history[::stride]
    if kept[-1] != history[-1]:
        kept.append(history[-1])
    return kept, stride


def _diagnostics(
    *,
    elbo_history: list[float],
    epochs: int,
    seed: int,
    priors: str,
    device: str,
    lr: float,
    elapsed: float,
    matrix,
    flags: list[list[str]],
    reflected: bool,
    hyperparameters: dict[str, float],
) -> dict:
    """Everything needed to answer "did this fit converge, and on what?".

    `elbo_improvement_last_10pct` is the one to read first: an ELBO still
    climbing appreciably over the final tenth of the run has not converged, and
    the answer is more --epochs rather than more respondents.
    """
    history, stride = _thin(elbo_history)
    tail = max(1, len(elbo_history) // 10)
    improvement = elbo_history[-1] - elbo_history[-tail]
    return {
        "elbo_final": elbo_history[-1],
        "elbo_history": history,
        "elbo_history_stride": stride,
        "elbo_improvement_last_10pct": improvement,
        "epochs": epochs,
        "seed": seed,
        "priors": priors,
        "device": device,
        "lr": lr,
        "seconds": round(elapsed, 3),
        # True means the solution came back mirrored and was flipped into the
        # positive-discrimination convention. It should never be true; if it is,
        # the guide initialisation is not doing its job.
        "reflected": reflected,
        "n_responses": int(matrix.n_responses),
        "n_respondents": int(matrix.n_respondents),
        "n_real_models": int(matrix.n_real_models),
        "n_items": int(matrix.n_items),
        "density": float(matrix.density),
        "flag_counts": flag_counts(flags),
        # The learned population scales, so nothing downstream has to refit to
        # recover them. Empty under vague priors; see _hyperparameters.
        "hyperparameters": hyperparameters,
        # Which method produced the reported widths. The means are variational
        # either way; this says where the intervals came from, so an artifact
        # written before the change is distinguishable from one written after.
        "interval_method": INTERVAL_METHOD,
    }
