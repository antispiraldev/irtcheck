"""Synthetic matrices and fabricated fits.

This module is why four agents can build wave 1 at once. `report`, `select`,
`validate` and the HTML output all consume an IrtFit; none of them needs a
fitter to exist. `synthetic_fit()` produces a *valid* IrtFit with known
ground-truth parameters without going anywhere near Pyro, so those four briefs
are unblocked from day one rather than waiting on the slowest one.

It is also the only place ground truth exists at all. Every correctness claim
the tool makes about the fitter — "recovered a correlates with true a" — is
checked against parameters generated here.

Deliberately numpy-only. If this module ever imports torch, the tests-light CI
job stops proving the lazy-import boundary and the whole arrangement quietly
rots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtcheck import __version__
from irtcheck.artifact import EmbeddedResponses, IrtFit, Posterior, compute_flags, utc_now
from irtcheck.matrix import ResponseMatrix
from irtcheck.records import ResponseRecord

# 95% interval under a normal marginal, where the equal-tailed interval and the
# highest-density interval coincide. Variational posteriors here are normal by
# construction, so this is exact rather than an approximation — but a fitter
# that switches to a non-normal guide has to compute HDIs properly instead.
Z95 = 1.959963985


@dataclass(slots=True)
class SyntheticTruth:
    """The parameters a synthetic matrix was generated from."""

    model_ids: list[str]
    respondent_ids: list[str]
    derives_from: list[str]
    item_ids: list[str]
    theta: np.ndarray  # per respondent
    a: np.ndarray  # per item
    b: np.ndarray  # per item

    @property
    def n_respondents(self) -> int:
        return len(self.respondent_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def make_truth(
    *,
    n_models: int = 8,
    n_items: int = 400,
    variants_per_model: int = 1,
    dead_fraction: float = 0.15,
    off_range_fraction: float = 0.08,
    inverted_fraction: float = 0.0,
    seed: int = 0,
) -> SyntheticTruth:
    """Draw a plausible universe of item and respondent parameters.

    The shape of the draw matters more than its exact constants: a suite in
    which every item discriminates well is not a suite this tool has anything
    to say about, so `dead_fraction` of items are drawn near-flat and
    `off_range_fraction` are drawn difficult far outside where the respondents
    sit. Those two populations are what `report` and the information curve
    exist to surface.
    """
    rng = np.random.default_rng(seed)

    model_ids = [f"model-{i:02d}" for i in range(n_models)]
    model_theta = rng.normal(0.0, 1.0, size=n_models)

    respondent_ids: list[str] = []
    derives_from: list[str] = []
    theta: list[float] = []
    for m, model in enumerate(model_ids):
        for v in range(variants_per_model):
            if variants_per_model == 1:
                respondent_ids.append(model)
            else:
                respondent_ids.append(f"{model}|v{v}")
            derives_from.append(model)
            # A prompt variant of a model is nearly the same respondent — the
            # jitter is small on purpose. Pseudo-respondents buy information
            # about items, not about the ability scale.
            theta.append(float(model_theta[m] + (0.0 if v == 0 else rng.normal(0.0, 0.15))))

    item_ids = [f"item_{i:05d}" for i in range(n_items)]

    a = rng.lognormal(mean=np.log(1.05), sigma=0.42, size=n_items)
    n_dead = int(round(dead_fraction * n_items))
    dead_idx = rng.choice(n_items, size=n_dead, replace=False)
    a[dead_idx] = rng.uniform(0.02, 0.22, size=n_dead)

    # Drawn well clear of -DEAD_THRESHOLD so these are unambiguously inverted
    # rather than merely negligible-and-negative, which is `dead`.
    n_inverted = int(round(inverted_fraction * n_items))
    spare = np.setdiff1d(np.arange(n_items), dead_idx)
    inverted_idx = rng.choice(spare, size=min(n_inverted, spare.size), replace=False)
    a[inverted_idx] = -rng.uniform(0.8, 2.2, size=inverted_idx.size)

    b = rng.normal(0.0, 1.05, size=n_items)
    n_off = int(round(off_range_fraction * n_items))
    remaining = np.setdiff1d(np.arange(n_items), np.union1d(dead_idx, inverted_idx))
    off_idx = rng.choice(remaining, size=min(n_off, remaining.size), replace=False)
    b[off_idx] = rng.normal(0.0, 0.35, size=off_idx.size) + np.where(
        rng.random(off_idx.size) < 0.5, -4.2, 4.2
    )

    return SyntheticTruth(
        model_ids=model_ids,
        respondent_ids=respondent_ids,
        derives_from=derives_from,
        item_ids=item_ids,
        theta=np.asarray(theta, dtype=float),
        a=a,
        b=b,
    )


def responses_from_truth(truth: SyntheticTruth, *, missing: float = 0.0, seed: int = 1) -> np.ndarray:
    """Sample a 0/1 response array of shape (respondents, items).

    `missing` drops that fraction of cells to NaN, because real matrices are
    ragged and a fitter that only ever sees a full rectangle has not been
    tested against what it will actually get.
    """
    rng = np.random.default_rng(seed)
    logits = truth.a[None, :] * (truth.theta[:, None] - truth.b[None, :])
    p = 1.0 / (1.0 + np.exp(-logits))
    responses = (rng.random(p.shape) < p).astype(float)
    if missing > 0:
        responses[rng.random(p.shape) < missing] = np.nan
    return responses


def synthetic_records(truth: SyntheticTruth, responses: np.ndarray) -> list[ResponseRecord]:
    """Flatten a response array into the records an adapter would have produced."""
    records: list[ResponseRecord] = []
    for r, respondent in enumerate(truth.respondent_ids):
        model_id = truth.derives_from[r]
        variant = respondent.split("|", 1)[1] if "|" in respondent else None
        for c, item in enumerate(truth.item_ids):
            value = responses[r, c]
            if np.isnan(value):
                continue
            extra: dict[str, object] = {"subject": f"subject-{c % 7}"}
            if variant is not None:
                extra["prompt_variant"] = variant
            records.append(
                ResponseRecord(
                    model_id=model_id, item_id=item, correct=int(value), extra=extra
                )
            )
    return records


def synthetic_matrix(
    *, missing: float = 0.0, seed: int = 0, **truth_kwargs
) -> tuple[ResponseMatrix, SyntheticTruth]:
    """A ResponseMatrix and the parameters it was generated from."""
    from irtcheck.matrix import build_matrix

    truth = make_truth(seed=seed, **truth_kwargs)
    responses = responses_from_truth(truth, missing=missing, seed=seed + 1)
    records = synthetic_records(truth, responses)
    key = ("model_id", "prompt_variant") if "|" in truth.respondent_ids[0] else ("model_id",)
    return build_matrix(records, respondent_key=key), truth


def _posterior_around(
    values: np.ndarray, sd: np.ndarray | float, rng: np.random.Generator
) -> Posterior:
    """A believable normal posterior centred near `values`.

    The mean is displaced from truth by roughly one standard error, because a
    fabricated artifact whose means are exactly the generating parameters would
    let a consumer accidentally depend on precision no real fit provides.
    """
    sd_arr = np.broadcast_to(np.asarray(sd, dtype=float), values.shape).copy()
    mean = values + rng.normal(0.0, sd_arr)
    return Posterior(
        mean=[float(x) for x in mean],
        sd=[float(x) for x in sd_arr],
        hdi_low=[float(x) for x in mean - Z95 * sd_arr],
        hdi_high=[float(x) for x in mean + Z95 * sd_arr],
    )


def synthetic_fit(
    *,
    missing: float = 0.0,
    seed: int = 0,
    precision: float = 1.0,
    **truth_kwargs,
) -> tuple[IrtFit, SyntheticTruth]:
    """A valid IrtFit fabricated from known parameters, with no fitting.

    `precision` scales how much the posterior knows: 1.0 gives interval widths
    of the order a real fit on this many respondents produces, higher values
    tighten them. It is the knob a fixture uses to reach a particular flag —
    note that `dead` needs a *narrow* interval and therefore many respondents.

    Returns the fit and the truth it came from, so a test can assert recovery.
    """
    rng = np.random.default_rng(seed + 977)
    matrix, truth = synthetic_matrix(missing=missing, seed=seed, **truth_kwargs)

    n_resp = matrix.item_response_counts()
    p_correct = matrix.item_p_correct()

    # Standard errors shrink like 1/sqrt(n). The constants are calibrated so a
    # 12-respondent matrix produces the intervals that matter: wide enough that
    # genuinely flat items land in insufficient-data rather than being ranked.
    denom = np.sqrt(np.maximum(n_resp, 1)) * max(precision, 1e-6)
    sd_a = np.clip(0.95 / denom, 0.01, 5.0)
    sd_b = np.clip(1.30 / denom, 0.01, 5.0)
    sd_theta = np.full(truth.n_respondents, 0.9 / np.sqrt(max(matrix.n_items, 1)))

    a_post = _posterior_around(truth.a, sd_a, rng)
    b_post = _posterior_around(truth.b, sd_b, rng)
    theta_post = _posterior_around(truth.theta, sd_theta, rng)

    flags = compute_flags(a_post, b_post, [float(p) for p in p_correct], theta_post.mean)

    fit = IrtFit(
        irtcheck_version=__version__,
        created=utc_now(),
        model={
            "kind": "2pl",
            "priors": "hierarchical",
            "identification": "a > 0, theta ~ N(0, 1)",
            "synthetic": True,
        },
        respondent_key=list(matrix.respondent_key),
        respondent_ids=list(matrix.respondent_ids),
        derives_from=list(matrix.derives_from),
        theta=theta_post,
        item_ids=list(matrix.item_ids),
        a=a_post,
        b=b_post,
        n_resp=[int(x) for x in n_resp],
        p_correct=[float(x) for x in np.nan_to_num(p_correct, nan=0.0)],
        flags=flags,
        diagnostics={
            "source": "synth.synthetic_fit",
            "seed": seed,
            "precision": precision,
            "elbo_final": None,
            "epochs": 0,
        },
        passthrough=matrix.passthrough,
        responses=EmbeddedResponses(
            rows=[int(x) for x in matrix.rows],
            cols=[int(x) for x in matrix.cols],
            obs=[int(x) for x in matrix.obs],
        ),
    )
    fit.validate()
    return fit, truth
