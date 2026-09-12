"""Reconciling the four parameterisations, and measuring agreement afterwards.

This module is the point of the whole directory. Two fits can be *the same
model* and produce parameter vectors that look unrelated, because the 2PL
likelihood

    P(correct) = sigmoid(a_i * (theta_j - b_i))

is invariant under two transformations, and every package resolves them
differently:

  - **reflection**: (a, b, theta) -> (-a, -b, -theta) is the identity. Ours
    resolves it toward positive mean `a`. py-irt draws `a` from a LogNormal and
    so cannot enter the mirrored mode at all. **mirt does not constrain the
    slope** — measured, not assumed: its default 2PL returned negative slopes on
    both datasets here (2 of 59 items on dense, 22 of 183 on sparse), so like us
    it resolves the reflection by where its optimiser starts rather than by
    construction. That makes mirt, not py-irt, the reference whose
    identification convention matches ours, and it is why `reflection_agrees`
    decides on theta rather than on the sign of `a`.
  - **scale and location**: (a, b, theta) -> (a/c, c*b + d, c*theta + d) is
    also the identity. Fixed by pinning the theta metric. Ours and mirt and
    py-irt-with-vague-priors all pin it the same way, theta ~ N(0, 1) as a
    *fixed* prior. py-irt with hierarchical priors does not pin it at all —
    `theta ~ Normal(mu_theta, 1/u_theta)` with both learned — so its output has
    to be standardised before it means anything.

And one difference that is not an invariance but a notation:

  - **mirt's slope-intercept form**. Internally mirt fits
    `P = 1/(1 + exp(-(a1*theta + d)))`. Its `d` is an intercept, not a
    difficulty: `b = -d / a1`. Comparing mirt's `d` against our `b` reports a
    disagreement that does not exist, which is exactly the trap
    docs/spec.md names.

Everything here is deliberately explicit and separately testable. `test_conventions.py`
checks each transformation against a likelihood that must not move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this, an item's difficulty is not a quantity any estimator can pin
# down: `b` enters the likelihood only through `a * (theta - b)`, so as `a`
# approaches zero the data stop constraining `b` and every implementation
# returns whatever its prior or its optimiser's last step happened to leave
# there. Comparing `b` across packages on such an item measures the priors,
# not the fit, so the b comparison states which items it used.
B_IDENTIFIED_MIN_A = 0.35


def slope_intercept_to_difficulty(a1: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mirt's (a1, d) -> our (a, b).

    `a1 * theta + d == a1 * (theta - b)` exactly when `b = -d / a1`, which is
    why this is a notation change and not an approximation. The guard is for an
    item whose slope came back at numerically zero; `b` is meaningless there and
    NaN says so rather than an inf pretending to be a difficulty.
    """
    a1 = np.asarray(a1, dtype=float)
    d = np.asarray(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(np.abs(a1) > 1e-9, -d / a1, np.nan)
    return a1, b


def reflect(
    a: np.ndarray, b: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mirror solution. Identical likelihood, opposite sign on everything."""
    return -np.asarray(a), -np.asarray(b), -np.asarray(theta)


def standardise_theta_metric(
    a: np.ndarray, b: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Put a fit whose theta metric is free onto theta ~ mean 0, sd 1.

    For py-irt's hierarchical priors, whose model learns both the mean and the
    variance of the ability distribution and therefore identifies neither the
    location nor the scale of the latent trait. With `c = sd(theta)` and
    `d = mean(theta)`:

        theta' = (theta - d) / c      b' = (b - d) / c      a' = a * c

    leaves every response probability unchanged, since
    `a'*(theta' - b') == a*(theta - b)`. Returns the transformed triple plus
    the `(c, d)` actually applied, because how far `c` is from 1 is itself a
    finding worth printing.
    """
    theta = np.asarray(theta, dtype=float)
    loc = float(theta.mean())
    scale = float(theta.std(ddof=0))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (
        np.asarray(a, dtype=float) * scale,
        (np.asarray(b, dtype=float) - loc) / scale,
        (theta - loc) / scale,
        scale,
        loc,
    )


def reflection_agrees(theta_ref: np.ndarray, theta_other: np.ndarray) -> bool:
    """Are two fits in the same reflection mode?

    Decided on theta rather than on `a`, because two of the three references
    constrain `a` positive and so agree on its sign by construction — checking
    `a` would be checking nothing. Abilities are signed in every
    implementation, so a negative correlation between them is the reflection
    and a positive one is agreement.
    """
    ref = np.asarray(theta_ref, dtype=float)
    other = np.asarray(theta_other, dtype=float)
    ok = np.isfinite(ref) & np.isfinite(other)
    if ok.sum() < 3:
        return True
    return bool(np.corrcoef(ref[ok], other[ok])[0, 1] >= 0)


@dataclass(slots=True)
class Agreement:
    """How close two vectors are, on the three readings that matter.

    - `pearson` / `spearman`: does the estimator rank items the same way? This
      is what irtcheck actually acts on — `report` ranks by information and
      `select` takes a top slice — so a rank correlation is the headline, not a
      consolation prize.
    - `slope`: the least-squares scale factor between them, through the origin
      for `a` (a scale invariance is multiplicative) and with an intercept for
      `b` (location matters there). A slope of 1.0 means the two are on the
      same metric; anything else is either a convention we failed to reconcile
      or shrinkage, and which one it is is decided by whether the residual
      scatter is small.
    - `max_abs_diff` / `rmse`: the tolerance actually achieved, stated after
      the fact rather than picked in advance.
    """

    n: int
    pearson: float
    spearman: float
    slope: float
    intercept: float
    rmse: float
    max_abs_diff: float
    rmse_after_slope: float
    max_abs_diff_after_slope: float

    def line(self, label: str) -> str:
        return (
            f"{label:<34} n={self.n:<4} r={self.pearson:+.4f} rho={self.spearman:+.4f} "
            f"slope={self.slope:+.3f} rmse={self.rmse:.3f} max|d|={self.max_abs_diff:.3f} "
            f"(after slope: rmse={self.rmse_after_slope:.3f} "
            f"max|d|={self.max_abs_diff_after_slope:.3f})"
        )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without dragging scipy into this module."""
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    return ranks


def agreement(
    reference: np.ndarray, other: np.ndarray, *, through_origin: bool = False
) -> Agreement:
    """Compare two aligned vectors, ignoring positions either one left NaN."""
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    ok = np.isfinite(ref) & np.isfinite(oth)
    ref, oth = ref[ok], oth[ok]
    n = int(ref.size)
    if n < 3:
        nan = float("nan")
        return Agreement(n, nan, nan, nan, nan, nan, nan, nan, nan)

    pearson = float(np.corrcoef(ref, oth)[0, 1])
    spearman = _spearman(ref, oth)

    if through_origin:
        slope = float(ref @ oth / (ref @ ref)) if ref @ ref > 0 else float("nan")
        intercept = 0.0
    else:
        slope, intercept = (float(v) for v in np.polyfit(ref, oth, 1))

    diff = oth - ref
    rescaled = oth - (slope * ref + intercept)
    return Agreement(
        n=n,
        pearson=pearson,
        spearman=spearman,
        slope=slope,
        intercept=intercept,
        rmse=float(np.sqrt(np.mean(diff**2))),
        max_abs_diff=float(np.max(np.abs(diff))),
        rmse_after_slope=float(np.sqrt(np.mean(rescaled**2))),
        max_abs_diff_after_slope=float(np.max(np.abs(rescaled))),
    )


def align(
    reference_ids: list[str], other_ids: list[str], *vectors: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Reorder `vectors` (indexed by `other_ids`) onto `reference_ids`.

    py-irt indexes items by first appearance in its JSON Lines input and mirt by
    CSV column, and neither promises our order. Positions missing from
    `other_ids` come back NaN, which `agreement` then drops — an estimator that
    silently dropped an all-correct column shows up as a smaller `n` rather than
    as a shifted comparison.
    """
    index = {item: i for i, item in enumerate(other_ids)}
    take = np.array([index.get(item, -1) for item in reference_ids])
    out = []
    for vec in vectors:
        vec = np.asarray(vec, dtype=float)
        picked = np.where(take >= 0, vec[np.clip(take, 0, None)], np.nan)
        out.append(picked)
    return tuple(out)


def probabilities(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The (respondents x items) probability surface a parameter set implies.

    The invariance-free way to compare two fits. Parameters can differ by any
    reflection or rescaling and still be the same model; this surface cannot.
    When the reconciled parameters disagree, comparing predicted probabilities
    says whether the two fits are the same model described differently or
    genuinely different models.
    """
    logits = np.asarray(a)[None, :] * (
        np.asarray(theta)[:, None] - np.asarray(b)[None, :]
    )
    return 1.0 / (1.0 + np.exp(-logits))
