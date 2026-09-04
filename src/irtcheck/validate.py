"""Leave-one-model-out: does a small anchor set reproduce the full-suite ranking?

This is the headline number. Selecting items using every model and then
reporting that the subset reproduces the ranking of those same models is
circular and worthless — the anchor set was chosen knowing the answer. So:

  1. hold model k out,
  2. fit the 2PL on what remains,
  3. select an anchor set of size n from *that* fit,
  4. score model k on the anchor items only,
  5. compare its rank to its full-suite ground-truth rank,
  6. repeat for every model; report Spearman and Kendall tau against n.

**Holding out a model means holding out every pseudo-respondent derived from
it.** With --respondent-key model_id,prompt_variant one real model is several
respondents; drop one row and that model's other variants leak its answers into
the fit that chooses the anchor set. The anchor set is then picked with
knowledge of the model it is about to be tested on and the correlation comes
out inflated, with nothing anywhere to catch it. `ResponseMatrix.drop_model()`
uses `derives_from` to do this correctly and is the only way respondents are
removed here — never a hand-rolled filter.

Both scores, per D.2, and the plain one is the headline:

  `anchor_accuracy`  proportion correct over the anchor items. This is what a
                     user actually does with an anchor set once they have it,
                     so it is the number that has to hold up.
  `anchor_theta`     ability re-estimated from the anchor responses with the
                     held-out fit's item parameters held fixed (MAP under the
                     N(0, 1) prior). Reported alongside, because the two
                     diverge exactly when an anchor set skews hard or easy, and
                     that divergence is worth seeing.

The fitter is injected. `leave_one_model_out` takes `fit_fn` and `select_fn`,
so the loop is developed and tested against a fake fitter returning known
parameters, and the wave-2 wiring commit connects the real one by changing
`default_fit_fn` and nothing else. Nothing in this module may import torch: it
has to run on a machine that has never installed it (CLAUDE.md, the lazy torch
boundary), and the fitter reaches it through `fit_fn`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, spearmanr

from irtcheck.artifact import IrtFit
from irtcheck.matrix import ResponseMatrix
from irtcheck.select import select_item_ids

# Fit one 2PL per holdout, not one per (holdout, size): the anchor sets for
# every size in the sweep come out of the same held-out fit, which is legitimate
# because n plays no part in fitting. With 5-15 models the fits are the entire
# cost of this command, and refitting per size would multiply it by len(sizes)
# for nothing.
DEFAULT_SIZES: tuple[int, ...] = (25, 50, 100, 200, 400)

# Rank correlation over two points is either +1 or -1 and means nothing.
MIN_MODELS = 3

# theta ~ N(0, 1), the identification the artifact records.
THETA_PRIOR_SD = 1.0

FitFn = Callable[[ResponseMatrix], IrtFit]
SelectFn = Callable[[IrtFit, int], Sequence[str]]


class ValidateError(ValueError):
    pass


# -- scoring one held-out model ----------------------------------------------


def map_theta(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, *, prior_sd: float = THETA_PRIOR_SD
) -> float:
    """MAP ability for one respondent, item parameters held fixed.

    The log posterior is strictly concave in theta whatever the sign of a — its
    second derivative is -sum(a^2 P (1-P)) - 1/sd^2 — so the score function is
    strictly decreasing and bisection cannot fail to converge or land on a local
    optimum. Newton would be faster and would need a guard for the all-right and
    all-wrong cases that Newton alone diverges on; this is called a few hundred
    times per run, so robustness is worth more than the iterations.
    """
    if y.size == 0:
        return float("nan")

    def score(theta: float) -> float:
        p = expit(a * (theta - b))
        return float(np.sum(a * (y - p)) - theta / prior_sd**2)

    lo, hi = -12.0, 12.0
    if score(lo) <= 0.0:
        return lo
    if score(hi) >= 0.0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(slots=True)
class ModelScore:
    """One held-out model at one anchor size."""

    model_id: str
    truth: float  # full-suite accuracy over ALL items — the ground truth rank
    anchor_accuracy: float  # headline
    anchor_theta: float  # secondary, item parameters held fixed
    n_respondents: int
    n_anchor_responses: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "truth": self.truth,
            "anchor_accuracy": self.anchor_accuracy,
            "anchor_theta": self.anchor_theta,
            "n_respondents": self.n_respondents,
            "n_anchor_responses": self.n_anchor_responses,
        }


def _model_respondents(matrix: ResponseMatrix, model_id: str) -> list[int]:
    return [i for i, src in enumerate(matrix.derives_from) if src == model_id]


def _item_parameters(fit: IrtFit, item_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """a and b for `item_ids`, looked up by id.

    By id and not by position: the fit came from a matrix with one model
    dropped, and nothing promises it ordered or even kept the same items.
    """
    index = {item: i for i, item in enumerate(fit.item_ids)}
    missing = [i for i in item_ids if i not in index]
    if missing:
        raise ValidateError(
            f"the anchor set names {len(missing)} item(s) the fit does not know, "
            f"e.g. {missing[:3]}. select_fn must return ids from the fit it was given."
        )
    a = np.asarray(fit.a.mean, dtype=float)
    b = np.asarray(fit.b.mean, dtype=float)
    picks = [index[i] for i in item_ids]
    return a[picks], b[picks]


def score_held_out(
    matrix: ResponseMatrix,
    model_id: str,
    item_ids: Sequence[str],
    fit: IrtFit,
) -> tuple[float, float, int]:
    """Score a held-out model on the anchor items, both ways.

    Returns (accuracy, theta, responses used). Both scores are computed per
    pseudo-respondent and then averaged over the model's respondents, so they
    aggregate the same way the ground truth does — a model is not allowed to
    weigh more in one number than in the other because it was sampled more
    often.
    """
    wanted = list(dict.fromkeys(item_ids))
    a_all, b_all = _item_parameters(fit, wanted)
    column = {item: c for c, item in enumerate(matrix.item_ids)}
    anchor_cols = np.array([column[i] for i in wanted if i in column], dtype=np.int64)
    # Anchor position, so item parameters stay aligned with the columns kept.
    keep = np.array([k for k, i in enumerate(wanted) if i in column], dtype=np.int64)
    a_all, b_all = a_all[keep], b_all[keep]
    position = {int(c): k for k, c in enumerate(anchor_cols)}

    accuracies: list[float] = []
    thetas: list[float] = []
    used = 0
    for r in _model_respondents(matrix, model_id):
        mask = (matrix.rows == r) & np.isin(matrix.cols, anchor_cols)
        y = matrix.obs[mask].astype(float)
        if y.size == 0:
            continue
        used += int(y.size)
        at = np.array([position[int(c)] for c in matrix.cols[mask]], dtype=np.int64)
        accuracies.append(float(y.mean()))
        thetas.append(map_theta(a_all[at], b_all[at], y))

    if not accuracies:
        return float("nan"), float("nan"), 0
    return float(np.mean(accuracies)), float(np.mean(thetas)), used


def full_suite_accuracy(matrix: ResponseMatrix) -> dict[str, float]:
    """Ground truth: each real model's aggregate accuracy over every item.

    `respondent_accuracy()` per respondent, then averaged over the respondents
    of a model — the same aggregation the anchor score uses, so the comparison
    is between two ways of *choosing items* and not between two ways of
    averaging.
    """
    per_respondent = matrix.respondent_accuracy()
    out: dict[str, float] = {}
    for model_id in sorted(set(matrix.derives_from)):
        rows = _model_respondents(matrix, model_id)
        values = [float(per_respondent[r]) for r in rows if not np.isnan(per_respondent[r])]
        out[model_id] = float(np.mean(values)) if values else float("nan")
    return out


# -- the sweep ----------------------------------------------------------------


def _correlate(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """(Spearman, Kendall tau), NaN when the inputs cannot support either."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < MIN_MODELS or np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return float("nan"), float("nan")
    rho = float(spearmanr(xs, ys).statistic)
    tau = float(kendalltau(xs, ys).statistic)
    return rho, tau


@dataclass(slots=True)
class SizeResult:
    """The whole leave-one-model-out loop, at one anchor size."""

    size: int
    models: list[ModelScore]
    n_selected: list[int]  # items actually selected per holdout; short means short
    spearman: float = float("nan")  # headline: plain accuracy over the anchor items
    kendall: float = float("nan")
    spearman_theta: float = float("nan")  # secondary: re-estimated ability
    kendall_theta: float = float("nan")

    def __post_init__(self) -> None:
        truth = [m.truth for m in self.models]
        self.spearman, self.kendall = _correlate(truth, [m.anchor_accuracy for m in self.models])
        self.spearman_theta, self.kendall_theta = _correlate(
            truth, [m.anchor_theta for m in self.models]
        )

    @property
    def min_selected(self) -> int:
        return min(self.n_selected) if self.n_selected else 0

    @property
    def short(self) -> bool:
        """True when some holdout could not supply a full anchor set."""
        return self.min_selected < self.size

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "items_selected": self.min_selected,
            "spearman": self.spearman,
            "kendall": self.kendall,
            "spearman_theta": self.spearman_theta,
            "kendall_theta": self.kendall_theta,
            "models": [m.to_dict() for m in self.models],
        }


@dataclass(slots=True)
class ValidationReport:
    results: list[SizeResult]
    model_ids: list[str]
    n_respondents: int
    n_items: int
    respondent_key: list[str] = field(default_factory=list)

    @property
    def n_models(self) -> int:
        return len(self.model_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "n_respondents": self.n_respondents,
            "n_items": self.n_items,
            "respondent_key": list(self.respondent_key),
            "models": list(self.model_ids),
            "headline": "kendall",  # plain accuracy over the anchor items (D.2)
            "sizes": [r.to_dict() for r in self.results],
        }


def parse_sizes(spec: str | Sequence[int]) -> tuple[int, ...]:
    """Read a --sizes value like "25,50,100". Sorted, deduplicated, positive."""
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        try:
            values = [int(p) for p in parts]
        except ValueError as exc:
            raise ValidateError(f"--sizes must be comma-separated integers: {exc}") from exc
    else:
        values = list(spec)
    if not values:
        raise ValidateError("--sizes is empty; give at least one anchor size, e.g. '25,100'")
    if any(v <= 0 for v in values):
        raise ValidateError(f"anchor sizes must be positive, got {sorted(values)}")
    return tuple(sorted(set(values)))


def leave_one_model_out(
    matrix: ResponseMatrix,
    *,
    fit_fn: FitFn,
    sizes: Sequence[int] = DEFAULT_SIZES,
    select_fn: SelectFn = select_item_ids,
    on_model: Callable[[str, int, int], None] | None = None,
) -> ValidationReport:
    """Run the holdout loop and report rank correlation as a function of n.

    `fit_fn` takes a matrix and returns an IrtFit; `select_fn` takes a fit and a
    size and returns item ids. Both are injected so the loop is testable against
    a fitter that returns known parameters — the whole of this module is
    exercised before a real fitter exists, and the wave-2 wiring commit swaps
    one callable.

    `select_fn` is called once per (holdout, size). Greedy selection is nested,
    so a caller with an expensive selector can pass one that caches per fit;
    with the default selector the fits dominate by orders of magnitude.
    """
    wanted = parse_sizes(sizes)
    model_ids = sorted(set(matrix.derives_from))
    if len(model_ids) < MIN_MODELS:
        raise ValidateError(
            f"leave-one-model-out needs at least {MIN_MODELS} real models, this matrix has "
            f"{len(model_ids)} ({', '.join(model_ids)}). Rank correlation over two models is "
            "either +1 or -1 and says nothing. Note that pseudo-respondents do not help here: "
            "they raise respondent count for fitting, not the number of models to rank."
        )

    truth = full_suite_accuracy(matrix)
    scores: dict[int, list[ModelScore]] = {n: [] for n in wanted}
    selected: dict[int, list[int]] = {n: [] for n in wanted}

    for position, model_id in enumerate(model_ids):
        if on_model is not None:
            on_model(model_id, position, len(model_ids))
        # The one correct way to hold a model out: every pseudo-respondent
        # derived from it goes too. Never filter respondent_ids by hand.
        held_out = matrix.drop_model(model_id)
        fit = fit_fn(held_out)
        for n in wanted:
            item_ids = list(select_fn(fit, n))
            accuracy, theta, used = score_held_out(matrix, model_id, item_ids, fit)
            selected[n].append(len(item_ids))
            scores[n].append(
                ModelScore(
                    model_id=model_id,
                    truth=truth[model_id],
                    anchor_accuracy=accuracy,
                    anchor_theta=theta,
                    n_respondents=len(_model_respondents(matrix, model_id)),
                    n_anchor_responses=used,
                )
            )

    return ValidationReport(
        results=[SizeResult(size=n, models=scores[n], n_selected=selected[n]) for n in wanted],
        model_ids=model_ids,
        n_respondents=matrix.n_respondents,
        n_items=matrix.n_items,
        respondent_key=list(matrix.respondent_key),
    )


def default_fit_fn(*, seed: int = 0, epochs: int = 2000, device: str = "cpu") -> FitFn:
    """The real fitter, imported lazily.

    Inside a function body on purpose. This module is read by a command that
    must import on a machine with no torch, and the import here is the single
    seam the wave-2 wiring commit touches.
    """

    def fit_fn(matrix: ResponseMatrix) -> IrtFit:
        try:
            from irtcheck.fit.fitter import fit_2pl
        except ImportError as exc:  # pragma: no cover - until wave1/fit lands
            raise ValidateError(
                "`irtcheck validate` refits the 2PL once per held-out model, and the "
                f"fitter is not available: {exc}. It is wired up by the wave-2 commit "
                "(see docs/build-plan.html). The holdout loop itself is complete and "
                "testable today by passing your own fit_fn to "
                "irtcheck.validate.leave_one_model_out."
            ) from exc

        return fit_2pl(matrix, seed=seed, epochs=epochs, device=device)

    return fit_fn
