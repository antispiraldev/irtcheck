"""Leave-one-model-out: does an anchor set place a model it never saw?

This is the headline number. Selecting items using every model and then
reporting that the subset reproduces the ranking of those same models is
circular and worthless — the anchor set was chosen knowing the answer. So:

  1. hold model k out,
  2. fit the 2PL on what remains,
  3. select an anchor set S_k of size n from *that* fit,
  4. score **every** model on S_k by plain accuracy,
  5. find where k lands among them, and how far that is from where it lands on
     the full suite,
  6. repeat for every model; report the mean distance against n, beside the
     same distance for random item sets of the same size.

**Every model is scored on the same set, and that is a correction.** Before
this protocol, step 4 scored only model k, each on its own S_k, and the
headline was a rank correlation across those scores. Different holdouts choose
different sets — on the twelve-model HELM matrix, the twelve 100-item sets
shared 11 items and their mean accuracy ranged 0.49-0.77 — so that correlation
mixed "does the set rank models" with "how hard did this holdout's set happen
to be". It flattered stratified selection (+0.991 at n=400) and punished plain
selection for reasons unrelated to ranking. A user scores every model on one
fixed set; this is that. The re-estimated-ability columns existed to correct
for sets differing in difficulty between holdouts, and went with the problem.

**The random baseline is not optional, and it is the most important column.**
Measured on HELM Lite (docs/validation.md §5), anchor sets chosen by the 2PL
placed held-out models *worse* than random sets of the same size from 100
items up, at twelve models and at ninety-five — and item-rest correlation did
no better. On synthetic data, where the truth is known, the same happened
below about fifty models and reversed above it: the fit's item estimates, not
the selection, were the limit (§5, studies/true_ability/). A number with
nothing to compare it to would have hidden all of that. The draws are seeded,
so the same artifact and settings give the same output.

**Holding out a model means holding out every pseudo-respondent derived from
it.** With --respondent-key model_id,prompt_variant one real model is several
respondents; drop one row and that model's other variants leak its answers into
the fit that chooses the anchor set. The anchor set is then picked with
knowledge of the model it is about to be tested on and the result comes out
inflated, with nothing anywhere to catch it. `ResponseMatrix.drop_model()` uses
`derives_from` to do this correctly and is the only way respondents are removed
here — never a hand-rolled filter.

The fitter is injected. `leave_one_model_out` takes `fit_fn` and `select_fn`,
so the loop is developed and tested against a fake fitter returning known
parameters. Nothing in this module may import torch: it has to run on a machine
that has never installed it (CLAUDE.md, the lazy torch boundary), and the
fitter reaches it through `fit_fn`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

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

# Random sets per (holdout, size). Scoring a set is a column slice, so this is
# cheap next to one fit; 200 puts the share-of-draws column within a few points.
RANDOM_DRAWS = 200

FitFn = Callable[[ResponseMatrix], IrtFit]
SelectFn = Callable[[IrtFit, int], Sequence[str]]


class ValidateError(ValueError):
    pass


# -- scoring every model on one set -------------------------------------------


def _model_respondents(matrix: ResponseMatrix, model_id: str) -> list[int]:
    return [i for i, src in enumerate(matrix.derives_from) if src == model_id]


class AccuracyTable:
    """Every model's plain accuracy on any item set, from one dense pass.

    Accuracy is computed per respondent and then averaged over a model's
    respondents — the aggregation `full_suite_accuracy` uses — so a model
    sampled at ten temperatures does not weigh more in one number than in the
    other. A respondent that answered none of the set is left out of its
    model's mean; a model with no such respondents scores NaN.
    """

    def __init__(self, matrix: ResponseMatrix) -> None:
        self.model_ids = sorted(set(matrix.derives_from))
        self.column = {item: c for c, item in enumerate(matrix.item_ids)}
        shape = (matrix.n_respondents, matrix.n_items)
        self._correct = np.zeros(shape, dtype=np.float32)
        self._answered = np.zeros(shape, dtype=np.float32)
        # add.at, not assignment: a matrix may carry a respondent answering an
        # item twice, and respondent_accuracy() counts both.
        np.add.at(self._correct, (matrix.rows, matrix.cols), matrix.obs)
        np.add.at(self._answered, (matrix.rows, matrix.cols), 1.0)
        owner = {m: k for k, m in enumerate(self.model_ids)}
        self._owner = np.array([owner[src] for src in matrix.derives_from], dtype=np.int64)
        self._per_model = np.bincount(self._owner, minlength=len(self.model_ids))

    def columns(self, item_ids: Sequence[str]) -> np.ndarray:
        return np.array([self.column[i] for i in dict.fromkeys(item_ids)], dtype=np.int64)

    def accuracy(self, cols: np.ndarray) -> np.ndarray:
        """Per-model accuracy on `cols`, in `model_ids` order."""
        answered = self._answered[:, cols].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_respondent = self._correct[:, cols].sum(axis=1) / answered
        ok = answered > 0
        models = len(self._per_model)
        total = np.bincount(self._owner[ok], weights=per_respondent[ok], minlength=models)
        count = np.bincount(self._owner[ok], minlength=models)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def placement(scores: np.ndarray, truth: np.ndarray, k: int) -> tuple[float, float]:
    """(where model k lands on `scores`, where it lands on `truth`), 1 = best.

    Both among the same models: those with a finite value in each, so a model
    that answered none of the set drops out of both rankings rather than
    shifting one. Ties share the average place. NaN if k itself has no score.
    """
    ok = np.isfinite(scores) & np.isfinite(truth)
    if not ok[k]:
        return float("nan"), float("nan")
    at = int(np.flatnonzero(ok).tolist().index(k))
    return (
        float(rankdata(-scores[ok], method="average")[at]),
        float(rankdata(-truth[ok], method="average")[at]),
    )


def full_suite_accuracy(matrix: ResponseMatrix) -> dict[str, float]:
    """Ground truth: each real model's aggregate accuracy over every item.

    `respondent_accuracy()` per respondent, then averaged over the respondents
    of a model — the same aggregation `AccuracyTable` uses, so the comparison
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
class ModelScore:
    """One held-out model at one anchor size."""

    model_id: str
    truth: float  # full-suite accuracy over ALL items
    true_place: float  # 1 = best, among the models scored on this set
    anchor_accuracy: float
    anchor_place: float
    n_items: int  # the anchor set's size for this holdout
    n_respondents: int
    random_error: float  # mean |place error| over random sets of n_items

    @property
    def error(self) -> float:
        return abs(self.anchor_place - self.true_place)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "truth": self.truth,
            "true_place": self.true_place,
            "anchor_accuracy": self.anchor_accuracy,
            "anchor_place": self.anchor_place,
            "place_error": self.error,
            "random_place_error": self.random_error,
            "n_items": self.n_items,
            "n_respondents": self.n_respondents,
        }


@dataclass(slots=True)
class SizeResult:
    """The whole leave-one-model-out loop, at one anchor size."""

    size: int
    models: list[ModelScore]
    random_errors: list[float]  # mean place error of each random draw, over holdouts
    random_spearman: float = float("nan")  # mean over draws
    spearman: float = float("nan")  # held-out places against full-suite places
    kendall: float = float("nan")

    def __post_init__(self) -> None:
        self.spearman, self.kendall = _correlate(
            [m.true_place for m in self.models], [m.anchor_place for m in self.models]
        )

    @property
    def n_selected(self) -> list[int]:
        return [m.n_items for m in self.models]

    @property
    def min_selected(self) -> int:
        return min(self.n_selected) if self.models else 0

    @property
    def short(self) -> bool:
        """True when some holdout could not supply a full anchor set."""
        return self.min_selected < self.size

    @property
    def mean_error(self) -> float:
        """Headline: mean places between where a held-out model lands and where it should."""
        errors = [m.error for m in self.models if np.isfinite(m.error)]
        return float(np.mean(errors)) if errors else float("nan")

    @property
    def random_error(self) -> float:
        return float(np.mean(self.random_errors)) if self.random_errors else float("nan")

    @property
    def beats_random(self) -> float:
        """Share of random draws whose mean error is larger than the anchor set's.

        Ties count half, so a size where every set places every model exactly
        reads 50% rather than 0% or 100%.
        """
        if not self.random_errors or not np.isfinite(self.mean_error):
            return float("nan")
        draws = np.asarray(self.random_errors)
        return float(np.mean(draws > self.mean_error) + 0.5 * np.mean(draws == self.mean_error))

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "items_selected": self.min_selected,
            "place_error": self.mean_error,
            "random_place_error": self.random_error,
            "beats_random": self.beats_random,
            "spearman": self.spearman,
            "kendall": self.kendall,
            "random_spearman": self.random_spearman,
            "models": [m.to_dict() for m in self.models],
        }


@dataclass(slots=True)
class ValidationReport:
    results: list[SizeResult]
    model_ids: list[str]  # the models held out
    n_models: int  # every real model, each scored on every set
    n_respondents: int
    n_items: int
    random_draws: int = RANDOM_DRAWS
    respondent_key: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "n_respondents": self.n_respondents,
            "n_items": self.n_items,
            "respondent_key": list(self.respondent_key),
            "held_out": list(self.model_ids),
            "random_draws": self.random_draws,
            "headline": "place_error",  # against random_place_error, same size
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
    holdouts: Sequence[str] | None = None,
    random_draws: int = RANDOM_DRAWS,
    seed: int = 0,
) -> ValidationReport:
    """Run the holdout loop and report placement error as a function of n.

    `fit_fn` takes a matrix and returns an IrtFit; `select_fn` takes a fit and a
    size and returns item ids from that fit. Both are injected so the loop is
    testable against a fitter that returns known parameters, and so a study can
    score another selector through exactly the same loop.

    `holdouts` restricts which models are held out — every model is still
    scored on every set. With ninety-five models, holding out all of them is
    ninety-five fits; a spread of two dozen answers the same question.

    Random sets are drawn from the items the held-out fit has responses for,
    which is what a user without model k could have drawn from, at the size the
    anchor set actually came out, so a short set is compared with a short one.
    """
    wanted = parse_sizes(sizes)
    model_ids = sorted(set(matrix.derives_from))
    if len(model_ids) < MIN_MODELS:
        raise ValidateError(
            f"leave-one-model-out needs at least {MIN_MODELS} real models, this matrix has "
            f"{len(model_ids)} ({', '.join(model_ids)}). Placing one model among two says "
            "nothing. Note that pseudo-respondents do not help here: they raise respondent "
            "count for fitting, not the number of models to rank."
        )
    held = model_ids if holdouts is None else list(dict.fromkeys(holdouts))
    unknown = [h for h in held if h not in set(model_ids)]
    if unknown:
        raise ValidateError(f"cannot hold out models the matrix does not have: {unknown[:3]}")
    if random_draws < 0:
        raise ValidateError(f"random_draws must be zero or more, got {random_draws}")

    table = AccuracyTable(matrix)
    truth_of = full_suite_accuracy(matrix)
    truth = np.array([truth_of[m] for m in table.model_ids], dtype=float)
    index = {m: k for k, m in enumerate(table.model_ids)}
    rng = np.random.default_rng(seed)

    scores: dict[int, list[ModelScore]] = {n: [] for n in wanted}
    # draw_errors[n][d] accumulates draw d's place error over holdouts.
    draw_errors = {n: np.zeros(random_draws) for n in wanted}
    draw_counts = {n: np.zeros(random_draws) for n in wanted}
    draw_places = {n: [[] for _ in range(random_draws)] for n in wanted}

    for position, model_id in enumerate(held):
        if on_model is not None:
            on_model(model_id, position, len(held))
        # The one correct way to hold a model out: every pseudo-respondent
        # derived from it goes too. Never filter respondent_ids by hand.
        fit = fit_fn(matrix.drop_model(model_id))
        known = set(fit.item_ids)
        answered = [
            table.column[i]
            for i, count in zip(fit.item_ids, fit.n_resp, strict=True)
            if count > 0 and i in table.column
        ]
        k = index[model_id]
        for n in wanted:
            item_ids = list(select_fn(fit, n))
            missing = [i for i in item_ids if i not in known]
            if missing:
                raise ValidateError(
                    f"the anchor set names {len(missing)} item(s) the fit does not know, "
                    f"e.g. {missing[:3]}. select_fn must return ids from the fit it was given."
                )
            cols = table.columns(item_ids)
            scored = table.accuracy(cols)
            anchor_place, true_place = placement(scored, truth, k)

            errors = []
            for d in range(random_draws):
                size = min(len(cols), len(answered))
                pick = rng.choice(answered, size=size, replace=False) if size else cols[:0]
                place, true = placement(table.accuracy(np.asarray(pick)), truth, k)
                error = abs(place - true)
                errors.append(error)
                if np.isfinite(error):
                    draw_errors[n][d] += error
                    draw_counts[n][d] += 1
                    draw_places[n][d].append((true, place))
            finite = [e for e in errors if np.isfinite(e)]
            scores[n].append(
                ModelScore(
                    model_id=model_id,
                    truth=truth_of[model_id],
                    true_place=true_place,
                    anchor_accuracy=float(scored[k]),
                    anchor_place=anchor_place,
                    n_items=len(cols),
                    n_respondents=len(_model_respondents(matrix, model_id)),
                    random_error=float(np.mean(finite)) if finite else float("nan"),
                )
            )

    results = []
    for n in wanted:
        with np.errstate(invalid="ignore", divide="ignore"):
            per_draw = draw_errors[n] / draw_counts[n]
        rhos = [
            _correlate([t for t, _ in places], [p for _, p in places])[0]
            for places in draw_places[n]
            if len(places) >= MIN_MODELS
        ]
        finite_rhos = [r for r in rhos if np.isfinite(r)]
        results.append(
            SizeResult(
                size=n,
                models=scores[n],
                random_errors=[float(e) for e in per_draw if np.isfinite(e)],
                random_spearman=float(np.mean(finite_rhos)) if finite_rhos else float("nan"),
            )
        )

    return ValidationReport(
        results=results,
        model_ids=list(held),
        n_models=len(model_ids),
        n_respondents=matrix.n_respondents,
        n_items=matrix.n_items,
        random_draws=random_draws,
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
