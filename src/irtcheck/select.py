"""Anchor-set selection: the smallest set of items that still measures.

Item information for a 2PL is

    I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))

which peaks at theta = b_i and falls away fast on either side. A test's
information is the sum of its items' information, pointwise in theta.

**Where the integral is taken is the whole argument of this tool.** Integrating
information against a fixed grid — the textbook default, usually N(0, 1) —
answers "is this a good test?" in the abstract. Integrating it against the
posterior theta means of the respondents *in this matrix* answers "does this
test measure the models I actually have?", and those two answers differ
whenever a suite was built for a generation of models that has since been
outgrown. An item that discriminates beautifully at theta = 4 when every model
sits between -1 and 1 carries no information about anything the user owns, and
surfacing that is the point.

**The default objective is summed test information, as the spec says**, and
that default survived a challenge to it, which is worth recording because the
argument against it is a good one.

Test information is additive over items, so `max-information` is linear in the
set, greedy forward selection on it is exactly `sort by score, take the top n`,
and it stacks the whole anchor set wherever the ability density is highest —
nothing in a linear objective ever says "we have enough precision here". The
obvious fix is `min-variance`, which minimises

    sum_j w_j / (1 + I_S(theta_j))

the mean posterior variance of the respondents' abilities under the N(0, 1)
prior the fit identifies theta with. It is concave in accumulated information,
so each item is scored against what the set already covers, and greedy is a
real algorithm rather than a sort.

`min-variance` does exactly what it claims, and it does not help. On its own
metric it wins comfortably — at n=10 it holds the *worst-served* model to an
ability standard error of 0.49 against 0.93, and at n=25 to 0.39 against 0.70.
On rank recovery, which is what `validate` measures and what the tool actually
claims, the two are indistinguishable: over 20 seeds at n=25 min-variance wins
7 and loses 11, at n=50 wins 2 and loses 10, and mean Kendall tau differs by
less than 0.03 — under one swapped pair out of 28. See
`tests/test_select_objectives.py`, which is the experiment, runnable.

The reason is worth knowing before someone re-derives it: spreading information
buys precision for the models at the edges of the ability range, and those are
the models whose *rank* was never in doubt. The pairs that actually swap sit in
the middle, bunched together, which is where a linear objective was already
putting its items. So the default stays the spec's, and `min-variance` stays
reachable for the case the measurement does not cover — reporting a per-model
ability with an honest standard error, rather than a ranking.

**Pseudo-respondents are weighted per real model.** With `--respondent-key
model_id,prompt_variant` one model can be ten respondents, and ten atoms at one
theta would drag the whole anchor set toward that one model's ability. Each
real model gets equal total weight, matching the rule that a report header
counts models and not respondents. This one is not a deviation from anything —
the spec says "the observed ability distribution", and twelve temperature
samples of one model are not twelve observations of the distribution.

**A short anchor set is padded, and that reverses an earlier decision on
measured grounds.** When fewer items are eligible than were asked for, the set
used to come back short, on the argument that filling it with items the report
said it could not read would undo the refusal. Measured against synthetic
ground truth (docs/validation.md §4), the short set was worse than the padded
one and worse than a random draw: at 800 items and eight models, 4.3 items
ranked 2,000 fresh models at Spearman 0.594 where a random fifty managed 0.918.
A rank correlation over four items is mostly ties; `insufficient-data` means
"we cannot tell how sharply this item separates models", not "this item
separates nobody", and most such items separate somebody.

So the confident items come first and the rest of the set is filled, by the
same objective, from items whose interval merely spans zero — never from
`inverted`, `ceiling`, `floor` or unanswered items, and never from an item whose
fitted slope points backwards. That last rule is the one that matters: filling
from every non-ceiling/floor item put 176 mis-keyed items into 198 short sets,
excluding `inverted` alone still put in 132, and excluding a negative point
estimate as well put in 26 and ranked better by 9 thousandths (±1). The
`insufficient-data` gate still earns its place wherever it can fill the set;
padding only decides what happens when it cannot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

from irtcheck.artifact import FLAG_CEILING, FLAG_FLOOR, FLAG_INVERTED, IrtFit

# theta ~ N(0, 1) is the identification the fit records, so an anchor set that
# tells us nothing still leaves precision 1. It also keeps the objective finite
# for the empty set, which is where greedy starts.
THETA_PRIOR_PRECISION = 1.0

OBJECTIVE_MIN_VARIANCE = "min-variance"
OBJECTIVE_MAX_INFORMATION = "max-information"
OBJECTIVES = (OBJECTIVE_MAX_INFORMATION, OBJECTIVE_MIN_VARIANCE)
# The spec's criterion, kept after measuring the alternative against it — see
# the module docstring and tests/test_select_objectives.py.
DEFAULT_OBJECTIVE = OBJECTIVE_MAX_INFORMATION


class SelectError(ValueError):
    pass


def item_information(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    theta: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Fisher information of each item at each ability, shape (items, thetas)."""
    a_col = np.asarray(a, dtype=float)[:, None]
    b_col = np.asarray(b, dtype=float)[:, None]
    theta_row = np.asarray(theta, dtype=float)[None, :]
    p = expit(a_col * (theta_row - b_col))
    return a_col**2 * p * (1.0 - p)


def ability_distribution(fit: IrtFit) -> tuple[np.ndarray, np.ndarray]:
    """Where the respondents sit, and how much each point counts.

    Points are posterior theta means — one per respondent, not a grid. Weights
    divide each real model's share equally among the pseudo-respondents derived
    from it, so a model sampled at ten temperatures does not outvote a model
    run once. Weights sum to 1.
    """
    theta = np.asarray(fit.theta.mean, dtype=float)
    counts: dict[str, int] = {}
    for source in fit.derives_from:
        counts[source] = counts.get(source, 0) + 1
    weights = np.array([1.0 / counts[src] for src in fit.derives_from], dtype=float)
    total = weights.sum()
    if total <= 0:  # pragma: no cover - IrtFit.validate() rejects an empty fit
        raise SelectError("this fit has no respondents to select against")
    return theta, weights / total


@dataclass(slots=True)
class AnchorSet:
    """A fixed anchor set, plus enough context to argue with it.

    `gains` is what each item bought at the moment it was added, in the units of
    whichever objective chose it: weighted test information under
    `max-information`, the drop in mean posterior variance under `min-variance`.
    Either way it decays, and watching where it flattens is how a user picks n
    rather than guessing — the item whose gain is 1e-4 is padding.
    """

    item_ids: list[str]
    item_indices: list[int]
    gains: list[float]
    theta: list[float]
    theta_weights: list[float]
    information: list[float]  # test information of the chosen set, per theta point
    theta_se: list[float]  # posterior sd of theta under that information
    objective: str
    requested: int
    n_usable: int  # candidates after flags *and* after dropping zero-response items
    n_items: int
    # How many of `item_ids`, counted from the end, came from the padding pool
    # rather than the confident one. The confident items always come first.
    n_padded: int = 0

    @property
    def n_selected(self) -> int:
        return len(self.item_ids)

    @property
    def n_confident(self) -> int:
        """Items chosen from the confident pool, before any padding."""
        return self.n_selected - self.n_padded

    @property
    def shortfall(self) -> int:
        """How many fewer items than asked for. Non-zero is a finding, not a bug."""
        return max(self.requested - self.n_selected, 0)

    @property
    def mean_theta_se(self) -> float:
        """Ability standard error this anchor set achieves, averaged over models."""
        return float(np.dot(self.theta_weights, self.theta_se))

    def to_dict(self) -> dict[str, Any]:
        # item_ids leads: the file is an anchor set first and provenance second,
        # and a consumer reaching for `["item_ids"]` should not have to scroll.
        return {
            "item_ids": list(self.item_ids),
            "count": self.n_selected,
            "requested": self.requested,
            "objective": self.objective,
            "n_usable_items": self.n_usable,
            "padded": self.n_padded,
            "n_items": self.n_items,
            "mean_theta_se": self.mean_theta_se,
            "gains": [float(g) for g in self.gains],
        }


def select_anchor(
    fit: IrtFit,
    n: int,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    candidates: Sequence[int] | None = None,
) -> AnchorSet:
    """Greedy forward selection of `n` items, maximising measurement precision.

    Candidates default to `IrtFit.usable_items()`: insufficient-data items are
    excluded because we cannot tell what they measure, and ceiling/floor items
    because they separate nobody whatever the posterior says. Dead items stay
    eligible and are simply never worth picking — their information is near
    zero by definition — which is a nice property to have fall out rather than
    to special-case.

    **Items with no responses in this fit are excluded here, not inherited from
    flag semantics.** An item nobody answered carries no information about these
    respondents by construction, whatever its posterior happens to say. Today a
    zero-response item also lands as `floor` — `p_correct` comes through as 0.0
    — so the flag path happens to exclude it too, but that is a bug being fixed
    rather than a guarantee: "everyone got it wrong" is not a claim anyone can
    make about an item nobody answered. Once it is fixed the only thing keeping
    such an item out of an anchor set would be its `a` interval spanning zero,
    which is a property of a real fit and not of every fit. Leave-one-model-out
    manufactures this case routinely: a ragged matrix where only the held-out
    model answered an item leaves that item with zero responses in the held-out
    fit that then chooses the anchor set.

    **When fewer than `n` are eligible, the set is padded** from
    `padding_items()`, and the confident items come first. See the module
    docstring for the measurement that decided it. Padding applies only to the
    default pool: a caller passing `candidates` has named the whole pool. Fewer
    than `n` items still come back when both pools together run out.
    """
    if n <= 0:
        raise SelectError(f"anchor set size must be positive, got {n}")
    if objective not in OBJECTIVES:
        raise SelectError(f"unknown objective {objective!r}; choose one of {', '.join(OBJECTIVES)}")

    pool = list(fit.usable_items()) if candidates is None else list(dict.fromkeys(candidates))
    bad = [i for i in pool if not 0 <= i < fit.n_items]
    if bad:
        raise SelectError(f"candidate item index out of range: {bad[:3]}")

    answered = [i for i in pool if fit.n_resp[i] > 0]
    unanswered = len(pool) - len(answered)
    pool = answered
    padding = padding_items(fit, exclude=pool) if candidates is None else []

    if not pool and not padding:
        if unanswered:
            raise SelectError(
                f"none of the {unanswered} candidate item(s) have a single response in "
                "this fit, so there is nothing to select from. An item nobody answered "
                "carries no information about these respondents whatever its posterior "
                "says."
            )
        raise SelectError(
            "no items are eligible for selection — every item is flagged "
            "inverted, ceiling or floor, discriminates backwards, or has no responses. "
            "With this many respondents most items cannot be read at all; add respondents "
            "(temperature samples or prompt variants via --respondent-key) before "
            "selecting an anchor set."
        )

    theta, weights = ability_distribution(fit)
    precision = np.full(theta.shape, THETA_PRIOR_PRECISION, dtype=float)
    chosen, gains, precision, test_info = _greedy(
        fit, pool, n, objective, weights, theta, precision
    )
    n_confident = len(chosen)
    if len(chosen) < n and padding:
        extra, extra_gains, precision, extra_info = _greedy(
            fit, padding, n - len(chosen), objective, weights, theta, precision
        )
        chosen += extra
        gains += extra_gains
        test_info = test_info + extra_info

    return AnchorSet(
        item_ids=[fit.item_ids[i] for i in chosen],
        item_indices=chosen,
        gains=gains,
        theta=[float(t) for t in theta],
        theta_weights=[float(w) for w in weights],
        information=[float(x) for x in test_info],
        theta_se=[float(x) for x in np.sqrt(1.0 / precision)],
        objective=objective,
        requested=n,
        n_usable=len(pool),
        n_items=fit.n_items,
        n_padded=len(chosen) - n_confident,
    )


def padding_items(fit: IrtFit, *, exclude: Sequence[int] = ()) -> list[int]:
    """Items a short anchor set may be filled from, in index order.

    Everything that is answered, not `ceiling` or `floor` (nothing to tell), not
    `inverted` (confidently backwards), and whose fitted slope is positive — an
    `insufficient-data` item leaning backwards is a mis-keyed item the interval
    could not yet convict, and information goes as `a^2`, so without this rule
    it is exactly the item selection would reach for.
    """
    skip = {FLAG_CEILING, FLAG_FLOOR, FLAG_INVERTED}
    excluded = set(exclude)
    a = np.asarray(fit.a.mean, dtype=float)
    return [
        i
        for i, flags in enumerate(fit.flags)
        if i not in excluded
        and fit.n_resp[i] > 0
        and a[i] > 0
        and not skip.intersection(flags)
    ]


def _greedy(
    fit: IrtFit,
    pool: list[int],
    n: int,
    objective: str,
    weights: np.ndarray,
    theta: np.ndarray,
    precision: np.ndarray,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    """Greedy forward selection of up to `n` items from `pool`.

    `precision` is carried in and out so padding continues the confident set
    rather than starting over — which matters for `min-variance`, where an
    item's gain depends on what the set already covers. Returns item indices
    into the fit, their gains, the updated precision, and the chosen items'
    test information.
    """
    if not pool:
        return [], [], precision, np.zeros_like(theta)
    a = np.asarray(fit.a.mean, dtype=float)[pool]
    b = np.asarray(fit.b.mean, dtype=float)[pool]
    info = item_information(a, b, theta)  # (pool, thetas)

    chosen: list[int] = []
    gains: list[float] = []
    available = np.ones(len(pool), dtype=bool)

    for _ in range(min(n, len(pool))):
        if objective == OBJECTIVE_MAX_INFORMATION:
            # Linear in the set, so the loop reduces to a weighted sort and the
            # gain vector never changes. Recomputing it each pass costs one
            # matrix-vector product on a few thousand items; keeping the two
            # objectives in one loop is worth more than saving it.
            gain = info @ weights
        else:
            current = weights / precision
            gain = current.sum() - (weights / (precision + info)).sum(axis=1)
        gain = np.where(available, gain, -np.inf)
        pick = int(np.argmax(gain))  # ties break toward the lower item index
        if not np.isfinite(gain[pick]):  # pragma: no cover - the range bounds the loop
            break
        available[pick] = False
        chosen.append(pick)
        gains.append(float(gain[pick]))
        precision = precision + info[pick]

    test_info = info[chosen].sum(axis=0) if chosen else np.zeros_like(theta)
    return [pool[i] for i in chosen], gains, precision, test_info


def select_item_ids(fit: IrtFit, n: int) -> list[str]:
    """`select_anchor` reduced to the item ids — the `select_fn` validate injects."""
    return select_anchor(fit, n).item_ids
