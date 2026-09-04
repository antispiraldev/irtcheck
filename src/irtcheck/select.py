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

Two choices here that the spec left open, both recorded because they change
which items come out:

**The objective is average posterior variance of theta, not summed
information.** Test information is additive over items, so "pick the set
maximising total information" is a linear objective, and greedy forward
selection on a linear objective is exactly `sort by score, take the top n`. It
also stacks the whole anchor set on top of whichever theta happens to be
densest, because nothing in it ever says "we have enough precision here". The
default objective instead minimises

    sum_j w_j / (1 + I_S(theta_j))

the mean posterior variance of the respondents' abilities under the N(0, 1)
prior the fit identifies theta with. It is concave in accumulated information,
so each item is scored against what the set already covers, greedy forward
selection is a real algorithm rather than a sort, and the resulting set spreads
to cover the ability range where models actually sit. `max-information` keeps
the linear behaviour for anyone who wants it.

**Pseudo-respondents are weighted per real model.** With `--respondent-key
model_id,prompt_variant` one model can be ten respondents, and ten atoms at one
theta would drag the whole anchor set toward that one model's ability. Each
real model gets equal total weight, matching the rule that a report header
counts models and not respondents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

from irtcheck.artifact import IrtFit

# theta ~ N(0, 1) is the identification the fit records, so an anchor set that
# tells us nothing still leaves precision 1. It also keeps the objective finite
# for the empty set, which is where greedy starts.
THETA_PRIOR_PRECISION = 1.0

OBJECTIVE_MIN_VARIANCE = "min-variance"
OBJECTIVE_MAX_INFORMATION = "max-information"
OBJECTIVES = (OBJECTIVE_MIN_VARIANCE, OBJECTIVE_MAX_INFORMATION)
DEFAULT_OBJECTIVE = OBJECTIVE_MIN_VARIANCE


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

    `gains` is the marginal drop in mean posterior variance each item bought
    when it was added. It decays, and watching where it flattens is how a user
    picks n rather than guessing: the item whose gain is 1e-4 is padding.
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
    n_usable: int
    n_items: int

    @property
    def n_selected(self) -> int:
        return len(self.item_ids)

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

    Fewer than `n` items come back when fewer than `n` are usable. That is
    reported, not padded: filling an anchor set with items we said we could not
    read would undo the refusal that makes the rest of the tool honest.
    """
    if n <= 0:
        raise SelectError(f"anchor set size must be positive, got {n}")
    if objective not in OBJECTIVES:
        raise SelectError(f"unknown objective {objective!r}; choose one of {', '.join(OBJECTIVES)}")

    pool = list(fit.usable_items()) if candidates is None else list(dict.fromkeys(candidates))
    bad = [i for i in pool if not 0 <= i < fit.n_items]
    if bad:
        raise SelectError(f"candidate item index out of range: {bad[:3]}")
    if not pool:
        raise SelectError(
            "no items are eligible for selection — every item is flagged "
            "insufficient-data, ceiling or floor. With this many respondents most "
            "items cannot be read at all; add respondents (temperature samples or "
            "prompt variants via --respondent-key) before selecting an anchor set."
        )

    theta, weights = ability_distribution(fit)
    a = np.asarray(fit.a.mean, dtype=float)[pool]
    b = np.asarray(fit.b.mean, dtype=float)[pool]
    info = item_information(a, b, theta)  # (pool, thetas)

    take = min(n, len(pool))
    chosen: list[int] = []
    gains: list[float] = []
    precision = np.full(theta.shape, THETA_PRIOR_PRECISION, dtype=float)
    available = np.ones(len(pool), dtype=bool)

    for _ in range(take):
        if objective == OBJECTIVE_MAX_INFORMATION:
            # Linear in the set, so this reduces to a weighted sort. Kept for
            # anyone who wants the textbook criterion and knows what it does.
            gain = info @ weights
        else:
            current = weights / precision
            gain = current.sum() - (weights / (precision + info)).sum(axis=1)
        gain = np.where(available, gain, -np.inf)
        pick = int(np.argmax(gain))  # ties break toward the lower item index
        if not np.isfinite(gain[pick]):  # pragma: no cover - `take` bounds the loop
            break
        available[pick] = False
        chosen.append(pick)
        gains.append(float(gain[pick]))
        precision = precision + info[pick]

    test_info = info[chosen].sum(axis=0) if chosen else np.zeros_like(theta)
    return AnchorSet(
        item_ids=[fit.item_ids[pool[i]] for i in chosen],
        item_indices=[pool[i] for i in chosen],
        gains=gains,
        theta=[float(t) for t in theta],
        theta_weights=[float(w) for w in weights],
        information=[float(x) for x in test_info],
        theta_se=[float(x) for x in np.sqrt(1.0 / precision)],
        objective=objective,
        requested=n,
        n_usable=len(pool),
        n_items=fit.n_items,
    )


def select_item_ids(fit: IrtFit, n: int) -> list[str]:
    """`select_anchor` reduced to the item ids — the `select_fn` validate injects."""
    return select_anchor(fit, n).item_ids
