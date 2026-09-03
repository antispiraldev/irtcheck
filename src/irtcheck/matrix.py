"""The response matrix: respondents x items, in sparse triplet form.

Stored long rather than dense because eval matrices are ragged — not every
model answers every item, subsets get re-run, harnesses drop failures — and
because Pyro's plate over observations wants exactly (respondent_index,
item_index, observation) anyway.

The field that matters most here is `derives_from`. See below.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from irtcheck.records import ResponseRecord

DEFAULT_RESPONDENT_KEY: tuple[str, ...] = ("model_id",)


class MatrixError(ValueError):
    pass


@dataclass(slots=True)
class ResponseMatrix:
    """A graded response matrix in sparse triplet form.

    respondent_ids  composed identities, in first-seen order
    item_ids        item identities, in first-seen order
    derives_from    the real model_id behind each respondent  <-- load-bearing
    rows/cols/obs   parallel arrays, one entry per response

    `derives_from` is what makes leave-one-model-out honest. With
    --respondent-key model_id,prompt_variant one real model becomes several
    respondents, and holding out a *respondent* would leave that model's other
    prompt variants in the fit — the anchor set would then be chosen with
    knowledge of the model it is about to be tested on, and the headline rank
    correlation would come out inflated for a reason no test would catch.
    validate holds out every respondent sharing a `derives_from`.
    """

    respondent_ids: list[str]
    item_ids: list[str]
    derives_from: list[str]
    rows: np.ndarray  # int32, index into respondent_ids
    cols: np.ndarray  # int32, index into item_ids
    obs: np.ndarray  # float32, 0.0 or 1.0
    respondent_key: tuple[str, ...] = DEFAULT_RESPONDENT_KEY
    passthrough: dict[str, dict[str, object]] = None  # field -> {item_id: value}

    def __post_init__(self) -> None:
        if self.passthrough is None:
            self.passthrough = {}

    # -- shape ---------------------------------------------------------------

    @property
    def n_respondents(self) -> int:
        return len(self.respondent_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_responses(self) -> int:
        return int(self.obs.size)

    @property
    def n_real_models(self) -> int:
        """Distinct real models, ignoring pseudo-respondent inflation.

        This is the number that goes in the report header. Printing 60
        respondents when they are twelve temperature samples of five models
        would misrepresent exactly the thing the tool exists to be honest about.
        """
        return len(set(self.derives_from))

    @property
    def density(self) -> float:
        total = self.n_respondents * self.n_items
        return self.n_responses / total if total else 0.0

    # -- per-item summaries --------------------------------------------------

    def item_response_counts(self) -> np.ndarray:
        return np.bincount(self.cols, minlength=self.n_items).astype(np.int64)

    def item_p_correct(self) -> np.ndarray:
        """Proportion correct per item. NaN for an item with no responses."""
        counts = self.item_response_counts()
        totals = np.bincount(self.cols, weights=self.obs, minlength=self.n_items)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    def respondent_accuracy(self) -> np.ndarray:
        """Proportion correct per respondent — the full-suite score.

        This is the ground truth `validate` compares its anchor-set ranking
        against: the aggregate accuracy the user's suite reports today.
        """
        counts = np.bincount(self.rows, minlength=self.n_respondents)
        totals = np.bincount(self.rows, weights=self.obs, minlength=self.n_respondents)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    # -- subsetting ----------------------------------------------------------

    def drop_model(self, model_id: str) -> ResponseMatrix:
        """Every respondent deriving from `model_id` removed.

        Not "the respondent named model_id" — every pseudo-respondent it
        produced. See the class docstring.
        """
        if model_id not in set(self.derives_from):
            raise MatrixError(
                f"no respondent derives from model {model_id!r}; "
                f"known models: {', '.join(sorted(set(self.derives_from)))}"
            )
        keep = [i for i, src in enumerate(self.derives_from) if src != model_id]
        return self._select_respondents(keep)

    def select_items(self, item_ids: Iterable[str]) -> ResponseMatrix:
        """Restrict to `item_ids`, preserving their given order."""
        wanted = list(dict.fromkeys(item_ids))
        index = {item: i for i, item in enumerate(self.item_ids)}
        unknown = [i for i in wanted if i not in index]
        if unknown:
            raise MatrixError(
                f"{len(unknown)} item id(s) not in this matrix, e.g. {unknown[:3]}"
            )
        old_to_new = {index[item]: new for new, item in enumerate(wanted)}
        mask = np.isin(self.cols, list(old_to_new))
        return ResponseMatrix(
            respondent_ids=list(self.respondent_ids),
            item_ids=wanted,
            derives_from=list(self.derives_from),
            rows=self.rows[mask].copy(),
            cols=np.array([old_to_new[c] for c in self.cols[mask]], dtype=np.int32),
            obs=self.obs[mask].copy(),
            respondent_key=self.respondent_key,
            passthrough={
                f: {k: v for k, v in values.items() if k in old_to_new or k in wanted}
                for f, values in self.passthrough.items()
            },
        )

    def _select_respondents(self, keep: list[int]) -> ResponseMatrix:
        old_to_new = {old: new for new, old in enumerate(keep)}
        mask = np.isin(self.rows, keep)
        return ResponseMatrix(
            respondent_ids=[self.respondent_ids[i] for i in keep],
            item_ids=list(self.item_ids),
            derives_from=[self.derives_from[i] for i in keep],
            rows=np.array([old_to_new[r] for r in self.rows[mask]], dtype=np.int32),
            cols=self.cols[mask].copy(),
            obs=self.obs[mask].copy(),
            respondent_key=self.respondent_key,
            passthrough={f: dict(v) for f, v in self.passthrough.items()},
        )


def build_matrix(
    records: Iterable[ResponseRecord],
    *,
    respondent_key: tuple[str, ...] = DEFAULT_RESPONDENT_KEY,
) -> ResponseMatrix:
    """Assemble a ResponseMatrix from a stream of records.

    Duplicate (respondent, item) pairs raise rather than being deduplicated.
    A harness that emitted the same item twice for one respondent either
    re-ran it — in which case which answer counts is the user's call, not
    ours — or the respondent key is too coarse to separate two real runs,
    which is a --respondent-key the user needs to know about.
    """
    respondent_index: dict[str, int] = {}
    item_index: dict[str, int] = {}
    derives_from: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    obs: list[float] = []
    seen: set[tuple[int, int]] = set()
    duplicates: Counter[tuple[str, str]] = Counter()
    passthrough: dict[str, dict[str, object]] = {}

    for record in records:
        composed = record.key(respondent_key)
        r = respondent_index.get(composed)
        if r is None:
            r = len(respondent_index)
            respondent_index[composed] = r
            derives_from.append(record.model_id)
        elif derives_from[r] != record.model_id:
            # Two different real models composing to one respondent id means the
            # key does not identify a respondent. Almost always a "|" already
            # present in a field value.
            raise MatrixError(
                f"respondent {composed!r} is claimed by two models "
                f"({derives_from[r]!r} and {record.model_id!r}). The respondent key "
                f"{'+'.join(respondent_key)} does not uniquely identify a respondent."
            )

        c = item_index.get(record.item_id)
        if c is None:
            c = len(item_index)
            item_index[record.item_id] = c

        if (r, c) in seen:
            duplicates[(composed, record.item_id)] += 1
            continue
        seen.add((r, c))

        rows.append(r)
        cols.append(c)
        obs.append(float(record.correct))

        for name in ("subject", "split"):
            if name in record.extra:
                passthrough.setdefault(name, {}).setdefault(record.item_id, record.extra[name])

    if duplicates:
        worst = duplicates.most_common(3)
        raise MatrixError(
            f"{len(duplicates)} (respondent, item) pair(s) appear more than once, e.g. "
            + "; ".join(f"{r}/{i} x{n + 1}" for (r, i), n in worst)
            + ". Deduplicate upstream, or compose a finer --respondent-key "
            "(adding prompt_variant or a seed field usually separates re-runs)."
        )

    if not rows:
        raise MatrixError("no responses were read — the input produced zero records")

    return ResponseMatrix(
        respondent_ids=list(respondent_index),
        item_ids=list(item_index),
        derives_from=derives_from,
        rows=np.array(rows, dtype=np.int32),
        cols=np.array(cols, dtype=np.int32),
        obs=np.array(obs, dtype=np.float32),
        respondent_key=tuple(respondent_key),
        passthrough=passthrough,
    )


def parse_respondent_key(spec: str) -> tuple[str, ...]:
    """Read a --respondent-key value like "model_id,prompt_variant"."""
    fields = tuple(f.strip() for f in spec.split(",") if f.strip())
    if not fields:
        raise MatrixError("--respondent-key cannot be empty")
    if "model_id" not in fields:
        raise MatrixError(
            "--respondent-key must include model_id. Pseudo-respondents are "
            "variations *of a model*, and validate has to be able to hold all of "
            "one model's variations out together."
        )
    duplicated = [f for f, n in Counter(fields).items() if n > 1]
    if duplicated:
        raise MatrixError(f"--respondent-key repeats {', '.join(duplicated)}")
    return fields
