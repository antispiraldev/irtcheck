"""The cached fit artifact — the interface every other command reads.

Gzipped JSON, not a pickle and not a torch checkpoint. Three reasons, all of
which have bitten someone before:

  - it stays readable without importing torch, which is what lets `report`,
    `select` and `validate` run on a machine that has never installed it;
  - it is inspectable and diffable, so "the numbers changed" is a question
    someone can answer with `zcat | jq` instead of a debugger;
  - it cannot execute code on load.

SCHEMA_VERSION is the merge-time race in this repo. Two agents working in
parallel can both bump it to the same number in branches that are individually
green, and the second merge silently wins. The `contracts` CI job asserts it
appears exactly once and that an artifact round-trips; see CLAUDE.md.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Flags an item can carry. Two of these are constantly conflated and must not
# be — the distinction is the whole "refusal is a feature" argument in the spec.
#
#   DEAD              we are confident this item does not discriminate: the
#                     upper end of its `a` interval is below DEAD_THRESHOLD.
#                     A finding about the *suite*. Reportable, excludable.
#
#   INSUFFICIENT_DATA we cannot tell: the `a` interval spans zero. A finding
#                     about the *data the user gave us*. Excluded from ranking
#                     and from selection rather than ranked anyway, and the
#                     report points at adding respondents.
#
# An item flagged INSUFFICIENT_DATA is never also flagged DEAD.
FLAG_DEAD = "dead"
FLAG_INSUFFICIENT_DATA = "insufficient-data"
FLAG_CEILING = "ceiling"
FLAG_FLOOR = "floor"
FLAG_OFF_RANGE = "off-range"

ALL_FLAGS = (
    FLAG_DEAD,
    FLAG_INSUFFICIENT_DATA,
    FLAG_CEILING,
    FLAG_FLOOR,
    FLAG_OFF_RANGE,
)

# Items whose discrimination is confidently below this carry FLAG_DEAD. Chosen
# to match the conventional psychometric floor for "worth keeping"; it is a
# threshold on a scale we identify ourselves, so it is a convention, not a law.
DEAD_THRESHOLD = 0.35

# p_correct at or beyond these is a ceiling/floor item: everyone gets it right,
# or everyone gets it wrong, so it separates nobody regardless of what the
# model says about `a`.
CEILING_THRESHOLD = 0.99
FLOOR_THRESHOLD = 0.01


class ArtifactError(ValueError):
    pass


@dataclass(slots=True)
class Posterior:
    """Marginal posterior summaries for one vector of parameters.

    Held as parallel lists rather than a list of dicts so the JSON stays small
    on a 4000-item suite and columns can be read without a comprehension.
    """

    mean: list[float]
    sd: list[float]
    hdi_low: list[float]
    hdi_high: list[float]

    def __len__(self) -> int:
        return len(self.mean)

    def spans_zero(self) -> list[bool]:
        return [lo <= 0.0 <= hi for lo, hi in zip(self.hdi_low, self.hdi_high, strict=True)]

    def validate(self, n: int, name: str) -> None:
        for attr in ("mean", "sd", "hdi_low", "hdi_high"):
            got = len(getattr(self, attr))
            if got != n:
                raise ArtifactError(f"{name}.{attr} has {got} entries, expected {n}")
        for lo, hi in zip(self.hdi_low, self.hdi_high, strict=True):
            if lo > hi:
                raise ArtifactError(f"{name} has an interval with hdi_low > hdi_high")


@dataclass(slots=True)
class EmbeddedResponses:
    """The response matrix the fit was built from, in sparse triplet form.

    The artifact carries its own responses so that `irtcheck validate
    suite.irt` works as specified, with one argument. Validation is
    leave-one-model-out: it refits the 2PL k times on k-1 models each, so it
    needs the raw matrix, not the posteriors. Requiring the user to hand back
    the original JSONL — and to have kept it, unchanged, next to the artifact —
    would make the headline command the most fragile one in the tool.

    The cost is size, and it is small: responses are three small integers each
    and gzip flattens them. A 15 x 4000 suite adds well under a megabyte.
    `irtcheck fit --no-embed-responses` opts out; `validate` then explains what
    is missing rather than failing obscurely.
    """

    rows: list[int]
    cols: list[int]
    obs: list[int]

    def __len__(self) -> int:
        return len(self.obs)

    def validate(self, n_respondents: int, n_items: int) -> None:
        if not (len(self.rows) == len(self.cols) == len(self.obs)):
            raise ArtifactError(
                "embedded responses have mismatched rows/cols/obs lengths "
                f"({len(self.rows)}/{len(self.cols)}/{len(self.obs)})"
            )
        if self.rows and (max(self.rows) >= n_respondents or min(self.rows) < 0):
            raise ArtifactError("embedded responses index a respondent that does not exist")
        if self.cols and (max(self.cols) >= n_items or min(self.cols) < 0):
            raise ArtifactError("embedded responses index an item that does not exist")
        bad = {o for o in self.obs} - {0, 1}
        if bad:
            raise ArtifactError(f"embedded responses contain non-binary values: {sorted(bad)}")


@dataclass(slots=True)
class IrtFit:
    """A fitted 2PL and everything needed to report, select and validate from it."""

    # provenance
    irtcheck_version: str
    created: str
    model: dict[str, Any]  # kind, priors, identification

    # respondents
    respondent_key: list[str]
    respondent_ids: list[str]
    derives_from: list[str]  # real model behind each respondent
    theta: Posterior

    # items
    item_ids: list[str]
    a: Posterior  # discrimination
    b: Posterior  # difficulty
    n_resp: list[int]
    p_correct: list[float]
    flags: list[list[str]]

    diagnostics: dict[str, Any] = field(default_factory=dict)
    passthrough: dict[str, dict[str, Any]] = field(default_factory=dict)
    responses: EmbeddedResponses | None = None
    schema_version: int = SCHEMA_VERSION

    # -- derived ------------------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_respondents(self) -> int:
        return len(self.respondent_ids)

    @property
    def n_real_models(self) -> int:
        """Distinct real models. The number that belongs in a report header."""
        return len(set(self.derives_from))

    @property
    def has_pseudo_respondents(self) -> bool:
        return self.n_respondents != self.n_real_models

    def flagged(self, flag: str) -> list[int]:
        return [i for i, flags in enumerate(self.flags) if flag in flags]

    def usable_items(self) -> list[int]:
        """Item indices eligible for ranking and selection.

        Excludes insufficient-data (we cannot tell) and ceiling/floor (nothing
        to tell). Dead items stay in: they are a confident finding, and a caller
        that wants them gone can filter. Selection maximises information, so a
        dead item would not be picked anyway.
        """
        skip = {FLAG_INSUFFICIENT_DATA, FLAG_CEILING, FLAG_FLOOR}
        return [i for i, flags in enumerate(self.flags) if not skip.intersection(flags)]

    def matrix(self):
        """Rebuild the ResponseMatrix this fit came from.

        `validate` uses this for the leave-one-model-out refits. Raises rather
        than guessing when the fit was written with --no-embed-responses.
        """
        import numpy as np

        from irtcheck.matrix import ResponseMatrix

        if self.responses is None:
            raise ArtifactError(
                "this artifact carries no responses, so it cannot be re-fitted. "
                "It was written with --no-embed-responses. Re-run `irtcheck fit` "
                "without that flag to use `validate`."
            )
        return ResponseMatrix(
            respondent_ids=list(self.respondent_ids),
            item_ids=list(self.item_ids),
            derives_from=list(self.derives_from),
            rows=np.asarray(self.responses.rows, dtype=np.int32),
            cols=np.asarray(self.responses.cols, dtype=np.int32),
            obs=np.asarray(self.responses.obs, dtype=np.float32),
            respondent_key=tuple(self.respondent_key),
            passthrough={k: dict(v) for k, v in self.passthrough.items()},
        )

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # schema_version leads the file so a reader can reject an artifact from
        # the future before parsing anything it might not understand.
        return {"schema_version": data.pop("schema_version"), **data}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        # Not sort_keys: to_dict puts schema_version first deliberately, so a
        # reader can reject an artifact from the future before parsing fields it
        # may not understand. Field order is otherwise fixed by the dataclass,
        # so output stays deterministic without sorting.
        payload = json.dumps(self.to_dict(), separators=(",", ":"))
        # Two bytes of the gzip header would otherwise make the same fit written
        # to two paths differ: mtime, and the stored original filename. Both are
        # suppressed — writing through `fileobj` with an empty `filename` keeps
        # the name out of the header — so identical content gives identical
        # bytes and the contracts job's round-trip assertion means something.
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as handle:
            handle.write(payload.encode("utf-8"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> IrtFit:
        path = Path(path)
        try:
            with gzip.open(path, "rb") as handle:
                data = json.loads(handle.read().decode("utf-8"))
        except OSError as exc:
            raise ArtifactError(f"cannot read {path}: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ArtifactError(
                f"{path} is not a valid irtcheck artifact (bad JSON): {exc}"
            ) from exc
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str | None = None) -> IrtFit:
        where = f" in {source}" if source else ""
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ArtifactError(
                f"artifact{where} has schema_version {version!r}, this irtcheck "
                f"reads {SCHEMA_VERSION}. Re-run `irtcheck fit`."
            )

        try:
            fit = cls(
                irtcheck_version=data["irtcheck_version"],
                created=data["created"],
                model=data["model"],
                respondent_key=list(data["respondent_key"]),
                respondent_ids=list(data["respondent_ids"]),
                derives_from=list(data["derives_from"]),
                theta=Posterior(**data["theta"]),
                item_ids=list(data["item_ids"]),
                a=Posterior(**data["a"]),
                b=Posterior(**data["b"]),
                n_resp=list(data["n_resp"]),
                p_correct=list(data["p_correct"]),
                flags=[list(f) for f in data["flags"]],
                diagnostics=data.get("diagnostics", {}),
                passthrough=data.get("passthrough", {}),
                responses=(
                    EmbeddedResponses(**data["responses"])
                    if data.get("responses") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ArtifactError(f"artifact{where} is missing field {exc}") from exc
        except TypeError as exc:
            raise ArtifactError(f"artifact{where} has a malformed field: {exc}") from exc

        fit.validate()
        return fit

    def validate(self) -> None:
        """Structural check. Cheap, and runs on every load."""
        n_items, n_resp = self.n_items, self.n_respondents
        if n_items == 0:
            raise ArtifactError("artifact has no items")
        if n_resp == 0:
            raise ArtifactError("artifact has no respondents")
        if len(self.derives_from) != n_resp:
            raise ArtifactError(
                f"derives_from has {len(self.derives_from)} entries for {n_resp} respondents"
            )
        self.theta.validate(n_resp, "theta")
        self.a.validate(n_items, "a")
        self.b.validate(n_items, "b")
        for name, values in (("n_resp", self.n_resp), ("p_correct", self.p_correct),
                             ("flags", self.flags)):
            if len(values) != n_items:
                raise ArtifactError(f"{name} has {len(values)} entries for {n_items} items")
        if self.responses is not None:
            self.responses.validate(n_resp, n_items)
        unknown = {f for flags in self.flags for f in flags} - set(ALL_FLAGS)
        if unknown:
            raise ArtifactError(f"unknown item flag(s): {', '.join(sorted(unknown))}")
        for i, flags in enumerate(self.flags):
            if FLAG_DEAD in flags and FLAG_INSUFFICIENT_DATA in flags:
                raise ArtifactError(
                    f"item {self.item_ids[i]!r} is flagged both {FLAG_DEAD} and "
                    f"{FLAG_INSUFFICIENT_DATA}. 'we are confident it does not "
                    "discriminate' and 'we cannot tell' are mutually exclusive claims."
                )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def compute_flags(
    a: Posterior,
    b: Posterior,
    p_correct: list[float],
    theta_mean: list[float],
    *,
    dead_threshold: float = DEAD_THRESHOLD,
) -> list[list[str]]:
    """Derive item flags from posteriors and observed rates.

    Shared by the fitter and by synth.py so that a fabricated artifact and a
    real one flag by identical rules — otherwise agents building against synth
    would be testing against a fiction.
    """
    theta_low = min(theta_mean) if theta_mean else 0.0
    theta_high = max(theta_mean) if theta_mean else 0.0
    span = max(theta_high - theta_low, 1e-6)

    flags: list[list[str]] = []
    for i in range(len(a)):
        item: list[str] = []
        undetermined = a.hdi_low[i] <= 0.0 <= a.hdi_high[i]
        if undetermined:
            item.append(FLAG_INSUFFICIENT_DATA)
        elif a.hdi_high[i] < dead_threshold:
            item.append(FLAG_DEAD)

        p = p_correct[i]
        if p == p or p is not None:  # not NaN
            if p >= CEILING_THRESHOLD:
                item.append(FLAG_CEILING)
            elif p <= FLOOR_THRESHOLD:
                item.append(FLAG_FLOOR)

        # Difficulty well outside where the respondents actually sit: the item
        # may discriminate beautifully, just not for anyone in this matrix.
        # This is the finding the information curve is drawn to show.
        if not undetermined and not (theta_low - span <= b.mean[i] <= theta_high + span):
            item.append(FLAG_OFF_RANGE)

        flags.append(item)
    return flags
