"""The input record: one (respondent, item) response.

This is the narrow waist of the whole tool. Every adapter — JSONL, CSV,
lm-eval-harness, Inspect — produces a stream of these, and everything
downstream consumes only these. An adapter that needs a new field on
ResponseRecord is an adapter that has found something the contract does not
cover, and that is a conversation, not a patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Carried through to reports when present, never required. `prompt_variant` is
# the one that earns its place: it is the field most likely to be composed into
# a respondent key, because prompt variants of one model are the cheapest
# pseudo-respondents a user can produce. See CLAUDE.md → Pseudo-respondents.
PASSTHROUGH_FIELDS = ("subject", "split", "raw_score", "prompt_variant")

REQUIRED_FIELDS = ("model_id", "item_id", "correct")

# Strings a harness might reasonably write for a boolean. Anything outside this
# set raises rather than being coerced: a response silently read as incorrect is
# a wrong difficulty estimate that nothing downstream can detect.
_TRUE = {"1", "true", "t", "yes", "y", "correct"}
_FALSE = {"0", "false", "f", "no", "n", "incorrect"}


class RecordError(ValueError):
    """A record that cannot be read. Carries enough context to find the line."""

    def __init__(self, message: str, *, source: str | None = None, line: int | None = None):
        self.source = source
        self.line = line
        where = ""
        if source:
            where = f" ({source}"
            where += f":{line})" if line is not None else ")"
        super().__init__(f"{message}{where}")


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    """One graded response.

    `extra` holds whichever of PASSTHROUGH_FIELDS the source supplied, plus
    anything a respondent key names. It is deliberately not typed further —
    the tool does not interpret these values, it only groups and reports by
    them.
    """

    model_id: str
    item_id: str
    correct: int  # exactly 0 or 1
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self, fields: tuple[str, ...]) -> str:
        """Compose this record's respondent identity from `fields`.

        Joined with "|" because it has to survive being a dict key, a column
        header and a line in a report, and item ids in the wild already contain
        every other plausible separator.
        """
        parts = []
        for name in fields:
            value = self.model_id if name == "model_id" else self.extra.get(name)
            if value is None:
                raise RecordError(
                    f"respondent key names {name!r} but this record has no such field "
                    f"(model_id={self.model_id!r}, item_id={self.item_id!r}). "
                    f"Available: {', '.join(sorted(self.extra) or ['<none>'])}"
                )
            parts.append(str(value))
        return "|".join(parts)


def coerce_correct(value: Any, *, source: str | None = None, line: int | None = None) -> int:
    """Read a `correct` field as 0 or 1, or raise.

    Accepts bools, 0/1 as int or float, and the usual string spellings. Rejects
    everything else — including 0.5, which is the shape a rubric score arrives
    in and is exactly the case the tool must refuse until thresholding is a
    documented, printed-in-the-header feature rather than a silent coercion.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise RecordError(f"`correct` must be 0 or 1, got {value}", source=source, line=line)
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        raise RecordError(
            f"`correct` must be 0 or 1, got {value}. A continuous score has to be "
            "thresholded deliberately — irtcheck will not guess a cutoff",
            source=source,
            line=line,
        )
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return 1
        if lowered in _FALSE:
            return 0
    raise RecordError(f"cannot read `correct` from {value!r}", source=source, line=line)


def build_record(
    raw: dict[str, Any], *, source: str | None = None, line: int | None = None
) -> ResponseRecord:
    """Turn a decoded mapping into a ResponseRecord, or raise RecordError."""
    missing = [f for f in REQUIRED_FIELDS if f not in raw or raw[f] is None]
    if missing:
        raise RecordError(
            f"missing required field(s) {', '.join(missing)}; "
            f"got keys {', '.join(sorted(raw)) or '<none>'}",
            source=source,
            line=line,
        )

    model_id = str(raw["model_id"])
    item_id = str(raw["item_id"])
    if not model_id or not item_id:
        raise RecordError("model_id and item_id must be non-empty", source=source, line=line)

    extra = {k: raw[k] for k in PASSTHROUGH_FIELDS if k in raw and raw[k] is not None}
    # Anything else the source carried is kept too, so a --respondent-key can
    # name a field this tool has never heard of (checkpoint, quantization, seed).
    for key, value in raw.items():
        if key not in REQUIRED_FIELDS and key not in extra and value is not None:
            extra[key] = value

    return ResponseRecord(
        model_id=model_id,
        item_id=item_id,
        correct=coerce_correct(raw["correct"], source=source, line=line),
        extra=extra,
    )
