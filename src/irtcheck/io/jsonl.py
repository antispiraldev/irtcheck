"""The reference format: one JSON object per line.

    {"model_id": "claude-sonnet-4-5", "item_id": "mmlu_hs_bio_0412", "correct": 1}

Everything else is an adapter onto this. When a harness writes something we
cannot read, the documented fallback is for the user to emit this from their
own scoring loop — so this reader is the one that has to have the clearest
errors in the package.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from irtcheck.io import Reader, register
from irtcheck.records import RecordError, ResponseRecord, build_record


def read_jsonl(path: Path) -> Iterator[ResponseRecord]:
    blank = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                blank += 1
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RecordError(
                    f"line is not valid JSON: {exc.msg}", source=str(path), line=number
                ) from exc
            if not isinstance(raw, dict):
                raise RecordError(
                    f"expected a JSON object, got {type(raw).__name__}",
                    source=str(path),
                    line=number,
                )
            yield build_record(raw, source=str(path), line=number)


def _sniff(path: Path, head: bytes) -> bool:
    for line in head.split(b"\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith(b"{"):
            return False
        try:
            raw = json.loads(stripped)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A truncated final line in the sniff window is not a failure —
            # an object that starts right is good enough evidence.
            return stripped.startswith(b'{"') or stripped.startswith(b"{ ")
        return isinstance(raw, dict) and "item_id" in raw and "model_id" in raw
    return False


register(
    Reader(
        name="jsonl",
        extensions=(".jsonl", ".ndjson"),
        sniff=_sniff,
        read=read_jsonl,
        description="reference format: one {model_id, item_id, correct} object per line",
    )
)
