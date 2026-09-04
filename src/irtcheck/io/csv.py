"""CSV/TSV with the same columns as the reference JSONL.

    model_id,item_id,correct
    claude-sonnet-4-5,mmlu_hs_bio_0412,1

Any of PASSTHROUGH_FIELDS present as extra columns is carried through, and so
is any other column — a `--respondent-key` can name one.

The whole file goes through `build_record()`, so a bad `correct` value reads
exactly as it would in JSONL. That matters more than it sounds: CSV has no
types, so `correct` arrives as the string "0.87" rather than the float 0.87,
and a naive reader would report "cannot read `correct` from '0.87'" instead of
the refusal-to-threshold message the JSONL reader gives. Numeric-looking values
in `correct` and `raw_score` are therefore converted before the record is
built, and nothing else is: `item_id` "0042" must not become "42".

## When your harness writes nothing irtcheck can read

This is the fallback the whole input contract rests on, and it is deliberately
tiny — if your scoring loop already knows which model got which item right, it
already has everything irtcheck needs. Emit one row per graded response:

    import csv

    with open("responses.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, ["model_id", "item_id", "correct"])
        writer.writeheader()
        for result in your_results:            # whatever your loop already has
            writer.writerow(
                {
                    "model_id": result.model,  # "claude-sonnet-4-5"
                    "item_id": result.item,    # stable across runs, unique across tasks
                    "correct": int(result.passed),
                }
            )

    irtcheck fit responses.csv -o suite.irt

or, if JSON is more natural where you are, the same three fields as JSONL —
`{"model_id": ..., "item_id": ..., "correct": ...}`, one object per line.

Three things are worth getting right while you are there, because they are
expensive to fix later:

- **`item_id` must be stable across runs and unique across tasks.** It is the
  join key between every model's answers. A row index is fine; a row index
  without the task name in front of it is not, because task A's item 0 and
  task B's item 0 will silently become one item.
- **`correct` must be 0 or 1.** irtcheck refuses 0.87 rather than picking a
  threshold for you. If your scorer is continuous, threshold it yourself, put
  the original in `raw_score`, and say in the PR/report what cutoff you used.
- **More models beat more items.** Respondent count is the binding constraint,
  so if you have five models, add prompt variants or temperature samples as
  extra columns and compose them in with
  `--respondent-key model_id,prompt_variant`.
"""

from __future__ import annotations

import csv as _csv
from collections.abc import Iterator
from pathlib import Path

from irtcheck.io import Reader, register
from irtcheck.records import REQUIRED_FIELDS, RecordError, ResponseRecord, build_record

# Fields whose values are numbers rather than labels, and so are safe to
# convert from CSV's untyped strings. `correct` because coerce_correct's error
# messages distinguish 0.87 from "0.87"; `raw_score` because a report that
# prints it wants a number.
_NUMERIC_FIELDS = ("correct", "raw_score")

_DELIMITERS = (",", "\t", ";", "|")


def _delimiter(path: Path, header_line: str) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    # Whichever candidate splits the header into the most fields wins, with a
    # comma breaking ties. Sniffer guesses from quoting habits and gets this
    # wrong on a three-column file with no quotes at all.
    best = max(_DELIMITERS, key=lambda d: (len(header_line.split(d)), d == ","))
    return best if len(header_line.split(best)) > 1 else ","


def _number(text: str) -> float | int | str:
    """"1" -> 1, "0.87" -> 0.87, "yes" -> "yes"."""
    stripped = text.strip()
    if not stripped:
        return text
    try:
        value = float(stripped)
    except ValueError:
        return text
    return int(value) if value.is_integer() and "." not in stripped else value


def read_csv(path: Path) -> Iterator[ResponseRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        if not first.strip():
            raise RecordError("file is empty", source=str(path))
        reader = _csv.DictReader(handle, delimiter=_delimiter(path, first))
        fields = reader.fieldnames or []
        missing = [f for f in REQUIRED_FIELDS if f not in fields]
        if missing:
            raise RecordError(
                f"header is missing column(s) {', '.join(missing)}; "
                f"got {', '.join(fields) or '<no header>'}",
                source=str(path),
                line=1,
            )
        for row in reader:
            # DictReader gives {} for a blank line and None keys for a row with
            # more cells than the header.
            if not row or all(v is None or not str(v).strip() for v in row.values()):
                continue
            if None in row:
                raise RecordError(
                    f"row has more cells ({len(fields) + len(row[None])}) than the "
                    f"header has columns ({len(fields)})",
                    source=str(path),
                    line=reader.line_num,
                )
            raw: dict[str, object] = {}
            for key, value in row.items():
                if value is None or value == "":
                    # An empty cell is an absent value, not the empty string:
                    # a missing required field must say "missing", and an
                    # absent passthrough must not become extra={"split": ""}.
                    continue
                raw[key] = _number(value) if key in _NUMERIC_FIELDS else value
            yield build_record(raw, source=str(path), line=reader.line_num)


def _sniff(path: Path, head: bytes) -> bool:
    # errors="ignore" because the sniff window is a byte count and lands
    # wherever it lands — a multibyte character straddling the end of it is not
    # evidence about the format, and the header line is at the start anyway.
    text = head.decode("utf-8-sig", errors="ignore")
    for line in text.splitlines():
        if not line.strip():
            continue
        columns = {c.strip() for c in line.split(_delimiter(path, line))}
        return all(f in columns for f in REQUIRED_FIELDS)
    return False


register(
    Reader(
        name="csv",
        extensions=(".csv", ".tsv"),
        sniff=_sniff,
        read=read_csv,
        description="delimited text with model_id, item_id, correct columns",
    )
)
