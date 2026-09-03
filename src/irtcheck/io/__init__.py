"""Reader registry.

Adapters register themselves at import time and this package discovers them by
walking its own directory. That is not cleverness for its own sake: wave-1
agent D adds three adapters in parallel with three other agents, and a hand-
maintained dispatch table here would be the one file they all had to edit.
Adding `io/csv.py` with a `register()` call at the bottom is enough — no shared
file is touched, so no merge conflict is reachable.

An adapter is:

    register(
        Reader(
            name="csv",
            extensions=(".csv",),
            sniff=lambda path, head: ...,   # bool; cheap, may read `head`
            read=lambda path: Iterator[ResponseRecord],
        )
    )
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from irtcheck.records import RecordError, ResponseRecord

# Bytes handed to a sniffer. Enough for a JSONL first line or a CSV header, and
# small enough that sniffing a 2 GB log costs nothing.
SNIFF_BYTES = 8192


@dataclass(frozen=True, slots=True)
class Reader:
    name: str
    extensions: tuple[str, ...]
    sniff: Callable[[Path, bytes], bool]
    read: Callable[[Path], Iterator[ResponseRecord]]
    description: str = ""


class FormatError(RecordError):
    pass


_READERS: dict[str, Reader] = {}
_DISCOVERED = False


def register(reader: Reader) -> Reader:
    if reader.name in _READERS:
        raise RuntimeError(
            f"two adapters both call themselves {reader.name!r}. Adapter names appear "
            "in --format, so they have to be unique."
        )
    _READERS[reader.name] = reader
    return reader


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    # Set before importing: a submodule that imports this package back would
    # otherwise recurse.
    _DISCOVERED = True
    for module in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if not module.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module.name}")


def readers() -> dict[str, Reader]:
    _discover()
    return dict(_READERS)


def read_any(path: str | Path, *, fmt: str | None = None) -> Iterator[ResponseRecord]:
    """Read a response file, choosing an adapter by name, extension, or content."""
    path = Path(path)
    if not path.exists():
        raise FormatError(f"no such file: {path}")
    available = readers()

    if fmt is not None:
        if fmt not in available:
            raise FormatError(
                f"unknown format {fmt!r}. Available: {', '.join(sorted(available))}"
            )
        return available[fmt].read(path)

    head = path.open("rb").read(SNIFF_BYTES)

    # Extension first — it is what the user meant when they named the file —
    # then content sniffing for the harness logs that use generic extensions.
    suffix = path.suffix.lower()
    by_extension = [r for r in available.values() if suffix in r.extensions]
    for reader in by_extension:
        if reader.sniff(path, head):
            return reader.read(path)
    for reader in available.values():
        if reader.sniff(path, head):
            return reader.read(path)
    if len(by_extension) == 1:
        return by_extension[0].read(path)

    raise FormatError(
        f"cannot tell what format {path.name} is. Pass --format explicitly "
        f"(one of: {', '.join(sorted(available))}), or convert to the reference "
        "JSONL: one {\"model_id\", \"item_id\", \"correct\"} object per line."
    )
