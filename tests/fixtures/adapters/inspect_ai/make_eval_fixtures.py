"""Regenerate the binary `.eval` fixtures beside this file.

    python3 tests/fixtures/adapters/inspect_ai/make_eval_fixtures.py

A `.eval` log is a zip of `header.json` plus one `samples/<id>_epoch_<n>.json`
per sample. Two are written, from the same source JSON log so that the three
forms can be asserted equal:

`deflate.eval`
    What Inspect wrote before it moved to zstandard (0.3.50-era). Python's
    stdlib reads it, so the adapter reads it.

`zstandard.eval`
    What current Inspect writes. It is *fabricated* — the members are stored,
    then the compression-method field is patched to 93 in both the local
    headers and the central directory — because compressing with zstd would
    need the dependency this project refuses to add. That is enough to
    exercise the refusal path, which is all the adapter does with such a file:
    it reads the central directory, sees a method it cannot inflate, and says
    `inspect log convert`. Nothing decompresses it, here or in the adapter.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "logs" / "2026-05-01T12-00-00+00-00_arc-easy_aTGeC2v9.json"

ZIP_ZSTANDARD = 93
LOCAL_HEADER = b"PK\x03\x04"
CENTRAL_HEADER = b"PK\x01\x02"


def members(log: dict) -> list[tuple[str, bytes]]:
    header = {k: v for k, v in log.items() if k != "samples"}
    entries = [("header.json", json.dumps(header, indent=2).encode())]
    for sample in log["samples"]:
        name = f"samples/{sample['id']}_epoch_{sample['epoch']}.json"
        entries.append((name, json.dumps(sample, indent=2).encode()))
    return entries


def write_zip(path: Path, entries: list[tuple[str, bytes]], compression: int) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 5, 1, 12, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)


def patch_compression_method(path: Path, method: int) -> None:
    """Rewrite every stored member's compression method to `method`."""
    data = bytearray(path.read_bytes())
    packed = struct.pack("<H", method)
    for signature, offset in ((LOCAL_HEADER, 8), (CENTRAL_HEADER, 10)):
        start = 0
        while (found := data.find(signature, start)) != -1:
            data[found + offset : found + offset + 2] = packed
            start = found + 4
    path.write_bytes(bytes(data))


def main() -> None:
    log = json.loads(SOURCE.read_text())
    entries = members(log)
    write_zip(HERE / "deflate.eval", entries, zipfile.ZIP_DEFLATED)
    fabricated = HERE / "zstandard.eval"
    write_zip(fabricated, entries, zipfile.ZIP_STORED)
    patch_compression_method(fabricated, ZIP_ZSTANDARD)
    print(f"wrote {HERE / 'deflate.eval'} and {fabricated}")


if __name__ == "__main__":
    main()
