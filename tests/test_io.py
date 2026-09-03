"""The reader registry and the reference JSONL format.

Wave-1 agent D adds three more adapters here without editing any shared file;
the registry test below is what proves that arrangement still holds."""

from __future__ import annotations

import json

import pytest

from irtcheck.io import FormatError, read_any, readers
from irtcheck.matrix import build_matrix
from irtcheck.records import RecordError

ROWS = [
    {"model_id": "a", "item_id": "i1", "correct": 1},
    {"model_id": "a", "item_id": "i2", "correct": 0, "subject": "bio"},
    {"model_id": "b", "item_id": "i1", "correct": True},
]


def write_jsonl(path, rows=ROWS):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_jsonl_is_registered():
    assert "jsonl" in readers()


def test_adapters_register_themselves_by_being_present():
    """No shared dispatch table to edit — that is what keeps agent D out of
    three other agents' way."""
    for name, reader in readers().items():
        assert reader.name == name
        assert callable(reader.read)


def test_read_reference_jsonl(tmp_path):
    records = list(read_any(write_jsonl(tmp_path / "r.jsonl")))
    assert len(records) == 3
    assert records[2].correct == 1  # `true` coerced
    assert records[1].extra["subject"] == "bio"


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n\n\n" + json.dumps(ROWS[1]) + "\n")
    assert len(list(read_any(path))) == 2


def test_bad_json_names_the_line(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n{not json\n")
    with pytest.raises(RecordError, match=r":2"):
        list(read_any(path))


def test_a_json_array_line_is_rejected(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('[1, 2, 3]\n')
    with pytest.raises(RecordError):
        list(read_any(path, fmt="jsonl"))


def test_missing_file(tmp_path):
    with pytest.raises(FormatError, match="no such file"):
        list(read_any(tmp_path / "nope.jsonl"))


def test_unknown_format_lists_what_is_available(tmp_path):
    with pytest.raises(FormatError, match="Available:"):
        list(read_any(write_jsonl(tmp_path / "r.jsonl"), fmt="parquet"))


def test_unrecognisable_file_points_at_the_reference_format(tmp_path):
    path = tmp_path / "mystery.bin"
    path.write_bytes(b"\x00\x01\x02 nothing readable here")
    with pytest.raises(FormatError, match="reference JSONL"):
        list(read_any(path))


def test_end_to_end_into_a_matrix(tmp_path):
    matrix = build_matrix(read_any(write_jsonl(tmp_path / "r.jsonl")))
    assert matrix.n_respondents == 2
    assert matrix.n_items == 2
    assert matrix.n_responses == 3
