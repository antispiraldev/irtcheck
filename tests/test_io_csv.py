"""The CSV adapter.

The load-bearing test is `test_csv_matches_the_reference_jsonl`: an adapter is
faithful only if the matrix it builds is the matrix a hand-written reference
JSONL of the same responses builds. Everything else here is about the ways CSV
differs from JSON — no types, no nulls, a header row — and about keeping the
error messages identical to the reference reader's, because a user comparing
the two formats should not have to learn two vocabularies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from irtcheck.io import read_any, readers
from irtcheck.matrix import build_matrix
from irtcheck.records import RecordError

FIXTURES = Path(__file__).parent / "fixtures" / "adapters" / "csv"


def assert_same_matrix(left, right) -> None:
    assert left.respondent_ids == right.respondent_ids
    assert left.item_ids == right.item_ids
    assert left.derives_from == right.derives_from
    assert left.rows.tolist() == right.rows.tolist()
    assert left.cols.tolist() == right.cols.tolist()
    assert left.obs.tolist() == right.obs.tolist()
    assert left.passthrough == right.passthrough


def test_csv_is_registered():
    assert "csv" in readers()


def test_csv_matches_the_reference_jsonl():
    from_csv = build_matrix(read_any(FIXTURES / "responses.csv"))
    from_jsonl = build_matrix(read_any(FIXTURES / "reference.jsonl"))
    assert_same_matrix(from_csv, from_jsonl)
    assert from_csv.n_respondents == 2
    assert from_csv.n_items == 3


def test_records_are_identical_field_for_field():
    """Not just the matrix: the passthrough values have to survive as the same
    types, or a report of the CSV and a report of the JSONL would differ."""
    assert list(read_any(FIXTURES / "responses.csv")) == list(
        read_any(FIXTURES / "reference.jsonl")
    )


def test_a_csv_is_sniffed_without_being_told():
    records = list(read_any(FIXTURES / "responses.csv", fmt=None))
    assert records[0].item_id == "mmlu_hs_bio_0412"


def test_tabs_are_read_as_a_delimiter():
    records = list(read_any(FIXTURES / "tab_separated.tsv"))
    assert [r.correct for r in records] == [1, 0]


def test_a_continuous_score_is_refused_in_the_same_words_as_jsonl(tmp_path):
    """CSV has no types, so `correct` arrives as the string "0.87". Reading it
    as a string would produce "cannot read `correct` from '0.87'" — a different
    complaint from the one the reference reader makes about the same data."""
    equivalent = tmp_path / "same.jsonl"
    equivalent.write_text(
        '{"model_id": "claude-sonnet-4-5", "item_id": "rubric_0001", "correct": 0.87}\n'
    )
    with pytest.raises(RecordError, match="will not guess a cutoff") as from_jsonl:
        list(read_any(equivalent))
    with pytest.raises(RecordError, match="will not guess a cutoff") as from_csv:
        list(read_any(FIXTURES / "continuous_score.csv"))
    assert str(from_csv.value).split(" (")[0] == str(from_jsonl.value).split(" (")[0]


def test_a_missing_column_names_it(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("model_id,item_id\na,i1\n")
    with pytest.raises(RecordError, match="missing column"):
        list(read_any(path, fmt="csv"))


def test_an_empty_cell_is_missing_not_empty(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("model_id,item_id,correct,subject\na,i1,1,\na,,1,bio\n")
    first = next(iter(read_any(path, fmt="csv")))
    assert "subject" not in first.extra
    with pytest.raises(RecordError, match="missing required field"):
        list(read_any(path, fmt="csv"))


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("model_id,item_id,correct\na,i1,1\n\n\nb,i1,0\n")
    assert len(list(read_any(path, fmt="csv"))) == 2


def test_an_unknown_column_survives_for_the_respondent_key(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(
        "model_id,item_id,correct,quantization\n"
        "a,i1,1,int4\na,i1,0,int8\nb,i1,1,int4\nb,i1,0,int8\n"
    )
    matrix = build_matrix(
        read_any(path, fmt="csv"), respondent_key=("model_id", "quantization")
    )
    assert matrix.n_respondents == 4
    assert matrix.n_real_models == 2


def test_a_ragged_row_is_an_error_not_a_silent_shift(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("model_id,item_id,correct\na,i1,1,extra\n")
    with pytest.raises(RecordError, match="more cells"):
        list(read_any(path, fmt="csv"))


def test_item_ids_keep_their_leading_zeros(tmp_path):
    """`_number` is applied to `correct` and `raw_score` and nothing else on
    purpose: item "0042" and item "42" are different items."""
    path = tmp_path / "r.csv"
    path.write_text("model_id,item_id,correct\na,0042,1\n")
    assert list(read_any(path, fmt="csv"))[0].item_id == "0042"


def test_a_utf8_bom_does_not_become_part_of_the_first_column(tmp_path):
    path = tmp_path / "r.csv"
    path.write_bytes("model_id,item_id,correct\na,i1,1\n".encode("utf-8-sig"))
    assert list(read_any(path, fmt="csv"))[0].model_id == "a"
