"""The input contract. Every adapter produces these, so an error here is an
error in every format at once."""

from __future__ import annotations

import pytest

from irtcheck.records import RecordError, build_record, coerce_correct


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, 1), (0, 0), (True, 1), (False, 0), (1.0, 1), (0.0, 0),
        ("1", 1), ("0", 0), ("true", 1), ("False", 0), ("YES", 1), ("n", 0),
    ],
)
def test_coerce_correct_accepts_the_usual_spellings(value, expected):
    assert coerce_correct(value) == expected


@pytest.mark.parametrize("value", [0.5, 2, -1, "maybe", None, [], 0.87])
def test_coerce_correct_refuses_everything_else(value):
    with pytest.raises(RecordError):
        coerce_correct(value)


def test_a_continuous_score_is_refused_with_a_reason():
    """0.87 is what a rubric score looks like. Guessing a cutoff would change
    what the model means without saying so."""
    with pytest.raises(RecordError, match="thresholded deliberately"):
        coerce_correct(0.87)


def test_error_names_the_line():
    with pytest.raises(RecordError, match=r"responses\.jsonl:42"):
        build_record({"model_id": "m"}, source="responses.jsonl", line=42)


def test_passthrough_and_unknown_fields_are_both_kept():
    record = build_record(
        {
            "model_id": "m", "item_id": "i", "correct": 1,
            "subject": "bio", "split": "test",
            "quantization": "q4",  # not a field irtcheck knows about
        }
    )
    assert record.extra["subject"] == "bio"
    # Kept so --respondent-key can name a field we have never heard of.
    assert record.extra["quantization"] == "q4"


def test_respondent_key_composes_across_fields():
    record = build_record(
        {"model_id": "m", "item_id": "i", "correct": 1, "prompt_variant": "v2"}
    )
    assert record.key(("model_id",)) == "m"
    assert record.key(("model_id", "prompt_variant")) == "m|v2"


def test_respondent_key_naming_a_missing_field_says_what_is_available():
    record = build_record({"model_id": "m", "item_id": "i", "correct": 1, "split": "test"})
    with pytest.raises(RecordError, match="Available: split"):
        record.key(("model_id", "prompt_variant"))


def test_missing_required_fields_are_all_reported_at_once():
    with pytest.raises(RecordError, match="item_id, correct"):
        build_record({"model_id": "m"})
