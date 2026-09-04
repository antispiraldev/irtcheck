"""The Inspect adapter.

Fixtures under `fixtures/adapters/inspect_ai/` are two trimmed `.json` logs
(one task, two models) plus the same log as `.eval` in both compressions —
see `make_eval_fixtures.py` for how the binaries are produced.

`test_inspect_matches_the_reference_jsonl` is the faithfulness test: the same
responses hand-written as reference JSONL must build the same matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from irtcheck.io import FormatError, read_any, readers
from irtcheck.matrix import MatrixError, build_matrix
from irtcheck.records import RecordError

FIXTURES = Path(__file__).parent / "fixtures" / "adapters" / "inspect_ai"
LOGS = FIXTURES / "logs"
GPT = LOGS / "2026-05-01T12-00-00+00-00_arc-easy_aTGeC2v9.json"
EDGE = FIXTURES / "edge"


def assert_same_matrix(left, right) -> None:
    assert left.respondent_ids == right.respondent_ids
    assert left.item_ids == right.item_ids
    assert left.derives_from == right.derives_from
    assert left.rows.tolist() == right.rows.tolist()
    assert left.cols.tolist() == right.cols.tolist()
    assert left.obs.tolist() == right.obs.tolist()
    assert left.passthrough == right.passthrough


def triples(records):
    return {(r.model_id, r.item_id, r.correct) for r in records}


def test_inspect_ai_is_registered_under_that_name():
    """`inspect` is a stdlib module; a module here shadowing it is a trap with
    no upside, and the --format name follows the module name."""
    assert "inspect_ai" in readers()
    assert "inspect" not in readers()


def test_inspect_matches_the_reference_jsonl():
    from_logs = build_matrix(read_any(LOGS, fmt="inspect_ai"))
    from_jsonl = build_matrix(read_any(FIXTURES / "reference.jsonl"))
    assert_same_matrix(from_logs, from_jsonl)
    assert from_logs.n_respondents == 2
    assert from_logs.n_items == 3


def test_item_ids_are_task_prefixed():
    assert build_matrix(read_any(LOGS, fmt="inspect_ai")).item_ids == [
        "arc_easy_Mercury_7175875",
        "arc_easy_Mercury_SC_405487",
        "arc_easy_MCAS_2000_4_6",
    ]


def test_a_json_log_is_sniffed_without_being_told():
    assert triples(read_any(GPT)) == triples(read_any(GPT, fmt="inspect_ai"))


def test_a_deflate_eval_zip_reads_the_same_as_its_json():
    """Old Inspect deflate-compressed `.eval`; the stdlib reads that, so we do
    too. Zip members come back in name order rather than sample order, which
    is why this compares content and not sequence."""
    assert triples(read_any(FIXTURES / "deflate.eval")) == triples(read_any(GPT))


def test_a_zstandard_eval_is_refused_with_the_conversion_command():
    """Current Inspect uses zstandard, which Python's zipfile cannot inflate
    before 3.14 and which irtcheck may not add a dependency for. Saying so —
    with the command that fixes it — is the whole feature."""
    with pytest.raises(FormatError, match="inspect log convert") as refusal:
        list(read_any(FIXTURES / "zstandard.eval"))
    assert "zstandard" in str(refusal.value)


def test_grades_bools_and_numbers_all_reach_zero_or_one():
    records = list(read_any(EDGE / "value_shapes.json", fmt="inspect_ai"))
    assert [r.correct for r in records] == [1, 0, 1, 0, 0]
    assert {r.extra["subject"] for r in records} == {"arithmetic"}


def test_no_answer_is_scored_the_way_the_harness_scores_it():
    """Inspect's own value_to_float maps "N" to 0, so the accuracy the user
    already reports counts a refusal as wrong. Reading it any other way would
    put irtcheck's matrix at odds with the number it is explaining."""
    records = list(read_any(EDGE / "value_shapes.json", fmt="inspect_ai"))
    assert records[-1].correct == 0


def test_partial_credit_is_refused():
    with pytest.raises(RecordError, match="partial credit"):
        list(read_any(EDGE / "partial_and_continuous.json", fmt="inspect_ai"))


def test_a_continuous_score_is_refused_without_a_threshold(tmp_path):
    log = json.loads((EDGE / "partial_and_continuous.json").read_text())
    log["samples"] = log["samples"][1:]  # drop the "P" sample; keep the 0.72 one
    path = tmp_path / "continuous_only.json"
    path.write_text(json.dumps(log))
    with pytest.raises(RecordError, match="will not guess a cutoff"):
        list(read_any(path, fmt="inspect_ai"))


def test_two_scorers_are_named_rather_than_picked_between():
    with pytest.raises(RecordError, match="IRTCHECK_INSPECT_SCORER"):
        list(read_any(EDGE / "two_scorers.json", fmt="inspect_ai"))


def test_the_scorer_can_be_chosen_explicitly(monkeypatch):
    monkeypatch.setenv("IRTCHECK_INSPECT_SCORER", "model_graded_qa")
    records = list(read_any(EDGE / "two_scorers.json", fmt="inspect_ai"))
    assert [r.correct for r in records] == [0]


def test_an_unscored_sample_is_a_missing_response_not_a_wrong_one():
    with pytest.warns(UserWarning, match="missing observations"):
        records = list(read_any(EDGE / "epochs_and_errors.json", fmt="inspect_ai"))
    assert len(records) == 3


def test_epochs_are_repeats_of_an_item_not_new_items():
    """--epochs 3 answers each item three times. Under the default respondent
    key that is a duplicate response and build_matrix says so; the honest
    reading is to make the repeats pseudo-respondents."""
    with pytest.warns(UserWarning):
        records = list(read_any(EDGE / "epochs_and_errors.json", fmt="inspect_ai"))
    assert {r.extra["epoch"] for r in records} == {1, 2}
    with pytest.raises(MatrixError, match="more than once"):
        build_matrix(records)
    matrix = build_matrix(records, respondent_key=("model_id", "epoch"))
    assert matrix.n_respondents == 2
    assert matrix.n_real_models == 1


def test_a_header_only_log_says_so(tmp_path):
    path = tmp_path / "header_only.json"
    path.write_text(json.dumps({"eval": {"task": "t", "model": "m"}, "status": "success"}))
    with pytest.raises(RecordError, match="header-only"):
        list(read_any(path, fmt="inspect_ai"))


def test_a_zip_that_is_not_an_inspect_log_is_not_claimed(tmp_path):
    import zipfile

    path = tmp_path / "notalog.eval"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(FormatError, match="not an Inspect .eval log"):
        list(read_any(path, fmt="inspect_ai"))


def test_a_directory_of_logs_reads_every_model():
    models = {r.model_id for r in read_any(LOGS, fmt="inspect_ai")}
    assert models == {"openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"}
