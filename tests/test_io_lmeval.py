"""The lm-eval-harness adapter.

Fixtures under `fixtures/adapters/lmeval/run/` are trimmed real logs: two
models, two tasks, laid out the way lm-eval lays them out
(`<output_path>/<model_name_sanitized>/`). One model's directory has the
`results_*.json` lm-eval writes beside the samples and one does not, so a
single read exercises both ways the adapter can learn a model id.

`test_lmeval_matches_the_reference_jsonl` is the one that proves faithfulness:
the same responses, hand-written as reference JSONL, must build the same
matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from irtcheck.io import FormatError, read_any, readers
from irtcheck.matrix import build_matrix
from irtcheck.records import RecordError

FIXTURES = Path(__file__).parent / "fixtures" / "adapters" / "lmeval"
RUN = FIXTURES / "run"
PYTHIA = RUN / "EleutherAI__pythia-14m"
GSM8K = PYTHIA / "samples_gsm8k_2026-05-01T12-00-00.000000.jsonl"
MMLU = PYTHIA / "samples_mmlu_high_school_biology_2026-05-01T12-00-00.000000.jsonl"
MULTI_FILTER = FIXTURES / "multi_filter"


def assert_same_matrix(left, right) -> None:
    assert left.respondent_ids == right.respondent_ids
    assert left.item_ids == right.item_ids
    assert left.derives_from == right.derives_from
    assert left.rows.tolist() == right.rows.tolist()
    assert left.cols.tolist() == right.cols.tolist()
    assert left.obs.tolist() == right.obs.tolist()
    assert left.passthrough == right.passthrough


def test_lmeval_is_registered():
    assert "lmeval" in readers()


def test_lmeval_matches_the_reference_jsonl():
    from_logs = build_matrix(read_any(RUN, fmt="lmeval"))
    from_jsonl = build_matrix(read_any(FIXTURES / "reference.jsonl"))
    assert_same_matrix(from_logs, from_jsonl)
    assert from_logs.n_respondents == 2
    assert from_logs.n_items == 4


def test_item_ids_are_task_prefixed_because_doc_ids_collide():
    """doc_id 0 exists in every task. Without the task prefix, gsm8k's first
    problem and MMLU's first question would be one item."""
    items = build_matrix(read_any(RUN, fmt="lmeval")).item_ids
    assert items == [
        "gsm8k_0",
        "gsm8k_1",
        "mmlu_high_school_biology_0",
        "mmlu_high_school_biology_1",
    ]


def test_the_model_id_comes_from_the_results_file():
    records = list(read_any(GSM8K, fmt="lmeval"))
    assert {r.model_id for r in records} == {"EleutherAI/pythia-14m"}


def test_the_model_id_falls_back_to_the_directory_lm_eval_named():
    qwen = RUN / "Qwen__Qwen2-0.5B" / "samples_gsm8k_2026-05-01T12-30-00.000000.jsonl"
    records = list(read_any(qwen, fmt="lmeval"))
    assert {r.model_id for r in records} == {"Qwen__Qwen2-0.5B"}


def test_an_explicit_model_id_wins(monkeypatch):
    monkeypatch.setenv("IRTCHECK_LMEVAL_MODEL_ID", "pythia-14m@step3000")
    assert next(iter(read_any(GSM8K, fmt="lmeval"))).model_id == "pythia-14m@step3000"


def test_acc_is_preferred_over_acc_norm():
    """MMLU logs both. They disagree on doc 1 — acc 0, acc_norm 1 — so which
    one the adapter picks is visible in the matrix, not a matter of taste."""
    records = list(read_any(MMLU, fmt="lmeval"))
    assert [r.correct for r in records] == [1, 0]
    assert {r.extra["metric"] for r in records} == {"acc"}


def test_the_metric_can_be_chosen_explicitly(monkeypatch):
    monkeypatch.setenv("IRTCHECK_LMEVAL_METRIC", "acc_norm")
    assert [r.correct for r in read_any(MMLU, fmt="lmeval")] == [1, 1]


def test_an_unknown_metric_is_named_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setenv("IRTCHECK_LMEVAL_MODEL_ID", "m")
    path = tmp_path / "samples_squad_2026-05-01T12-00-00.jsonl"
    path.write_text(
        json.dumps({"doc_id": 0, "metrics": ["f1", "bleu"], "f1": 0.8, "bleu": 0.3}) + "\n"
    )
    with pytest.raises(RecordError, match="IRTCHECK_LMEVAL_METRIC"):
        list(read_any(path, fmt="lmeval"))


def test_a_continuous_metric_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("IRTCHECK_LMEVAL_MODEL_ID", "m")
    path = tmp_path / "samples_squad_2026-05-01T12-00-00.jsonl"
    path.write_text(json.dumps({"doc_id": 0, "metrics": ["f1"], "f1": 0.8}) + "\n")
    with pytest.raises(RecordError, match="will not guess a cutoff"):
        list(read_any(path, fmt="lmeval"))


def test_one_filter_is_kept_and_the_drop_is_announced():
    """gsm8k logs every doc once per filter. Those are the same items scored
    two ways; keeping both would make each item look like two items whose
    answers happen to correlate almost perfectly."""
    with pytest.warns(UserWarning, match="flexible-extract"):
        records = list(read_any(MULTI_FILTER, fmt="lmeval"))
    assert [(r.item_id, r.correct) for r in records] == [("gsm8k_0", 0), ("gsm8k_1", 1)]
    assert {r.extra["filter"] for r in records} == {"strict-match"}


def test_the_filter_can_be_chosen_explicitly(monkeypatch):
    monkeypatch.setenv("IRTCHECK_LMEVAL_FILTER", "flexible-extract")
    with pytest.warns(UserWarning, match="strict-match"):
        records = list(read_any(MULTI_FILTER, fmt="lmeval"))
    assert [(r.item_id, r.correct) for r in records] == [("gsm8k_0", 1), ("gsm8k_1", 1)]


def test_a_filter_that_matched_nothing_is_an_error_not_an_empty_read(monkeypatch):
    monkeypatch.setenv("IRTCHECK_LMEVAL_FILTER", "get-answer")
    with pytest.raises(RecordError, match="matched nothing"):
        list(read_any(MULTI_FILTER, fmt="lmeval"))


def test_the_subject_is_the_doc_subject_then_the_task():
    by_item = {r.item_id: r.extra["subject"] for r in read_any(RUN, fmt="lmeval")}
    assert by_item["gsm8k_0"] == "gsm8k"
    assert by_item["mmlu_high_school_biology_0"] == "high_school_biology"


def test_a_samples_log_is_sniffed_as_lmeval_not_as_the_reference_format():
    """Both formats are .jsonl, so the extension cannot decide. An lm-eval
    record has doc_id and no model_id; a reference record is the other way
    round."""
    records = list(read_any(GSM8K))
    assert records[0].item_id == "gsm8k_0"
    assert list(read_any(FIXTURES / "reference.jsonl"))[0].item_id == "gsm8k_0"


def test_reading_the_reference_format_as_lmeval_says_what_is_missing():
    with pytest.raises(RecordError, match="log_samples"):
        list(read_any(FIXTURES / "reference.jsonl", fmt="lmeval"))


def test_a_directory_with_no_sample_logs_says_to_re_run(tmp_path):
    (tmp_path / "results_2026-05-01T12-00-00.json").write_text("{}")
    with pytest.raises(RecordError, match="--log_samples"):
        list(read_any(tmp_path, fmt="lmeval"))


def test_a_missing_path_is_still_a_format_error(tmp_path):
    with pytest.raises(FormatError, match="no such file"):
        list(read_any(tmp_path / "nope", fmt="lmeval"))
