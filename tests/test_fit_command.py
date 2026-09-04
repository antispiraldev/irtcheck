"""`irtcheck fit` end to end: a JSONL file in, a loadable artifact out."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

pytest.importorskip("pyro", reason="the fitter and its tests need torch and pyro")

from irtcheck.artifact import IrtFit  # noqa: E402
from irtcheck.cli import app  # noqa: E402
from irtcheck.synth import make_truth, responses_from_truth, synthetic_records  # noqa: E402

pytestmark = pytest.mark.slow

runner = CliRunner()


def write_jsonl(path, *, n_models=6, n_items=25, variants_per_model=1, seed=3):
    truth = make_truth(
        n_models=n_models, n_items=n_items, variants_per_model=variants_per_model, seed=seed
    )
    responses = responses_from_truth(truth, seed=seed + 1)
    with path.open("w", encoding="utf-8") as handle:
        for record in synthetic_records(truth, responses):
            handle.write(
                json.dumps(
                    {
                        "model_id": record.model_id,
                        "item_id": record.item_id,
                        "correct": record.correct,
                        **record.extra,
                    }
                )
                + "\n"
            )
    return truth


def test_fit_writes_an_artifact_report_can_read(tmp_path):
    responses = tmp_path / "responses.jsonl"
    truth = write_jsonl(responses)
    output = tmp_path / "suite.irt"

    result = runner.invoke(
        app, ["fit", str(responses), "-o", str(output), "--epochs", "200", "--seed", "0"]
    )
    assert result.exit_code == 0, result.output
    assert output.exists()

    fit = IrtFit.load(output)
    assert fit.n_items == truth.n_items
    assert fit.n_respondents == truth.n_respondents
    assert fit.diagnostics["epochs"] == 200
    assert fit.diagnostics["seed"] == 0
    assert fit.responses is not None
    # Passthrough fields survive the whole path, for `report` to group by.
    assert "subject" in fit.passthrough


def test_fit_creates_the_output_directory(tmp_path):
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, n_items=10)
    output = tmp_path / "nested" / "deeper" / "suite.irt"
    result = runner.invoke(app, ["fit", str(responses), "-o", str(output), "--epochs", "20"])
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_respondent_key_composes_pseudo_respondents(tmp_path):
    """Prompt variants are extra respondents to estimate with, and must not
    become extra models in the header."""
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, n_models=4, n_items=15, variants_per_model=3)
    output = tmp_path / "suite.irt"

    result = runner.invoke(
        app,
        [
            "fit",
            str(responses),
            "-o",
            str(output),
            "--respondent-key",
            "model_id,prompt_variant",
            "--epochs",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    fit = IrtFit.load(output)
    assert fit.n_respondents == 12
    assert fit.n_real_models == 4
    assert fit.respondent_key == ["model_id", "prompt_variant"]


def test_no_embed_responses_shrinks_the_artifact(tmp_path):
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, n_items=40)
    with_responses = tmp_path / "with.irt"
    without = tmp_path / "without.irt"

    for output, flag in ((with_responses, "--embed-responses"), (without, "--no-embed-responses")):
        result = runner.invoke(
            app, ["fit", str(responses), "-o", str(output), "--epochs", "20", flag]
        )
        assert result.exit_code == 0, result.output

    assert IrtFit.load(without).responses is None
    assert without.stat().st_size < with_responses.stat().st_size


def test_vague_priors_are_selectable_and_recorded(tmp_path):
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, n_items=15)
    output = tmp_path / "suite.irt"
    result = runner.invoke(
        app, ["fit", str(responses), "-o", str(output), "--priors", "vague", "--epochs", "20"]
    )
    assert result.exit_code == 0, result.output
    assert IrtFit.load(output).model["priors"] == "vague"


def test_an_unknown_prior_family_fails_with_a_useful_message(tmp_path):
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, n_items=10)
    result = runner.invoke(
        app,
        ["fit", str(responses), "-o", str(tmp_path / "s.irt"), "--priors", "psychic",
         "--epochs", "10"],
    )
    assert result.exit_code == 1
    assert "hierarchical" in result.output


def test_a_bad_respondent_key_fails_before_anything_slow(tmp_path):
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, n_items=10)
    result = runner.invoke(
        app,
        ["fit", str(responses), "-o", str(tmp_path / "s.irt"), "--respondent-key", "temperature"],
    )
    assert result.exit_code == 1
    assert "model_id" in result.output


def test_a_missing_input_file_is_reported_not_traced(tmp_path):
    result = runner.invoke(app, ["fit", str(tmp_path / "nope.jsonl"), "-o", str(tmp_path / "s.irt")])
    assert result.exit_code == 1
    assert "no such file" in result.output


def test_a_continuous_score_is_refused_rather_than_thresholded(tmp_path):
    """Coercing 0.87 to a 1 would be a silent change to what the model means."""
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        '{"model_id": "m1", "item_id": "i1", "correct": 0.87}\n', encoding="utf-8"
    )
    result = runner.invoke(app, ["fit", str(responses), "-o", str(tmp_path / "s.irt")])
    assert result.exit_code == 1
    assert "0 or 1" in result.output
