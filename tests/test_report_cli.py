"""`irtcheck report` end to end, through the real CLI.

These go through cli.py rather than calling `run` directly, because the frozen
flag declarations are half of this command's contract: --sort, --limit, --json
and --html are declared there and this brief only implements what they promise.
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from irtcheck.cli import app
from irtcheck.synth import synthetic_fit

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """rich wraps to the terminal width; 80 columns would split the sentences
    these tests assert on. Width is presentation, not behaviour.

    Do not set TERM=dumb here: rich pins a dumb terminal to 80 columns and
    ignores COLUMNS entirely."""
    monkeypatch.setenv("COLUMNS", "160")


@pytest.fixture(scope="module")
def healthy_path(tmp_path_factory):
    fit, _ = synthetic_fit(n_models=120, n_items=200, precision=1.5, seed=3)
    return fit.save(tmp_path_factory.mktemp("healthy") / "suite.irt")


@pytest.fixture(scope="module")
def thin_path(tmp_path_factory):
    fit, _ = synthetic_fit(
        n_models=5, variants_per_model=3, n_items=200, precision=0.5, seed=7
    )
    return fit.save(tmp_path_factory.mktemp("thin") / "suite.irt")


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def flat(text: str) -> str:
    """Plain, unwrapped text: styling and line breaks are presentation."""
    return re.sub(r"\s+", " ", ANSI.sub("", text))


def invoke(*args):
    return runner.invoke(app, ["report", *(str(a) for a in args)])


# -- the happy path -----------------------------------------------------------


def test_report_renders(healthy_path):
    result = invoke(healthy_path, "--limit", "5")
    assert result.exit_code == 0, result.output
    out = flat(result.output)
    assert "120 real models" in out
    assert "200 items" in out
    assert "item_" in out


@pytest.mark.parametrize("sort", ["discrimination", "difficulty", "id"])
def test_every_documented_sort_works(healthy_path, sort):
    result = invoke(healthy_path, "--sort", sort, "--limit", "5")
    assert result.exit_code == 0, result.output
    assert f"sorted by {sort}" in flat(result.output)


def test_limit_zero_prints_every_item(healthy_path):
    result = invoke(healthy_path, "--limit", "0")
    assert result.exit_code == 0
    assert "showing" not in flat(result.output)
    assert "200 items, sorted by" in flat(result.output)


# -- the refusal --------------------------------------------------------------


def test_thin_data_leads_with_the_refusal_and_names_respondent_key(thin_path):
    result = invoke(thin_path, "--limit", "5")
    assert result.exit_code == 0, result.output
    out = flat(result.output)
    assert "Not enough respondents to rank these items" in out
    assert "--respondent-key model_id,prompt_variant,temperature" in out
    assert out.index("Not enough respondents") < out.index("real models")


def test_the_header_never_reports_pseudo_respondents_as_models(thin_path):
    out = flat(invoke(thin_path, "--limit", "1").output)
    assert "5 real models" in out
    assert "15 respondents" in out
    assert "15 real models" not in out


# -- --json -------------------------------------------------------------------


def test_json_is_parseable_and_complete(thin_path):
    result = invoke(thin_path, "--json", "--limit", "3")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["items_shown"] == 3
    assert payload["refusal"]["level"] == "refusal"
    assert payload["header"]["n_real_models"] == 5
    assert payload["header"]["n_respondents"] == 15
    for record in payload["items"]:
        assert set(payload["columns"]) <= record.keys()


def test_json_carries_what_the_table_shows(thin_path):
    """A user scripting against this should not have to parse the table."""
    table = flat(invoke(thin_path, "--limit", "3").output)
    payload = json.loads(invoke(thin_path, "--json", "--limit", "3").output)
    for record in payload["items"]:
        assert record["item_id"] in table
        assert f"{record['a']:.2f}" in table
        assert f"[{record['a_hdi_low']:.2f}, {record['a_hdi_high']:.2f}]" in table
        for flag in record["flags"]:
            assert flag in table


def test_json_emits_nothing_but_json(healthy_path):
    """Piped into jq, so no rich panel may precede it — including the refusal."""
    result = invoke(healthy_path, "--json", "--limit", "2")
    assert result.output.lstrip().startswith("{")
    json.loads(result.output)


# -- --html -------------------------------------------------------------------
#
# Was "flags this brief does not implement": --html exited 2 and named wave 2,
# brief E. That brief has landed, so the stub assertion is gone and the flag's
# behaviour is asserted in tests/test_html.py, which owns the renderer. Kept
# here: that the terminal report still prints when --html is also passed, which
# is what "Also write a self-contained HTML report here" promises.


def test_html_writes_the_file_and_still_prints_the_table(healthy_path, tmp_path):
    target = tmp_path / "out.html"
    result = invoke(healthy_path, "--html", target, "--limit", "3")
    assert result.exit_code == 0, result.output
    assert target.exists()
    assert "item" in flat(result.output)


# -- failure modes ------------------------------------------------------------


def test_a_missing_artifact_is_a_clean_error(tmp_path):
    result = invoke(tmp_path / "nope.irt")
    assert result.exit_code == 2
    assert "cannot read" in flat(result.output)


def test_a_file_that_is_not_an_artifact_is_a_clean_error(tmp_path):
    junk = tmp_path / "junk.irt"
    junk.write_bytes(b"not gzip, not json")
    result = invoke(junk)
    assert result.exit_code == 2
    assert "junk.irt" in flat(result.output)
    assert "Traceback" not in result.output


def test_an_unknown_sort_exits_with_the_valid_choices(healthy_path):
    result = invoke(healthy_path, "--sort", "vibes")
    assert result.exit_code == 2
    assert "discrimination, difficulty, id" in flat(result.output)


def test_a_negative_limit_exits_cleanly(healthy_path):
    result = invoke(healthy_path, "--limit", "-3")
    assert result.exit_code == 2
    assert "0 for every item" in flat(result.output)


def test_an_artifact_from_the_future_is_rejected_not_half_read(tmp_path):
    fit, _ = synthetic_fit(n_models=5, n_items=10, seed=0)
    payload = fit.to_dict()
    payload["schema_version"] = 999
    path = tmp_path / "future.irt"
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(payload).encode("utf-8"))
    result = invoke(path)
    assert result.exit_code == 2
    assert "schema_version" in flat(result.output)


# -- the boundary this command exists to keep ---------------------------------


def test_reporting_never_imports_torch(healthy_path):
    """The whole point of the cached artifact: analysing a fit works on a
    machine that has never installed torch. Run out of process because the
    session may already have imported it."""
    code = (
        "import sys;"
        "from typer.testing import CliRunner;"
        "from irtcheck.cli import app;"
        f"r = CliRunner().invoke(app, ['report', {str(healthy_path)!r}, '--limit', '3']);"
        "sys.exit(2 if r.exit_code else (1 if {'torch', 'pyro'} & set(sys.modules) else 0))"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
