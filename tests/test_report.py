"""The report's claims, tested against artifacts with known ground truth.

Two artifacts recur here and they are the two cases the brief is about: a
*healthy* fit with enough respondents that item parameters are pinned down, and
a *thin* one at the respondent count a real user actually has. The thin one is
not an error fixture — it is the ordinary case, and most of what is asserted
below is about the report being legible and honest in it rather than looking
broken.

No torch anywhere: every fixture comes from synth.synthetic_fit, which
fabricates a valid IrtFit from known parameters with numpy alone.
"""

from __future__ import annotations

import json
import pathlib
import re
import tempfile

import pytest
from rich.console import Console

from irtcheck.artifact import (
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
    IrtFit,
)
from irtcheck.report import (
    CAUTION_SHARE,
    COLUMNS,
    REFUSAL_SHARE,
    Column,
    ItemRow,
    ReportError,
    build_header,
    build_refusal,
    build_report,
    item_rows,
    render,
    row_json,
    sort_rows,
    suggested_respondent_key,
    summarise_diagnostics,
)
from irtcheck.synth import synthetic_fit

# Enough respondents that intervals are narrow: some items reach `dead`, few
# land in insufficient-data. This is what a hundred-plus models buys.
HEALTHY = dict(n_models=120, n_items=200, precision=1.5, seed=3)

# Five models behind fifteen pseudo-respondents — the shape the spec assumes,
# and the shape in which most items are honestly unrankable.
THIN = dict(n_models=5, variants_per_model=3, n_items=200, precision=0.5, seed=7)


@pytest.fixture(scope="module")
def healthy() -> IrtFit:
    fit, _ = synthetic_fit(**HEALTHY)
    return fit


@pytest.fixture(scope="module")
def thin() -> IrtFit:
    fit, _ = synthetic_fit(**THIN)
    return fit


def text_of(renderable_call) -> str:
    console = Console(width=160, record=True, no_color=True, highlight=False)
    renderable_call(console)
    return console.export_text()


def _tmp_path_for(fit: IrtFit):
    """A throwaway path to round-trip a fixture through, so mutating a copy's
    diagnostics cannot leak into the module-scoped fixture other tests share."""
    return pathlib.Path(tempfile.mkdtemp()) / "copy.irt"


def flat(text: str) -> str:
    """Collapse rich's line wrapping so a sentence can be matched as a sentence."""
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", text))


def _row(item_id: str, *, a_mean: float, hdi: tuple[float, float], flags: list[str]) -> ItemRow:
    """A hand-built row, for ordering rules that need an exact arrangement."""
    return ItemRow(
        index=0,
        item_id=item_id,
        a_mean=a_mean,
        a_sd=(hdi[1] - hdi[0]) / 4,
        a_hdi_low=hdi[0],
        a_hdi_high=hdi[1],
        b_mean=0.0,
        b_sd=0.5,
        b_hdi_low=-1.0,
        b_hdi_high=1.0,
        n_resp=10,
        p_correct=0.5,
        flags=list(flags),
    )


# -- the two fixtures are the two cases ---------------------------------------


def test_the_fixtures_are_the_cases_the_brief_describes(healthy, thin):
    healthy_share = build_header(healthy).insufficient_share
    thin_share = build_header(thin).insufficient_share
    assert healthy_share < CAUTION_SHARE
    assert thin_share >= REFUSAL_SHARE
    # dead is a finding a healthy fit can reach and a thin one mostly cannot.
    assert len(healthy.flagged(FLAG_DEAD)) > 0


# -- header -------------------------------------------------------------------


def test_header_counts_real_models_not_respondents(thin):
    header = build_header(thin)
    assert header.n_real_models == 5
    assert header.n_respondents == 15
    assert header.has_pseudo_respondents
    assert header.respondents_per_model == pytest.approx(3.0)


def test_header_renders_real_models_and_makes_the_inflation_obvious(thin):
    report = build_report(thin, limit=3)
    out = flat(text_of(lambda c: render(report, c)))
    assert "5 real models" in out
    assert "15 respondents" in out
    assert "3.0 per model" in out
    assert "model_id,prompt_variant" in out


def test_header_without_pseudo_respondents_says_one_each(healthy):
    report = build_report(healthy, limit=1)
    out = flat(text_of(lambda c: render(report, c)))
    assert "120 real models" in out
    assert "one respondent each" in out


def test_header_carries_every_field_the_spec_requires(thin):
    header = build_header(thin)
    assert header.n_items == 200
    assert header.n_usable_items == len(thin.usable_items())
    assert header.dead_count == len(thin.flagged(FLAG_DEAD))
    assert header.insufficient_count == len(thin.flagged(FLAG_INSUFFICIENT_DATA))
    assert header.ceiling_rate == len(thin.flagged(FLAG_CEILING)) / 200
    assert header.floor_rate == len(thin.flagged(FLAG_FLOOR)) / 200
    assert header.diagnostics["seed"] == THIN["seed"]
    assert header.model["kind"] == "2pl"


def test_dead_and_insufficient_data_are_never_merged(thin):
    """Two different claims: one about the suite, one about the data. A single
    'bad items' count would erase the distinction the tool exists to make."""
    header = build_header(thin)
    out = flat(text_of(lambda c: render(build_report(thin, limit=3), c)))
    assert f"dead {header.dead_count}" in out
    assert f"insufficient-data {header.insufficient_count}" in out
    # and the two are described as different kinds of statement
    assert "confidently do not discriminate" in out
    assert "cannot tell" in out
    merged = header.dead_count + header.insufficient_count
    assert f"{merged} bad" not in out


def test_a_thresholded_raw_score_is_printed_in_the_header(thin):
    """CLAUDE.md: thresholding a continuous score is a documented escape hatch
    that has to print its threshold, not a quiet cast."""
    thresholded = IrtFit.from_dict(
        {**thin.to_dict(), "diagnostics": {**thin.diagnostics, "score_threshold": 0.6}}
    )
    assert build_header(thresholded).score_threshold == 0.6
    out = flat(text_of(lambda c: render(build_report(thresholded, limit=1), c)))
    assert "binarised at 0.6" in out


# -- refusal ------------------------------------------------------------------


def test_thin_data_refuses_and_names_respondent_key(thin):
    refusal = build_refusal(build_header(thin))
    assert refusal.level == "refusal"
    assert "insufficient-data" in refusal.headline
    assert "--respondent-key" in refusal.next_step
    assert refusal.suggested_respondent_key.startswith("model_id,prompt_variant")


def test_the_refusal_is_printed_above_everything_else(thin):
    out = text_of(lambda c: render(build_report(thin, limit=5), c))
    refusal_at = out.index("Not enough respondents")
    header_at = out.index("real models")
    first_item_at = out.index(build_report(thin, limit=5).rows[0].item_id)
    assert refusal_at < header_at < first_item_at


def test_the_refusal_names_a_concrete_next_command(thin):
    out = flat(text_of(lambda c: render(build_report(thin, limit=1), c)))
    assert "--respondent-key model_id,prompt_variant,temperature" in out
    assert "irtcheck fit" in out


def test_a_healthy_fit_does_not_refuse(healthy):
    report = build_report(healthy, limit=5)
    assert report.refusal.level == "none"
    out = text_of(lambda c: render(report, c))
    assert "Not enough respondents" not in out


def test_zero_dead_items_is_explained_rather_than_left_looking_broken(thin):
    """At 5-15 models `dead` is rare by construction. A reader who sees 0 dead
    and 110 unrankable should be told that is the expected shape."""
    refusal = build_refusal(build_header(thin))
    detail = " ".join(refusal.detail)
    if build_header(thin).dead_count == 0:
        assert "expected rather than suspicious" in detail
    assert "statement about the data supplied" in detail


def test_too_few_real_models_cautions_even_when_most_items_are_rankable():
    fit, _ = synthetic_fit(n_models=3, n_items=60, precision=8.0, seed=1)
    header = build_header(fit)
    assert header.insufficient_share < CAUTION_SHARE  # the items look fine
    refusal = build_refusal(header)
    assert refusal.level == "caution"  # the respondents do not
    assert "below the range this tool assumes" in " ".join(refusal.detail)


def test_a_caution_is_a_footer_line_not_a_top_panel():
    fit, _ = synthetic_fit(n_models=3, n_items=60, precision=8.0, seed=1)
    out = flat(text_of(lambda c: render(build_report(fit, limit=3), c)))
    assert "cannot rank these items yet" not in out
    assert "--respondent-key" in out  # still says how to fix it


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (["model_id"], "model_id,prompt_variant"),
        (["model_id", "prompt_variant"], "model_id,prompt_variant,temperature"),
        ([], "model_id,prompt_variant"),
    ],
)
def test_the_suggested_key_extends_the_one_that_produced_the_fit(current, expected):
    """model_id has to stay in the key or leave-one-model-out cannot hold a
    model's pseudo-respondents out together."""
    suggestion = suggested_respondent_key(current)
    assert suggestion == expected
    assert suggestion.split(",")[0] == "model_id"


# -- sorting and limiting -----------------------------------------------------


def test_discrimination_sort_puts_unrankable_items_last(thin):
    rows = sort_rows(item_rows(thin), "discrimination")
    rankable = [r.rankable for r in rows]
    assert rankable == sorted(rankable, reverse=True), "insufficient-data must sort last"
    ranked = [r.a_mean for r in rows if r.rankable]
    assert ranked == sorted(ranked, reverse=True)


def test_a_flattering_mean_does_not_buy_an_unrankable_item_a_high_rank():
    """The failure mode this rule exists for: a wide interval whose mean is the
    largest in the suite would otherwise take the top row of the table."""
    rows = [
        _row("loud-but-unknown", a_mean=4.0, hdi=(-0.5, 8.5), flags=[FLAG_INSUFFICIENT_DATA]),
        _row("modest-but-known", a_mean=1.2, hdi=(0.9, 1.5), flags=[]),
    ]
    assert [r.item_id for r in sort_rows(rows, "discrimination")] == [
        "modest-but-known",
        "loud-but-unknown",
    ]


def test_difficulty_sort_is_easy_to_hard(healthy):
    rows = sort_rows(item_rows(healthy), "difficulty")
    assert [r.b_mean for r in rows] == sorted(r.b_mean for r in rows)


def test_id_sort_is_alphabetical(healthy):
    rows = sort_rows(item_rows(healthy), "id")
    assert [r.item_id for r in rows] == sorted(r.item_id for r in rows)


def test_sorting_is_total_so_output_is_reproducible(healthy):
    for key in ("discrimination", "difficulty", "id"):
        first = [r.item_id for r in sort_rows(item_rows(healthy), key)]
        second = [r.item_id for r in sort_rows(item_rows(healthy), key)]
        assert first == second


def test_unknown_sort_names_the_valid_choices(healthy):
    with pytest.raises(ReportError, match="discrimination, difficulty, id"):
        build_report(healthy, sort="a-mean")


def test_limit_zero_means_every_item(healthy):
    assert len(build_report(healthy, limit=0).rows) == healthy.n_items


def test_limit_truncates_and_the_footer_says_so(healthy):
    report = build_report(healthy, limit=7)
    assert len(report.rows) == 7
    out = flat(text_of(lambda c: render(report, c)))
    assert "showing 7 of 200 items" in out
    assert "--limit 0 for all" in out


def test_limit_beyond_the_suite_is_not_an_error(healthy):
    assert len(build_report(healthy, limit=10_000).rows) == healthy.n_items


def test_negative_limit_is_rejected(healthy):
    with pytest.raises(ReportError, match="0 for every item"):
        build_report(healthy, limit=-1)


# -- the table ----------------------------------------------------------------


def test_table_shows_every_declared_column(healthy):
    out = text_of(lambda c: render(build_report(healthy, limit=3), c))
    for column in COLUMNS:
        assert column.heading in out


def test_table_rows_carry_intervals_and_flags(thin):
    report = build_report(thin, sort="id", limit=200)
    out = text_of(lambda c: render(report, c))
    row = next(r for r in report.rows if FLAG_INSUFFICIENT_DATA in r.flags)
    line = next(line for line in out.splitlines() if line.strip().startswith(row.item_id))
    assert f"{row.a_mean:.2f}" in line
    assert f"[{row.a_hdi_low:.2f}, {row.a_hdi_high:.2f}]" in line
    assert FLAG_INSUFFICIENT_DATA in line
    assert str(row.n_resp) in line


# -- --json -------------------------------------------------------------------


def test_json_carries_every_column_the_table_shows(thin):
    payload = build_report(thin, limit=5).to_json()
    for record in payload["items"]:
        for column in COLUMNS:
            for key in column.keys:
                assert key in record, f"column {column.heading!r} is missing from --json"


def test_json_columns_list_matches_the_table(thin):
    payload = build_report(thin, limit=5).to_json()
    assert payload["columns"] == [key for column in COLUMNS for key in column.keys]


def test_json_values_are_the_values_the_table_rendered(thin):
    report = build_report(thin, limit=5)
    payload = report.to_json()
    for row, record in zip(report.rows, payload["items"], strict=True):
        for column in COLUMNS:
            rendered = column.render(row)
            for key, value in zip(column.keys, column.values(row), strict=True):
                assert record[key] == value
            if isinstance(record[column.keys[0]], float):
                assert f"{record[column.keys[0]]:.2f}".lstrip("+-") in rendered.replace("+", "")


def test_json_is_serialisable_and_round_trips(thin):
    payload = build_report(thin, limit=5).to_json()
    assert json.loads(json.dumps(payload)) == payload


def test_json_item_order_is_the_table_order(healthy):
    report = build_report(healthy, sort="difficulty", limit=12)
    payload = report.to_json()
    assert [r["item_id"] for r in payload["items"]] == [r.item_id for r in report.rows]


def test_json_says_how_much_was_shown(healthy):
    payload = build_report(healthy, limit=12).to_json()
    assert payload["items_shown"] == 12
    assert payload["items_total"] == healthy.n_items
    assert payload["limit"] == 12
    assert payload["sort"] == "discrimination"


def test_json_header_keeps_dead_and_insufficient_apart(thin):
    header = build_report(thin, limit=1).to_json()["header"]
    assert header["dead_count"] == len(thin.flagged(FLAG_DEAD))
    assert header["insufficient_data_count"] == len(thin.flagged(FLAG_INSUFFICIENT_DATA))
    assert "bad_items" not in header
    assert header["n_real_models"] == 5 and header["n_respondents"] == 15


def test_json_carries_the_refusal_so_a_script_can_act_on_it(thin):
    payload = build_report(thin, limit=1).to_json()
    assert payload["refusal"]["level"] == "refusal"
    assert "--respondent-key" in payload["refusal"]["next_step"]
    assert payload["refusal"]["suggested_respondent_key"]


def test_row_json_covers_the_extras_a_script_wants(thin):
    record = row_json(item_rows(thin)[0])
    assert {"a_sd", "b_sd", "rankable", "dead"} <= record.keys()


def test_every_column_declares_at_least_one_json_key():
    for column in COLUMNS:
        assert isinstance(column, Column)
        assert column.keys, f"{column.heading} would be invisible to --json"


# -- the report is a summary, not the artifact -------------------------------


def _with_history(fit: IrtFit, points: int = 2000) -> IrtFit:
    """The same fit, carrying a full-length ELBO trace like a real one does.

    synth.py fabricates artifacts without a trace, so the flood this guards
    against is invisible to every other fixture in this file — which is exactly
    how it shipped.
    """
    fit.diagnostics["elbo_history"] = [-400.0 + i * 0.1 for i in range(points)]
    fit.diagnostics["elbo_history_stride"] = 1
    fit.diagnostics["elbo_final"] = fit.diagnostics["elbo_history"][-1]
    return fit


def test_the_elbo_trace_is_not_printed(thin):
    """A report must not paste the fit's telemetry into the terminal.

    Measured before the fix: 1127 lines of output for a 30-item suite, 1056 of
    them raw ELBO values, putting the item table below the fold. The trace
    belongs in the artifact; the report says how many points it has.
    """
    fit = _with_history(IrtFit.load(thin.save(_tmp_path_for(thin))))
    body = flat(text_of(lambda c: render(build_report(fit), c)))
    assert "elbo_history_points=2000" in body
    # Three consecutive trace values would mean the list itself got rendered.
    assert not re.search(r"-39[0-9]\.\d+, -39[0-9]\.\d+, -39[0-9]\.\d+", body), (
        "the ELBO trace is being rendered into the report body"
    )
    assert len(body.splitlines()) < 40


def test_the_json_summarises_diagnostics_rather_than_dumping_them(thin):
    """`report --json` is piped into jq, and 2000 of its 2654 lines were trace."""
    fit = _with_history(IrtFit.load(thin.save(_tmp_path_for(thin))))
    payload = build_report(fit).to_json()
    diagnostics = payload["header"]["diagnostics"]
    assert "elbo_history" not in diagnostics
    assert diagnostics["elbo_history_points"] == 2000
    assert diagnostics["elbo_final"] == fit.diagnostics["elbo_final"]
    # The claim is about the diagnostics block, not the whole payload: the item
    # array is legitimately long and is what a caller asked for.
    block = json.dumps(diagnostics, indent=2).splitlines()
    assert len(block) < 40, (
        f"the diagnostics block is {len(block)} lines of a report that is meant "
        "to be piped into jq"
    )


def test_any_long_diagnostics_value_is_summarised_not_just_the_elbo_trace(thin):
    """The filter is on type, so the next list-valued diagnostics key is safe.

    Special-casing `elbo_history` would have fixed the instance and left the
    class of bug in place.
    """
    fit = IrtFit.load(thin.save(_tmp_path_for(thin)))
    fit.diagnostics["some_future_trace"] = list(range(5000))
    fit.diagnostics["nested"] = {"scalar": 1, "trace": list(range(5000))}
    summary = summarise_diagnostics(fit.diagnostics)
    assert summary["some_future_trace_points"] == 5000
    assert "some_future_trace" not in summary
    assert summary["nested"] == {"scalar": 1}


def test_the_dead_threshold_advice_states_the_measured_respondent_count(thin):
    """It said "roughly a hundred respondents" and that is not what it takes.

    `dead` needs the interval wholly inside (0, DEAD_THRESHOLD), so it has to
    clear zero as well as the threshold; that caps sd(a) at 0.089. Measured on
    synth matrices with the fitter's conditional-information intervals, no item
    reaches `dead` at 300 respondents and eighteen do at 1000.
    """
    header = build_header(thin)
    if header.dead_count:
        pytest.skip("this fixture has dead items, so the advice is not shown")
    detail = " ".join(build_refusal(header).detail)
    assert "thousand respondents, not a hundred" in detail
    assert "0.089" in detail
