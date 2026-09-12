"""What the HTML report claims, and the two ways it could quietly lie.

The first is the plot. Its job is to make a mismatch between a suite and the
models it was run on *visible*, so the tests below construct a suite whose
items are calibrated two and a half logits above where the models sit and
assert that the rendered page says so — in the verdict, and in the geometry of
the SVG, where the information peak has to land clear of the band the
respondents occupy. `MISMATCH` is the fixture the brief is about; a test suite
that only ever rendered a healthy fit would not be testing the feature.

The second is the page as an artefact. It has to open from `file://` on a
machine with no network, so nothing in it may be fetched: no stylesheet link,
no font URL, no script src, no image. `test_nothing_is_fetched_at_open_time`
asserts that by looking for the absence of any URL at all, which is why the
inline SVG carries no `xmlns` — in text/html it does not need one, and leaving
it out means the assertion can be "no http anywhere" rather than a list of
exceptions that would grow until it caught nothing.

Two tests deliberately measure rather than pattern-match, because the
introspecting test that passes vacuously is the one that costs the most:
`test_nothing_overflows_the_viewbox` parses every coordinate *and estimates
text extents*, which is how the `respondents` rug label was caught running off
the left edge; `test_the_curve_agrees_with_select` recomputes the curve from
`select`'s own functions over `select`'s own candidate pool, so the plot cannot
drift away from what anchor selection actually optimises.

No torch anywhere. Every fixture is fabricated from known parameters with
numpy alone.
"""

from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

from irtcheck import __version__
from irtcheck.artifact import (
    FLAG_INSUFFICIENT_DATA,
    EmbeddedResponses,
    IrtFit,
    Posterior,
    compute_flags,
    utc_now,
)
from irtcheck.cli import app
from irtcheck.html import (
    COVERAGE_ALIGNED,
    COVERAGE_MISMATCH,
    GRID_POINTS,
    MIN_SPAN,
    PLOT_X0,
    PLOT_X1,
    SVG_H,
    SVG_W,
    VERDICT_ALIGNED,
    VERDICT_MISMATCH,
    VERDICT_TITLES,
    VERDICT_UNREADABLE,
    Curve,
    _ticks,
    build_curve,
    information_pool,
    render_html,
    render_svg,
    write_html,
)
from irtcheck.matrix import build_matrix
from irtcheck.report import build_header, build_report, item_rows
from irtcheck.select import ability_distribution, item_information, select_anchor
from irtcheck.synth import (
    Z95,
    make_truth,
    responses_from_truth,
    synthetic_fit,
    synthetic_records,
)

# A suite calibrated for a stronger generation of models than the ones in the
# matrix: item difficulties shifted +2.4, respondent abilities squeezed toward
# -0.5. This is the case the plot exists for.
MISMATCH = dict(b_shift=2.4, theta_scale=0.4, theta_shift=-0.5, n_models=40, seed=11)

# Enough respondents that item parameters are pinned down, and no shift: the
# suite and the models line up.
HEALTHY = dict(n_models=120, n_items=200, precision=1.5, seed=3)

# Five real models behind fifteen pseudo-respondents — the shape the spec
# assumes, in which most items are honestly unrankable.
THIN = dict(n_models=5, variants_per_model=3, n_items=200, precision=0.5, seed=7)


def shifted_fit(
    *,
    b_shift: float = 0.0,
    theta_scale: float = 1.0,
    theta_shift: float = 0.0,
    n_models: int = 10,
    n_items: int = 180,
    variants_per_model: int = 1,
    seed: int = 11,
    precision: float = 1.2,
) -> IrtFit:
    """A fabricated fit whose items sit where we say, relative to its models.

    `synth.synthetic_fit` draws difficulties and abilities from the same centred
    distribution, so it cannot express "this suite is aimed two logits above
    these models" — the one case the centrepiece plot is built to surface. This
    shifts the *truth* before responses are sampled from it, so the artifact is
    internally consistent: the responses, `p_correct` and the flags all come
    out of the shifted universe rather than being edited afterwards.
    """
    truth = make_truth(
        n_models=n_models,
        n_items=n_items,
        variants_per_model=variants_per_model,
        dead_fraction=0.12,
        off_range_fraction=0.0,
        seed=seed,
    )
    truth.b = truth.b + b_shift
    truth.theta = truth.theta * theta_scale + theta_shift

    responses = responses_from_truth(truth, seed=seed + 1)
    records = synthetic_records(truth, responses)
    key = ("model_id", "prompt_variant") if variants_per_model > 1 else ("model_id",)
    matrix = build_matrix(records, respondent_key=key)

    n_resp = matrix.item_response_counts()
    p_correct = matrix.item_p_correct()
    denom = np.sqrt(np.maximum(n_resp, 1)) * precision
    sd_a = np.clip(0.95 / denom, 0.01, 5.0)
    sd_b = np.clip(1.30 / denom, 0.01, 5.0)
    sd_theta = np.full(truth.n_respondents, 0.9 / np.sqrt(max(matrix.n_items, 1)))

    def posterior(values, sd) -> Posterior:
        mean = np.asarray(values, dtype=float)
        spread = np.broadcast_to(np.asarray(sd, dtype=float), mean.shape).copy()
        return Posterior(
            mean=[float(x) for x in mean],
            sd=[float(x) for x in spread],
            hdi_low=[float(x) for x in mean - Z95 * spread],
            hdi_high=[float(x) for x in mean + Z95 * spread],
        )

    a_post = posterior(truth.a, sd_a)
    b_post = posterior(truth.b, sd_b)
    theta_post = posterior(truth.theta, sd_theta)

    fit = IrtFit(
        irtcheck_version=__version__,
        created=utc_now(),
        model={
            "kind": "2pl",
            "priors": "hierarchical",
            "identification": "a > 0, theta ~ N(0, 1)",
            "synthetic": True,
        },
        respondent_key=list(matrix.respondent_key),
        respondent_ids=list(matrix.respondent_ids),
        derives_from=list(matrix.derives_from),
        theta=theta_post,
        item_ids=list(matrix.item_ids),
        a=a_post,
        b=b_post,
        n_resp=[int(x) for x in n_resp],
        p_correct=[float(x) for x in np.nan_to_num(p_correct, nan=0.0)],
        flags=compute_flags(
            a_post, b_post, [float(p) for p in p_correct], theta_post.mean
        ),
        diagnostics={"source": "tests.test_html.shifted_fit", "seed": seed},
        passthrough=matrix.passthrough,
        responses=EmbeddedResponses(
            rows=[int(x) for x in matrix.rows],
            cols=[int(x) for x in matrix.cols],
            obs=[int(x) for x in matrix.obs],
        ),
    )
    fit.validate()
    return fit


@pytest.fixture(scope="module")
def mismatched() -> IrtFit:
    return shifted_fit(**MISMATCH)


@pytest.fixture(scope="module")
def healthy() -> IrtFit:
    fit, _ = synthetic_fit(**HEALTHY)
    return fit


@pytest.fixture(scope="module")
def thin() -> IrtFit:
    fit, _ = synthetic_fit(**THIN)
    return fit


@pytest.fixture(scope="module")
def unreadable() -> IrtFit:
    """A fit so thin that every item's discrimination interval spans zero.

    The page still has to render: `build_curve` divides by the peak of a curve
    that is identically zero, and the verdict has to say so rather than
    claiming alignment with a suite it cannot read at all.
    """
    fit, _ = synthetic_fit(n_models=3, n_items=30, precision=0.04, seed=5)
    return fit


@pytest.fixture(scope="module")
def page(mismatched: IrtFit) -> str:
    return render_html(mismatched, limit=25, source="mismatch.irt")


# -- the page is self-contained ---------------------------------------------


def test_nothing_is_fetched_at_open_time(page: str):
    """Opens offline, with no missing assets. Asserted as the absence of any
    URL at all: a list of allowed exceptions would grow until it caught
    nothing."""
    assert page.startswith("<!doctype html>")
    for smell in ("http://", "https://", "//cdn", "<link", "src=", "@import", "url("):
        assert smell not in page, f"the page would fetch something: {smell!r}"
    assert "<style>" in page and "<svg" in page


def test_the_page_paints_its_own_colours_in_both_themes(page: str):
    """A chart whose text is invisible on half of readers' machines is a bug."""
    assert "prefers-color-scheme: dark" in page
    assert "background: var(--paper)" in page
    # Every colour the SVG uses is a token, so one document serves both themes.
    for token in ("--ink", "--muted", "--accent", "--warn", "--paper"):
        assert token in page


def test_written_to_disk_and_reopened(mismatched: IrtFit, tmp_path):
    out = write_html(mismatched, tmp_path / "report.html", limit=10)
    assert out.read_text(encoding="utf-8").rstrip().endswith("</html>")


def test_user_supplied_ids_are_escaped(healthy: IrtFit):
    fit = IrtFit.from_dict(healthy.to_dict())
    fit.item_ids[0] = "<script>alert(1)</script>&"
    fit.derives_from[0] = "<img src=x>"
    fit.respondent_ids[0] = "<img src=x>"
    page = render_html(fit, limit=5, sort="id")
    assert "<script>alert(1)</script>&" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;&amp;" in page
    assert "<img src=x>" not in page


# -- the plot is the pitch ---------------------------------------------------


def test_the_theta_marks_come_from_the_fitted_respondents(mismatched: IrtFit):
    """Not a grid. Every rug tick carries the posterior theta mean it was
    drawn at, and the set of them is the fit's own respondents."""
    svg = render_svg(build_curve(mismatched))
    drawn = sorted(float(v) for v in re.findall(r'class="rug" data-theta="([-\d.]+)"', svg))
    assert len(drawn) == mismatched.n_respondents
    assert drawn == pytest.approx(sorted(mismatched.theta.mean), abs=1e-6)


def test_pseudo_respondents_are_weighted_per_real_model(thin: IrtFit):
    """The same weighting `select` uses: ten temperature samples of one model
    do not outvote a model run once."""
    curve = build_curve(thin)
    _, weights = ability_distribution(thin)
    assert curve.theta_weights == pytest.approx(list(weights))
    assert sum(curve.theta_weights) == pytest.approx(1.0)
    assert len(curve.models) == thin.n_real_models < thin.n_respondents


def test_the_curve_agrees_with_select(mismatched: IrtFit):
    """The plot must integrate what `select` optimises, over the items `select`
    can actually choose from. Recomputed here from select's own functions."""
    pool = information_pool(mismatched)
    assert pool == [
        i for i in mismatched.usable_items() if mismatched.n_resp[i] > 0
    ]
    # And that pool is the one selection uses, not merely one that looks like it.
    assert select_anchor(mismatched, len(pool)).n_usable == len(pool)

    curve = build_curve(mismatched)
    a = np.asarray(mismatched.a.mean)[pool]
    b = np.asarray(mismatched.b.mean)[pool]
    expected = item_information(a, b, np.asarray(curve.grid)).sum(axis=0)
    assert curve.info == pytest.approx(list(expected))
    assert len(curve.grid) == GRID_POINTS

    at_theta = item_information(a, b, np.asarray(curve.theta)).sum(axis=0)
    assert curve.info_at_theta == pytest.approx(list(at_theta))
    assert curve.info_at_models == pytest.approx(
        float(np.dot(curve.theta_weights, at_theta))
    )


def test_the_mismatch_case_is_obvious(mismatched: IrtFit, page: str):
    """The whole brief. A suite measuring where no model sits has to read as a
    mismatch in the numbers, in the words, and in the geometry."""
    curve = build_curve(mismatched)
    assert curve.verdict == VERDICT_MISMATCH
    assert curve.coverage < COVERAGE_MISMATCH
    assert curve.peak_outside_models
    assert curve.peak_theta > curve.theta_hi + 1.0

    assert VERDICT_TITLES[VERDICT_MISMATCH] in page
    assert f"θ = {curve.peak_theta:+.2f}" in page
    assert f"{curve.coverage:.0%} of its peak information" in page

    # Geometry: the information peak is drawn clear of the band the
    # respondents occupy, with room to see the gap.
    svg = render_svg(curve)
    band = re.search(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', svg)
    peak = re.search(r'<circle cx="([\d.]+)"', svg)
    assert band and peak
    band_right = float(band.group(1)) + float(band.group(2))
    plot_width = PLOT_X1 - PLOT_X0
    assert float(peak.group(1)) - band_right > 0.15 * plot_width


def test_an_aligned_suite_says_so(healthy: IrtFit):
    curve = build_curve(healthy)
    assert curve.verdict == VERDICT_ALIGNED
    assert curve.coverage >= COVERAGE_ALIGNED
    assert not curve.peak_outside_models
    assert VERDICT_TITLES[VERDICT_ALIGNED] in render_html(healthy, limit=5)


def test_a_fit_with_no_readable_items_refuses_to_read_the_plot(unreadable: IrtFit):
    curve = build_curve(unreadable)
    assert curve.n_pool == 0
    assert curve.verdict == VERDICT_UNREADABLE
    assert curve.coverage == 0.0
    page = render_html(unreadable, limit=5)
    assert VERDICT_TITLES[VERDICT_UNREADABLE] in page
    assert "<svg" in page  # still renders, flat


def test_a_single_respondent_still_renders(mismatched: IrtFit):
    """One model is below anything the spec assumes, and it degenerates the
    ability axis to a point. The page still has to come out."""
    fit, _ = synthetic_fit(n_models=1, n_items=12, precision=3.0, seed=1)
    curve = build_curve(fit)
    assert curve.theta_lo == curve.theta_hi
    assert curve.hi - curve.lo >= MIN_SPAN
    assert len(curve.models) == 1
    assert "<svg" in render_html(fit, limit=0)


def test_an_artifact_without_embedded_responses_still_renders(healthy: IrtFit):
    """`fit --no-embed-responses` leaves the response count unknown; the header
    block drops the density line rather than printing a wrong one."""
    payload = healthy.to_dict()
    payload["responses"] = None
    page = render_html(IrtFit.from_dict(payload), limit=5)
    assert "dense" not in page
    assert "<svg" in page


def test_the_dashed_line_appears_only_when_it_differs(mismatched: IrtFit):
    curve = build_curve(mismatched)
    assert curve.n_pool < curve.n_items
    assert curve.show_all_items
    assert 'class="info-all"' in render_svg(curve)

    every_item_readable = Curve(**{**vars_of(curve), "show_all_items": False})
    assert 'class="info-all"' not in render_svg(every_item_readable)


def vars_of(curve: Curve) -> dict:
    return {name: getattr(curve, name) for name in curve.__slots__}


# -- the drawing stays inside its frame -------------------------------------

# Conservative advance width per character for the monospace faces the page
# asks for, as a fraction of font size. IBM Plex Mono is 0.6em; 0.62 leaves a
# little room for whatever the reader's machine substitutes.
CHAR_WIDTH = 0.62

TEXT = re.compile(
    r'<text x="(?P<x>[-\d.]+)" y="(?P<y>[-\d.]+)"[^>]*class="(?P<cls>[^"]*)"[^>]*>'
    r"(?P<body>[^<]*)</text>"
)
FONT_SIZES = {"tick": 11.0, "rug-label": 10.0, "axis-title": 11.5, "callout": 12.0}


def test_nothing_overflows_the_viewbox(mismatched: IrtFit, healthy: IrtFit, thin: IrtFit):
    """Coordinates *and* text extents. Checking only the attributes would have
    passed while the word `respondents` hung off the left edge, which is how
    that bug was found."""
    for fit in (mismatched, healthy, thin):
        svg = render_svg(build_curve(fit))
        for attribute in ("x", "y", "x1", "x2", "cx", "width"):
            for value in re.findall(rf'\b{attribute}="([-\d.]+)"', svg):
                assert -0.01 <= float(value) <= SVG_W + 0.01, f"{attribute}={value}"
        for attribute in ("y", "y1", "y2", "cy", "height"):
            for value in re.findall(rf'\b{attribute}="([-\d.]+)"', svg):
                assert -0.01 <= float(value) <= SVG_H + 0.01, f"{attribute}={value}"

        for path in re.findall(r'\sd="([^"]+)"', svg):
            for x, y in re.findall(r"([-\d.]+),([-\d.]+)", path):
                assert 0.0 <= float(x) <= SVG_W and 0.0 <= float(y) <= SVG_H

        for match in TEXT.finditer(svg):
            classes = match.group("cls").split()
            size = next(
                (FONT_SIZES[c] for c in classes if c in FONT_SIZES), 12.0
            )
            width = len(match.group("body")) * size * CHAR_WIDTH
            x = float(match.group("x"))
            if "end" in classes:
                left, right = x - width, x
            elif "middle" in classes:
                left, right = x - width / 2, x + width / 2
            else:
                left, right = x, x + width
            if "axis-title" in classes and x < PLOT_X0:
                continue  # rotated onto the vertical axis; x is its centre line
            assert left >= -0.5, f"{match.group('body')!r} runs off the left edge"
            assert right <= SVG_W + 0.5, f"{match.group('body')!r} runs off the right"


def test_every_axis_label_names_a_value_the_plot_reaches(mismatched: IrtFit):
    curve = build_curve(mismatched)
    for value in _ticks(curve.lo, curve.hi, 8):
        assert curve.lo <= value <= curve.hi
    assert _ticks(0.0, 1.0, 5)  # a degenerate axis still produces labels
    assert _ticks(-0.4, 0.4, 5)


# -- the honest bits ---------------------------------------------------------


def test_the_header_counts_real_models_not_respondents(thin: IrtFit):
    """CLAUDE.md: reporting fifteen respondents when they are three prompt
    variants of five models misrepresents the one thing this tool is here to be
    honest about."""
    page = render_html(thin, limit=5)
    assert "5 real models" in page
    assert "15 respondents" in page
    assert "15 real models" not in page
    assert "--respondent-key model_id,prompt_variant" in page


def test_the_refusal_is_on_the_page_when_the_data_is_thin(thin: IrtFit):
    report = build_report(thin, limit=5)
    assert report.refusal.is_refusal
    page = render_html(thin, limit=5)
    assert report.refusal.headline in page
    assert "cannot rank these items yet" in page
    assert report.refusal.suggested_respondent_key in page
    # dead and insufficient-data are counted separately, never summed.
    header = report.header
    assert f"<b>{header.insufficient_count}</b>" in page
    assert f"<b>{header.dead_count}</b>" in page


def test_a_healthy_fit_carries_no_refusal_panel(healthy: IrtFit):
    page = render_html(healthy, limit=5)
    assert "cannot rank these items yet" not in page
    assert "read the ranking with care" not in page


def test_insufficient_data_items_are_not_ranked(thin: IrtFit):
    """A table that misrepresents insufficient-data as a low rank is a bug.
    They carry no rank number, are marked unrankable for the re-sort, and sort
    last under the ranking sort."""
    page = render_html(thin, limit=0)
    rows = re.findall(
        r"<tr(?P<cls>[^>]*)data-id='(?P<id>[^']*)'[^>]*"
        r"data-rankable='(?P<rankable>[01])'>(?P<cells>.*?)</tr>",
        page,
        flags=re.S,
    )
    assert len(rows) == thin.n_items
    unrankable = {
        row.item_id
        for row in item_rows(thin)
        if FLAG_INSUFFICIENT_DATA in row.flags
    }
    assert unrankable

    seen_unrankable = False
    ranks: list[int] = []
    for cls, item_id, rankable, cells in rows:
        first_cell = re.search(r"<td class='num rank'>(.*?)</td>", cells).group(1)
        if item_id in unrankable:
            assert rankable == "0"
            assert "unrankable" in cls
            assert first_cell == "&mdash;", f"{item_id} was given a rank"
            seen_unrankable = True
        else:
            assert rankable == "1"
            assert not seen_unrankable, "a rankable item sorted below an unrankable one"
            ranks.append(int(first_cell))
    assert ranks == sorted(ranks) == list(range(1, len(ranks) + 1))

    # The page says the rule, and the re-sort in it keeps the rule rather than
    # undoing what the server-side ordering was careful about.
    assert "excluded from the ranking, not ranked low" in page
    assert "excluded from the ranking rather than ranked low" in page  # in the script
    assert "if (ra !== rb) { return rb - ra; }" in page


def test_the_table_only_offers_orderings_the_terminal_will_produce(healthy: IrtFit):
    page = render_html(healthy, limit=5)
    keys = set(re.findall(r"data-key='([^']+)'", page))
    assert keys == {"item", "a", "b"}


def test_the_score_threshold_is_printed_when_one_was_used(healthy: IrtFit):
    """CLAUDE.md: thresholding a continuous score is a documented escape hatch
    that has to print its threshold, not a quiet cast."""
    fit = IrtFit.from_dict(healthy.to_dict())
    fit.diagnostics["score_threshold"] = 0.62
    page = render_html(fit, limit=5)
    assert build_header(fit).score_threshold == 0.62
    assert "binarised at 0.62" in page


def test_the_footer_says_how_much_of_the_suite_is_shown(healthy: IrtFit):
    page = render_html(healthy, limit=12, source="suite.irt")
    assert f"Showing 12 of {healthy.n_items} items" in page
    assert "--limit 0" in page
    assert "suite.irt" in page
    every = render_html(healthy, limit=0)
    assert f"All {healthy.n_items} items" in every
    assert "Showing" not in every


def test_a_synthetic_artifact_says_it_is_synthetic(healthy: IrtFit):
    assert "SYNTHETIC ARTIFACT" in render_html(healthy, limit=5)


# -- wiring ------------------------------------------------------------------


def test_report_html_writes_the_file(mismatched: IrtFit, tmp_path):
    artifact = mismatched.save(tmp_path / "suite.irt")
    out = tmp_path / "report.html"
    result = CliRunner().invoke(app, ["report", str(artifact), "--html", str(out)])
    assert result.exit_code == 0, result.output
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<svg" in page


def test_report_html_alongside_json_keeps_stdout_pipeable(
    mismatched: IrtFit, tmp_path, capsys
):
    """`--html` is documented as *also* writing a file, and `--json` is
    documented as pipeable into jq. Both at once therefore means the 'wrote'
    line goes to stderr, not into the middle of the JSON.

    Driven through `commands.report.run` rather than CliRunner because this
    click version merges the two streams in the runner, and a test that cannot
    tell them apart would pass whatever the code did.
    """
    import json

    from irtcheck.commands import report as impl

    artifact = mismatched.save(tmp_path / "suite.irt")
    out = tmp_path / "report.html"
    impl.run(artifact=artifact, html=out, as_json=True, limit=5)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["header"]["n_real_models"] == 40
    assert out.exists()
    assert "wrote" in captured.err


def test_the_html_report_runs_without_torch():
    """Analysing a cached fit has to work on a machine that never installed it.
    In a subprocess, because the session may already have imported torch."""
    code = (
        "import irtcheck.html, sys; "
        "sys.exit(1 if ('torch' in sys.modules or 'pyro' in sys.modules) else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
