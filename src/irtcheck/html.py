"""The HTML report: one self-contained file, and one plot that is the pitch.

`--html` writes a single file that opens from `file://` on a laptop in a
tunnel. No stylesheet link, no font request, no CDN, no image URL: every byte
the page needs is in the page. That is not an aesthetic preference — the tool's
whole claim is "no network at runtime", and a report that phones out for a
webfont breaks it in the one place a user would notice.

**The centrepiece is the test information curve drawn against the ability
distribution of the respondents in the matrix**, because that is the plot the
spec calls the pitch: it shows at a glance when a suite measures precisely in
an ability range none of the user's models occupy. Both things therefore share
one set of axes — the curve

    I(theta) = sum_i a_i^2 P_i(theta) (1 - P_i(theta))

in teal, and the respondents in amber: a band over the range they span, one
faint vertical line per respondent, and a rug of ticks under the axis. A suite
whose information piles up at theta = +2 while every model sits at -0.5 is then
a teal mountain with nothing under it and an amber bundle out on the flat.

Three decisions in here are worth defending.

**The curve is computed over exactly the pool `select` chooses from** —
`usable_items()` intersected with items that have at least one response, via
`select.item_information` and `select.ability_distribution`. If this module
integrated against its own grid, or included items whose discrimination
interval spans zero, the plot would advertise precision the anchor sets built
from the same artifact cannot deliver. The dashed line is the same sum over
*every* item, drawn only when it differs, so the gap between "what this suite
would measure if we could read all of it" and "what this fit can actually use"
is visible rather than argued about.

**The plot draws the respondents, never a grid.** Marks sit at posterior theta
means, one per respondent, weighted the way `select` weights them: each real
model carries equal total weight, so ten temperature samples of one model do
not drag the coverage number toward that model's ability.

**No matplotlib.** It is a declared dependency and this module could import it,
but hand-written SVG is a few kilobytes instead of a multi-second import, stays
sharp at any zoom, and — the reason that settles it — takes its colours from
CSS custom properties, so one document is legible in a light browser and in a
dark one without rendering twice. Every axis tick is generated from the data's
own range, so a label never names a value the plot does not reach.

Nothing here recomputes what `report` already knows: the header block, the flag
counts, the refusal and the per-item rows all come from `report.build_report`,
and the columns are `report.COLUMNS`, so the terminal table and this one cannot
drift apart. No torch, no pyro, no `irtcheck.fit`; see CLAUDE.md → The lazy
torch boundary.
"""

from __future__ import annotations

import html as _html
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from irtcheck.artifact import (
    CEILING_THRESHOLD,
    DEAD_THRESHOLD,
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
    FLAG_INVERTED,
    FLAG_OFF_RANGE,
    FLOOR_THRESHOLD,
    IrtFit,
)
from irtcheck.report import (
    COLUMNS,
    DEFAULT_SORT,
    Header,
    Refusal,
    Report,
    _diagnostics_line,
    build_report,
    item_rows,
    sort_rows,
)
from irtcheck.select import (
    THETA_PRIOR_PRECISION,
    ability_distribution,
    item_information,
)

# -- the curve ---------------------------------------------------------------

# Resolution of the theta grid the curve is drawn on. 241 points over a span of
# ten puts a sample every 0.04, finer than the width of the stroke.
GRID_POINTS = 241

# Padding either side of the plotted range, in theta units, so the curve is not
# clipped at the frame and the outermost respondent is not drawn on the axis.
GRID_PAD = 0.75

# Item difficulties enter the plotted range at this percentile from each end. A
# single wild `b` from a barely-identified item would otherwise set the axis
# scale for the whole suite and squash everything else into the middle.
B_PERCENTILE = 2.0

# Difficulties beyond this are not allowed to widen the axis; respondents
# always are, however far out they sit. An item at b = 40 tells the reader
# nothing they cannot read off the `off-range` count.
MAX_ABS_B = 8.0

# A fit whose respondents all sit within a hair of each other still needs an
# axis wide enough to read.
MIN_SPAN = 4.0

# Share of the suite's *peak* information that actually reaches the models in
# this matrix. At or above ALIGNED the suite is measuring where the models are;
# below MISMATCH it is measuring somewhere else, which is the finding the plot
# exists to make obvious. Conventions, like DEAD_THRESHOLD, not laws.
COVERAGE_ALIGNED = 0.75
COVERAGE_MISMATCH = 0.40

VERDICT_ALIGNED = "aligned"
VERDICT_PARTIAL = "partial"
VERDICT_MISMATCH = "mismatch"
VERDICT_UNREADABLE = "unreadable"


@dataclass(slots=True)
class ModelPosition:
    """Where one real model sits, and how precisely the whole suite pins it.

    `theta` is the mean over that model's pseudo-respondents; `theta_se` is
    1/sqrt(prior precision + information), the same arithmetic `select` reports
    for an anchor set, here for the suite as a whole.
    """

    model_id: str
    theta: float
    n_respondents: int
    information: float
    theta_se: float


@dataclass(slots=True)
class Curve:
    """Everything the plot draws, and the numbers the verdict is made of."""

    grid: list[float]
    info: list[float]  # over the pool `select` chooses from
    info_all: list[float]  # over every item; drawn only when it differs
    show_all_items: bool
    theta: list[float]  # posterior means, one per respondent
    theta_weights: list[float]  # each real model gets equal total weight
    info_at_theta: list[float]
    models: list[ModelPosition]
    n_pool: int
    n_items: int
    lo: float
    hi: float
    theta_lo: float
    theta_hi: float
    theta_mean: float
    peak_theta: float
    peak_info: float
    info_at_models: float
    coverage: float
    verdict: str

    @property
    def max_info(self) -> float:
        return max([*self.info, *(self.info_all if self.show_all_items else []), 0.0])

    @property
    def peak_outside_models(self) -> bool:
        return not (self.theta_lo <= self.peak_theta <= self.theta_hi)


def information_pool(fit: IrtFit) -> list[int]:
    """The items the curve is summed over — `select`'s candidate pool, exactly.

    `usable_items()` drops insufficient-data (we cannot say what it measures)
    and ceiling/floor (it separates nobody). `n_resp > 0` drops items nobody
    answered, which `select_anchor` excludes explicitly and for its own
    reasons. Drawing a curve over a wider pool than selection can draw on would
    promise precision no anchor set from this artifact can deliver.
    """
    return [i for i in fit.usable_items() if fit.n_resp[i] > 0]


def _plot_range(fit: IrtFit, pool: list[int]) -> tuple[float, float]:
    theta = np.asarray(fit.theta.mean, dtype=float)
    lo = float(theta.min()) - GRID_PAD
    hi = float(theta.max()) + GRID_PAD
    if pool:
        b = np.asarray(fit.b.mean, dtype=float)[pool]
        b_lo = float(np.percentile(b, B_PERCENTILE)) - GRID_PAD
        b_hi = float(np.percentile(b, 100.0 - B_PERCENTILE)) + GRID_PAD
        lo = min(lo, max(b_lo, -MAX_ABS_B))
        hi = max(hi, min(b_hi, MAX_ABS_B))
    if hi - lo < MIN_SPAN:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * MIN_SPAN, mid + 0.5 * MIN_SPAN
    return lo, hi


def build_curve(fit: IrtFit) -> Curve:
    """Test information against the respondents' abilities, plus the verdict."""
    pool = information_pool(fit)
    theta, weights = ability_distribution(fit)
    lo, hi = _plot_range(fit, pool)
    grid = np.linspace(lo, hi, GRID_POINTS)

    a = np.asarray(fit.a.mean, dtype=float)
    b = np.asarray(fit.b.mean, dtype=float)

    def total(items: list[int], at: np.ndarray) -> np.ndarray:
        if not items:
            return np.zeros(at.shape, dtype=float)
        return item_information(a[items], b[items], at).sum(axis=0)

    info = total(pool, grid)
    info_all = total(list(range(fit.n_items)), grid)
    info_at_theta = total(pool, theta)

    # One point per real model, at the mean of its pseudo-respondents. The
    # header counts models rather than respondents, and so does this.
    model_ids: list[str] = list(dict.fromkeys(fit.derives_from))
    model_theta = np.array(
        [
            float(np.mean([theta[j] for j, s in enumerate(fit.derives_from) if s == src]))
            for src in model_ids
        ],
        dtype=float,
    )
    model_info = total(pool, model_theta)
    models = [
        ModelPosition(
            model_id=src,
            theta=float(model_theta[k]),
            n_respondents=sum(1 for s in fit.derives_from if s == src),
            information=float(model_info[k]),
            theta_se=float(
                1.0 / math.sqrt(THETA_PRIOR_PRECISION + max(float(model_info[k]), 0.0))
            ),
        )
        for k, src in enumerate(model_ids)
    ]

    peak_index = int(np.argmax(info))
    peak_info = float(info[peak_index])
    info_at_models = float(np.dot(weights, info_at_theta))
    coverage = info_at_models / peak_info if peak_info > 0 else 0.0

    if not pool or peak_info <= 0.0:
        verdict = VERDICT_UNREADABLE
    elif coverage >= COVERAGE_ALIGNED:
        verdict = VERDICT_ALIGNED
    elif coverage >= COVERAGE_MISMATCH:
        verdict = VERDICT_PARTIAL
    else:
        verdict = VERDICT_MISMATCH

    return Curve(
        grid=[float(x) for x in grid],
        info=[float(x) for x in info],
        info_all=[float(x) for x in info_all],
        show_all_items=bool(np.max(np.abs(info_all - info)) > 1e-9),
        theta=[float(x) for x in theta],
        theta_weights=[float(x) for x in weights],
        info_at_theta=[float(x) for x in info_at_theta],
        models=models,
        n_pool=len(pool),
        n_items=fit.n_items,
        lo=lo,
        hi=hi,
        theta_lo=float(theta.min()),
        theta_hi=float(theta.max()),
        theta_mean=float(np.dot(weights, theta)),
        peak_theta=float(grid[peak_index]),
        peak_info=peak_info,
        info_at_models=info_at_models,
        coverage=coverage,
        verdict=verdict,
    )


# -- the sentence the plot is making -----------------------------------------

VERDICT_TITLES = {
    VERDICT_ALIGNED: "This suite measures where your models sit",
    VERDICT_PARTIAL: "Only part of this suite's precision reaches your models",
    VERDICT_MISMATCH: "This suite is measuring an ability range your models do not occupy",
    VERDICT_UNREADABLE: "There is not enough here to say where this suite measures",
}


def verdict_lines(curve: Curve, header: Header) -> list[str]:
    """The plot, in words, for the reader who skips plots.

    Written from the same numbers the plot is drawn from — `coverage` is the
    height of the amber bundle against the height of the teal peak — so the
    text cannot disagree with the picture above it.
    """
    if curve.verdict == VERDICT_UNREADABLE:
        return [
            f"None of the {curve.n_items} items are usable for measurement in this fit: "
            "every one is flagged insufficient-data, inverted, ceiling or floor, or was answered by "
            "nobody at all. The curve above is flat because there is nothing to draw.",
            "Add respondents before reading anything into where this suite measures.",
        ]

    plural = "s" if header.n_real_models != 1 else ""
    lines = [
        f"Peak information is at θ = {curve.peak_theta:+.2f}. Your "
        f"{header.n_real_models} model{plural} occupy θ {curve.theta_lo:+.2f} to "
        f"{curve.theta_hi:+.2f} (weighted mean {curve.theta_mean:+.2f}). At the abilities "
        f"they actually occupy this suite delivers {curve.coverage:.0%} of its peak "
        f"information ({curve.info_at_models:.2f} against {curve.peak_info:.2f})."
    ]

    if curve.verdict == VERDICT_ALIGNED:
        lines.append(
            "The precision this suite carries and the abilities you are measuring line up: "
            "an anchor set chosen from this fit is being asked to separate models the items "
            "were built to separate."
        )
    else:
        where = (
            "outside the range your models span"
            if curve.peak_outside_models
            else "inside the range your models span, but away from where most of them sit"
        )
        lines.append(
            f"That peak is {where}, so most of what this suite can measure is spent on "
            "ability differences none of these models exhibit. Accuracy on it will still "
            "move between models; it will just move less than the item count suggests."
        )
        off_range = header.flag_counts[FLAG_OFF_RANGE]
        if off_range:
            lines.append(
                f"{off_range} of {header.n_items} items are flagged off-range for exactly "
                "this reason: their difficulty sits outside the ability range these "
                "respondents occupy. They may discriminate beautifully — just not for "
                "anyone in this matrix."
            )
        lines.append(
            "Two things fix this and they are different fixes. Add harder or easier items "
            "to move the suite toward your models; or add models that span more of the "
            "range, which is also what sharpens every item parameter below."
        )

    if curve.n_pool < curve.n_items:
        tail = (
            "; the dashed line adds the rest, whose discrimination this fit cannot confirm."
            if curve.show_all_items
            else "."
        )
        lines.append(
            f"The curve is summed over the {curve.n_pool} of {curve.n_items} items this fit "
            f"can read and irtcheck select can choose from{tail}"
        )
    return lines


# -- SVG ---------------------------------------------------------------------

# One viewBox, fixed. The page scales the drawing with CSS (`width: 100%`), so
# these are layout units rather than pixels, and every coordinate written below
# is inside them — tests/test_html.py parses the SVG back out and asserts it.
SVG_W = 780.0
SVG_H = 430.0
# The left gutter holds the rotated axis title and the widest right-aligned
# label outside the frame, which is the word "respondents" beside the rug —
# 11 monospace characters at 10px. Narrow it and that label runs off the
# viewBox, which is a class of bug tests/test_html.py checks for by measuring
# text extents rather than only the coordinates.
PLOT_X0 = 86.0
PLOT_X1 = SVG_W - 18.0
PLOT_Y0 = 22.0
PLOT_Y1 = 330.0
RUG_Y0 = 344.0
RUG_Y1 = 366.0
XLABEL_Y = 384.0
XTITLE_Y = 406.0

# How close to an edge a callout may sit before its anchor flips, so a label
# near the frame runs inward instead of off the drawing.
LABEL_MARGIN = 46.0


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def _nice_step(span: float, target: int) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def _ticks(lo: float, hi: float, target: int = 7) -> list[float]:
    """Round numbers strictly inside [lo, hi].

    Strictly inside is the point: an axis label has to name a value the plot
    actually reaches, so the ticks are generated from the data's range rather
    than the range being rounded outward to suit the ticks.
    """
    step = _nice_step(hi - lo, target)
    out: list[float] = []
    value = math.ceil(lo / step - 1e-9) * step
    while value <= hi + 1e-9:
        out.append(0.0 if abs(value) < step * 1e-6 else value)
        value += step
    return out


def _theta_label(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    return f"{value:+.1f}".replace("-", "−")


def _info_label(value: float, step: float) -> str:
    places = 0 if step >= 1 else (1 if step >= 0.1 else 2)
    return f"{value:.{places}f}"


def _anchor(x: float) -> tuple[str, float]:
    """Keep a label inside the frame by moving its anchor, not its position."""
    if x < PLOT_X0 + LABEL_MARGIN:
        return "start", max(x, PLOT_X0)
    if x > PLOT_X1 - LABEL_MARGIN:
        return "end", min(x, PLOT_X1)
    return "middle", x


def _polyline(
    xs: list[float],
    ys: list[float],
    sx: Callable[[float], float],
    sy: Callable[[float], float],
) -> str:
    points = [f"{_fmt(sx(x))},{_fmt(sy(y))}" for x, y in zip(xs, ys, strict=True)]
    return "M" + " L".join(points)


def render_svg(curve: Curve) -> str:
    """The plot. Teal is what the suite can measure; amber is who it measured."""
    span = max(curve.hi - curve.lo, 1e-9)

    def sx(theta: float) -> float:
        return PLOT_X0 + (theta - curve.lo) * (PLOT_X1 - PLOT_X0) / span

    headroom = max(curve.max_info, 1e-6) * 1.08
    y_step = _nice_step(headroom, 5)
    y_max = max(y_step * math.ceil(headroom / y_step), y_step)

    def sy(info: float) -> float:
        return PLOT_Y1 - min(max(info, 0.0), y_max) * (PLOT_Y1 - PLOT_Y0) / y_max

    caption = (
        f"Test information peaks at theta {curve.peak_theta:+.2f}; the respondents span "
        f"{curve.theta_lo:+.2f} to {curve.theta_hi:+.2f} and receive "
        f"{curve.coverage:.0%} of that peak."
    )
    parts: list[str] = [
        f'<svg viewBox="0 0 {_fmt(SVG_W)} {_fmt(SVG_H)}" role="img" '
        f'aria-label="{esc(caption)}" class="curve">',
        f"<title>{esc(caption)}</title>",
    ]

    # Where the models sit, behind everything else.
    band_x0, band_x1 = sx(curve.theta_lo), sx(curve.theta_hi)
    band_w = max(band_x1 - band_x0, 2.5)
    parts.append(
        f'<rect x="{_fmt(band_x0)}" y="{_fmt(PLOT_Y0)}" width="{_fmt(band_w)}" '
        f'height="{_fmt(PLOT_Y1 - PLOT_Y0)}" class="band"/>'
    )

    # Gridlines and the information axis.
    for value in _ticks(0.0, y_max, 5):
        y = sy(value)
        parts.append(
            f'<line x1="{_fmt(PLOT_X0)}" y1="{_fmt(y)}" x2="{_fmt(PLOT_X1)}" '
            f'y2="{_fmt(y)}" class="grid"/>'
        )
        parts.append(
            f'<text x="{_fmt(PLOT_X0 - 9)}" y="{_fmt(y + 3.5)}" class="tick end">'
            f"{esc(_info_label(value, y_step))}</text>"
        )

    # One faint line per respondent, drawn *through* the plot rather than only
    # under it: the eye has to be able to see whether the mountain stands over
    # the bundle or somewhere else entirely.
    for theta in curve.theta:
        x = sx(theta)
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(PLOT_Y0)}" x2="{_fmt(x)}" '
            f'y2="{_fmt(PLOT_Y1)}" class="respondent"/>'
        )

    if curve.show_all_items:
        parts.append(
            f'<path d="{_polyline(curve.grid, curve.info_all, sx, sy)}" class="info-all"/>'
        )

    line = _polyline(curve.grid, curve.info, sx, sy)
    parts.append(
        f'<path d="M{_fmt(sx(curve.grid[0]))},{_fmt(PLOT_Y1)} {line[1:]} '
        f'L{_fmt(sx(curve.grid[-1]))},{_fmt(PLOT_Y1)} Z" class="info-area"/>'
    )
    parts.append(f'<path d="{line}" class="info"/>')

    # The peak, named. Without the number the reader has to eyeball it against
    # the axis, and the whole claim of the plot is a comparison of two places.
    if curve.peak_info > 0:
        px, py = sx(curve.peak_theta), sy(curve.peak_info)
        parts.append(
            f'<line x1="{_fmt(px)}" y1="{_fmt(py)}" x2="{_fmt(px)}" '
            f'y2="{_fmt(PLOT_Y1)}" class="peak-rule"/>'
        )
        parts.append(f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" r="3" class="peak-dot"/>')
        label_y = py - 9 if py - 9 > PLOT_Y0 + 24 else py + 17
        anchor, lx = _anchor(px)
        parts.append(
            f'<text x="{_fmt(lx)}" y="{_fmt(label_y)}" class="callout peak {anchor}">'
            # Two places, unlike the axis ticks: the callout names one specific
            # value the verdict text repeats, and "+0.0" for a peak at +0.01
            # reads as a rounding rather than as a location.
            f"peak θ {esc(f'{curve.peak_theta:+.2f}'.replace('-', '−'))}</text>"
        )

    anchor, lx = _anchor(0.5 * (band_x0 + band_x1))
    parts.append(
        f'<text x="{_fmt(lx)}" y="{_fmt(PLOT_Y0 + 13)}" class="callout models {anchor}">'
        "your models</text>"
    )

    # The ability axis, its ticks, and the rug of respondents beneath it.
    parts.append(
        f'<line x1="{_fmt(PLOT_X0)}" y1="{_fmt(PLOT_Y1)}" x2="{_fmt(PLOT_X1)}" '
        f'y2="{_fmt(PLOT_Y1)}" class="axis"/>'
    )
    for value in _ticks(curve.lo, curve.hi, 8):
        x = sx(value)
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(PLOT_Y1)}" x2="{_fmt(x)}" '
            f'y2="{_fmt(PLOT_Y1 + 5)}" class="axis"/>'
        )
        parts.append(
            f'<text x="{_fmt(x)}" y="{_fmt(XLABEL_Y)}" class="tick middle">'
            f"{esc(_theta_label(value))}</text>"
        )

    for theta, weight in zip(curve.theta, curve.theta_weights, strict=True):
        x = sx(theta)
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(RUG_Y0)}" x2="{_fmt(x)}" '
            f'y2="{_fmt(RUG_Y1)}" class="rug" data-theta="{theta:.6f}" '
            f'data-weight="{weight:.6f}"/>'
        )
    parts.append(
        f'<text x="{_fmt(PLOT_X0 - 9)}" y="{_fmt(0.5 * (RUG_Y0 + RUG_Y1) + 3.5)}" '
        'class="rug-label end">respondents</text>'
    )

    mid_y = 0.5 * (PLOT_Y0 + PLOT_Y1)
    parts.append(
        f'<text x="16.0" y="{_fmt(mid_y)}" class="axis-title middle" '
        f'transform="rotate(-90 16 {_fmt(mid_y)})">test information I(θ)</text>'
    )
    parts.append(
        f'<text x="{_fmt(0.5 * (PLOT_X0 + PLOT_X1))}" y="{_fmt(XTITLE_Y)}" '
        'class="axis-title middle">ability θ — the scale this fit identifies as '
        "θ ~ N(0, 1)</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# -- page --------------------------------------------------------------------


def esc(value: object) -> str:
    return _html.escape(str(value), quote=True)


FLAG_CLASSES = {
    FLAG_DEAD: "dead",
    FLAG_INVERTED: "inverted",
    FLAG_INSUFFICIENT_DATA: "unknown",
    FLAG_CEILING: "edge",
    FLAG_FLOOR: "edge",
    FLAG_OFF_RANGE: "off",
}

# Columns a reader may reorder the table by, and the sort key each one uses.
# Deliberately the same three orderings as report.SORT_KEYS: offering an
# ordering the terminal refuses to produce would make this page a second,
# disagreeing interface to the same numbers.
SORTABLE = {"item": "item", "a": "a", "b": "b"}

STYLE = """
:root {
  --paper: #F3F6F7; --surface: #FFFFFF; --surface-2: #EAEFF1;
  --ink: #131A20; --ink-soft: #3D4C55; --muted: #67797F;
  --rule: #D6DEE1; --rule-firm: #B9C6CB;
  --accent: #10627C; --accent-soft: #6FA8BC; --accent-dim: #DCEBF0;
  --warn: #9E6913; --warn-dim: #F3E4C8;
  --stop: #93373A; --stop-dim: #F7E7E6;
  /* `inverted` gets its own colour rather than sharing `dead`'s red:
     the two are opposite findings and a reader scanning chips should
     not have to read the text to tell them apart. */
  --invert: #6B3E8E;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #0D1418; --surface: #141D22; --surface-2: #1B262C;
    --ink: #DCE7EB; --ink-soft: #AEC0C7; --muted: #849AA2;
    --rule: #23323A; --rule-firm: #354952;
    --accent: #5DB4CE; --accent-soft: #3C7E93; --accent-dim: #12303B;
    --warn: #D6A458; --warn-dim: #3A2D11;
    --stop: #DE8C8C; --stop-dim: #32201F;
    --invert: #C39BDA;
  }
}
* { box-sizing: border-box; }
html { background: var(--paper); }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system,
               "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 15.5px; line-height: 1.62; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 40px 26px 90px; }
code, .mono, td.num, th.num {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
  letter-spacing: .13em; text-transform: uppercase; color: var(--muted);
}
h1 {
  font-family: Spectral, Georgia, "Times New Roman", serif; font-weight: 600;
  font-size: clamp(1.85rem, 4vw, 2.5rem); line-height: 1.1;
  letter-spacing: -.015em; margin: 12px 0 0;
}
h2 {
  font-family: Spectral, Georgia, "Times New Roman", serif; font-weight: 600;
  font-size: 1.3rem; margin: 50px 0 2px; letter-spacing: -.01em;
}
.sub { color: var(--ink-soft); margin: 6px 0 0; max-width: 76ch; }
.mast { border-bottom: 1px solid var(--rule-firm); padding-bottom: 22px; }
.facts {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  border-bottom: 1px solid var(--rule-firm); margin: 0;
}
@media (max-width: 720px) { .facts { grid-template-columns: repeat(2, 1fr); } }
.fact { padding: 14px 18px 16px; border-left: 1px solid var(--rule); }
.fact:first-child { border-left: 0; padding-left: 0; }
.fact dt {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 10.5px;
  letter-spacing: .11em; text-transform: uppercase; color: var(--muted);
  margin-bottom: 4px;
}
.fact dd { margin: 0; font-size: 1.5rem; line-height: 1.2; font-weight: 600; }
.fact dd small {
  display: block; font-size: 11.5px; font-weight: 400; color: var(--muted);
  line-height: 1.45;
}
.panel {
  border: 1px solid var(--rule-firm); border-left-width: 4px; border-radius: 3px;
  background: var(--surface); padding: 18px 22px; margin: 26px 0 0;
}
.panel h3 { margin: 0 0 10px; font-size: 1.05rem; }
.panel p { margin: 10px 0 0; max-width: 80ch; }
.panel p:first-of-type { margin-top: 0; }
.panel pre {
  margin: 14px 0 0; padding: 10px 12px; background: var(--surface-2);
  border-radius: 3px; overflow-x: auto; font-size: 12.5px;
}
.panel.refusal { border-color: var(--stop); background: var(--stop-dim); }
.panel.caution { border-color: var(--warn); background: var(--warn-dim); }
.panel.mismatch, .panel.partial { border-color: var(--warn); }
.panel.aligned { border-color: var(--accent); }
.panel.unreadable { border-color: var(--muted); }
figure { margin: 16px 0 0; }
svg.curve { display: block; width: 100%; height: auto; }
svg.curve .band { fill: var(--warn-dim); stroke: var(--warn); stroke-width: 1; }
svg.curve .grid { stroke: var(--rule); stroke-width: 1; }
svg.curve .axis { stroke: var(--rule-firm); stroke-width: 1; }
svg.curve .respondent { stroke: var(--warn); stroke-width: 1; opacity: .34; }
svg.curve .rug { stroke: var(--warn); stroke-width: 1.8; }
svg.curve .info { fill: none; stroke: var(--accent); stroke-width: 2.2; }
svg.curve .info-area { fill: var(--accent); opacity: .15; stroke: none; }
svg.curve .info-all {
  fill: none; stroke: var(--accent-soft); stroke-width: 1.3; stroke-dasharray: 5 3;
}
svg.curve .peak-rule {
  stroke: var(--accent); stroke-width: 1; stroke-dasharray: 3 3; opacity: .8;
}
svg.curve .peak-dot { fill: var(--accent); }
svg.curve text {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
svg.curve .tick { font-size: 11px; fill: var(--muted); }
svg.curve .rug-label { font-size: 10px; fill: var(--muted); }
svg.curve .axis-title { font-size: 11.5px; fill: var(--ink-soft); }
svg.curve .callout { font-size: 12px; font-weight: 500; }
svg.curve .callout.peak { fill: var(--accent); }
svg.curve .callout.models { fill: var(--warn); }
svg.curve .start { text-anchor: start; }
svg.curve .middle { text-anchor: middle; }
svg.curve .end { text-anchor: end; }
figcaption { font-size: 12.5px; color: var(--muted); margin-top: 10px; max-width: 92ch; }
.key { display: flex; flex-wrap: wrap; gap: 4px 20px; margin: 0 0 8px; font-size: 12.5px; }
.key span { display: inline-flex; align-items: center; gap: 7px; color: var(--ink-soft); }
.swatch { display: inline-block; width: 22px; height: 0; border-top: 2px solid var(--accent); }
.swatch.dashed { border-top: 2px dashed var(--accent-soft); }
.swatch.fill {
  height: 11px; width: 22px; border: 0; background: var(--warn-dim);
  box-shadow: inset 0 0 0 1px var(--warn);
}
.swatch.rug { height: 12px; width: 2px; border: 0; background: var(--warn); }
.grid-rows { border-top: 1px solid var(--rule-firm); margin: 14px 0 0; }
.grid-rows > div {
  display: grid; grid-template-columns: 168px minmax(0, 1fr); gap: 18px;
  padding: 11px 0; border-bottom: 1px solid var(--rule);
}
@media (max-width: 620px) { .grid-rows > div { grid-template-columns: 1fr; gap: 2px; } }
.grid-rows dt {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
  letter-spacing: .06em; text-transform: uppercase; color: var(--muted); padding-top: 3px;
}
.grid-rows dd { margin: 0; }
.grid-rows dd small {
  display: block; color: var(--muted); font-size: 12.5px; line-height: 1.5;
}
.tablewrap { overflow-x: auto; margin-top: 14px; }
table { border-collapse: collapse; font-size: 13px; width: 100%; }
caption {
  text-align: left; color: var(--muted); font-size: 12.5px; padding-bottom: 8px;
  max-width: 92ch;
}
th, td {
  padding: 6px 14px 6px 0; border-bottom: 1px solid var(--rule);
  text-align: left; white-space: nowrap;
}
thead th {
  border-bottom: 1px solid var(--rule-firm); font-size: 11px; letter-spacing: .07em;
  text-transform: uppercase; color: var(--muted); font-weight: 500; vertical-align: bottom;
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
}
td.num, th.num { text-align: right; }
th button {
  font: inherit; color: inherit; background: none; border: 0; padding: 0;
  cursor: pointer; letter-spacing: inherit; text-transform: inherit;
}
th button:hover { color: var(--ink); text-decoration: underline; }
th[aria-sort] button { color: var(--ink); text-decoration: underline; }
tr.unrankable td { color: var(--muted); }
.chip {
  display: inline-block; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: .04em; padding: 1px 6px; border-radius: 2px;
  border: 1px solid var(--rule-firm); color: var(--ink-soft); margin-right: 4px;
}
.chip.dead { border-color: var(--stop); color: var(--stop); }
.chip.inverted { border-color: var(--invert); color: var(--invert); }
.chip.unknown { border-color: var(--warn); color: var(--warn); }
.chip.off { border-color: var(--accent-soft); color: var(--accent); }
.legend { font-size: 12.5px; color: var(--muted); margin-top: 14px; max-width: 92ch; }
.legend b { color: var(--ink-soft); font-weight: 500; }
footer {
  margin-top: 50px; padding-top: 16px; border-top: 1px solid var(--rule-firm);
  color: var(--muted); font-size: 12.5px;
}
footer p { margin: 0 0 4px; }
"""

SCRIPT = """
(function () {
  var table = document.getElementById('items');
  if (!table || !table.tBodies.length || !table.tHead) { return; }
  var body = table.tBodies[0];
  var rows = Array.prototype.slice.call(body.rows);
  function attr(row, name) { return row.getAttribute('data-' + name) || ''; }
  function number(row, name) {
    var value = parseFloat(attr(row, name));
    return isNaN(value) ? 0 : value;
  }
  function order(key, dir) {
    return function (a, b) {
      /* Sorting by discrimination is the ranking sort, and insufficient-data
         items are excluded from the ranking rather than ranked low - however
         flattering their point estimate. Same rule as report.sort_rows. */
      if (key === 'a') {
        var ra = number(a, 'rankable'), rb = number(b, 'rankable');
        if (ra !== rb) { return rb - ra; }
      }
      var d = key === 'item'
        ? attr(a, 'id').localeCompare(attr(b, 'id'))
        : number(a, key) - number(b, key);
      if (d !== 0) { return dir * d; }
      return attr(a, 'id').localeCompare(attr(b, 'id'));
    };
  }
  var heads = table.tHead.rows[0].cells;
  Array.prototype.forEach.call(heads, function (cell) {
    var key = cell.getAttribute('data-key');
    var button = cell.querySelector('button');
    if (!key || !button) { return; }
    button.addEventListener('click', function () {
      var dir = cell.getAttribute('aria-sort') === 'ascending' ? -1 : 1;
      Array.prototype.forEach.call(heads, function (other) {
        other.removeAttribute('aria-sort');
      });
      cell.setAttribute('aria-sort', dir === 1 ? 'ascending' : 'descending');
      rows.slice().sort(order(key, dir)).forEach(function (row) {
        body.appendChild(row);
      });
    });
  });
})();
"""


def _ranks(fit: IrtFit) -> dict[int, int | None]:
    """Rank by discrimination over every item in the fit, not just the shown ones.

    None for insufficient-data. A rank number beside an item whose
    discrimination interval spans zero is exactly the misrepresentation the
    flag exists to prevent, and unlike a row position it would survive every
    re-sort of the table.
    """
    ranks: dict[int, int | None] = {}
    position = 0
    for row in sort_rows(item_rows(fit), "discrimination"):
        if row.rankable:
            position += 1
            ranks[row.index] = position
        else:
            ranks[row.index] = None
    return ranks


def _facts(header: Header) -> str:
    respondents = (
        f"<small>{header.n_respondents} respondents, "
        f"{header.respondents_per_model:.1f} per model</small>"
        if header.has_pseudo_respondents
        else "<small>one respondent each</small>"
    )
    cells = [
        ("real models", str(header.n_real_models), respondents),
        (
            "items",
            str(header.n_items),
            f"<small>{header.n_usable_items} usable for ranking and selection</small>",
        ),
        (
            "dead",
            str(header.dead_count),
            f"<small>a interval wholly below {DEAD_THRESHOLD}</small>",
        ),
        (
            "insufficient-data",
            str(header.insufficient_count),
            "<small>a interval spans zero &mdash; cannot tell</small>",
        ),
    ]
    return "<dl class='facts'>" + "".join(
        f"<div class='fact'><dt>{esc(name)}</dt><dd>{value}{note}</dd></div>"
        for name, value, note in cells
    ) + "</dl>"


def _refusal_panel(refusal: Refusal) -> str:
    if refusal.level == "none":
        return ""
    title = "cannot rank these items yet" if refusal.is_refusal else "read the ranking with care"
    detail = "".join(f"<p>{esc(line)}</p>" for line in refusal.detail)
    lead, _, command = refusal.next_step.partition("\n")
    command_html = f"<pre>{esc(command.strip())}</pre>" if command.strip() else ""
    kind = "refusal" if refusal.is_refusal else "caution"
    return (
        f"<section class='panel {kind}'><h3>{esc(title)}</h3>"
        f"<p><b>{esc(refusal.headline)}</b></p>{detail}"
        f"<p>{esc(lead)}</p>{command_html}</section>"
    )


def _verdict_panel(curve: Curve, header: Header) -> str:
    lines = "".join(f"<p>{esc(line)}</p>" for line in verdict_lines(curve, header))
    return (
        f"<section class='panel {esc(curve.verdict)}'>"
        f"<h3>{esc(VERDICT_TITLES[curve.verdict])}</h3>{lines}</section>"
    )


def _plot_figure(curve: Curve) -> str:
    key: list[tuple[str, str]] = [
        ("<span class='swatch'></span>", "test information, over the items this fit can read"),
    ]
    if curve.show_all_items:
        key.append(
            ("<span class='swatch dashed'></span>", "every item, unreadable ones included")
        )
    key += [
        ("<span class='swatch fill'></span>", "ability range the models span"),
        ("<span class='swatch rug'></span>", "one respondent, at its posterior θ mean"),
    ]
    legend = "".join(f"<span>{swatch}{esc(label)}</span>" for swatch, label in key)
    return (
        f"<figure><div class='key'>{legend}</div>{render_svg(curve)}"
        "<figcaption>Information is additive over items and peaks at each item's "
        "difficulty, so the teal curve is where this suite can tell two models apart. "
        "The amber marks are the respondents of this matrix, at the posterior &theta; "
        "means the fit gave them &mdash; not a grid. Teal with no amber under it is "
        "precision spent on ability nobody here has."
        "</figcaption></figure>"
    )


def _header_rows(header: Header) -> str:
    key = esc(",".join(header.respondent_key))
    rows: list[tuple[str, str]] = []

    if header.has_pseudo_respondents:
        rows.append(
            (
                "respondents",
                f"<b>{header.n_real_models} real models</b> &rarr; "
                f"{header.n_respondents} respondents "
                f"({header.respondents_per_model:.1f} per model, "
                f"<code>--respondent-key {key}</code>)"
                "<small>pseudo-respondents inform item parameters; they are not model "
                "rankings</small>",
            )
        )
    else:
        rows.append(
            (
                "respondents",
                f"<b>{header.n_real_models} real models</b>, one respondent each "
                f"(<code>--respondent-key {key}</code>)",
            )
        )

    items = (
        f"<b>{header.n_items} items</b> "
        f"({header.n_usable_items} usable for ranking and selection)"
    )
    if header.n_responses is not None:
        density = f", {header.density:.0%} dense" if header.density is not None else ""
        items += f"<small>{header.n_responses} responses{density}</small>"
    rows.append(("items", items))

    # Never one "bad items" number: these are different claims about different
    # things and the counts are never added together.
    rows.append(
        (
            "dead",
            f"<b>{header.dead_count}</b> ({header.flag_rates[FLAG_DEAD]:.1%}) confidently "
            "do not discriminate &mdash; the whole a interval lies inside "
            f"&plusmn;{DEAD_THRESHOLD}<small>a finding about the suite</small>",
        )
    )
    if header.flag_counts[FLAG_INVERTED]:
        rows.append(
            (
                "inverted",
                f"<b>{header.flag_counts[FLAG_INVERTED]}</b> "
                f"({header.flag_rates[FLAG_INVERTED]:.1%}) discriminate "
                "<em>backwards</em> &mdash; the a interval lies wholly below zero, so "
                "weaker respondents get these right more often<small>check the answer "
                "key before dropping them: this is information pointing the wrong way, "
                "not dead weight</small>",
            )
        )
    rows.append(
        (
            "insufficient-data",
            f"<b>{header.insufficient_count}</b> ({header.insufficient_share:.1%}) cannot "
            "be told either way &mdash; the a interval spans zero; excluded from ranking "
            "and selection<small>a finding about the data supplied, and never also "
            "dead</small>",
        )
    )
    rows.append(
        (
            "ceiling / floor",
            f"ceiling {header.flag_counts[FLAG_CEILING]} ({header.ceiling_rate:.1%}, "
            f"p &ge; {CEILING_THRESHOLD})&nbsp;&nbsp;&nbsp;"
            f"floor {header.flag_counts[FLAG_FLOOR]} ({header.floor_rate:.1%}, "
            f"p &le; {FLOOR_THRESHOLD})",
        )
    )
    rows.append(
        (
            "off-range",
            f"{header.flag_counts[FLAG_OFF_RANGE]} "
            f"({header.flag_rates[FLAG_OFF_RANGE]:.1%}) have a difficulty outside the "
            "ability range these respondents occupy",
        )
    )
    if header.score_threshold is not None:
        rows.append(
            (
                "score threshold",
                f"<b>continuous scores were binarised at {header.score_threshold:g}</b> "
                "&mdash; &lsquo;correct&rsquo; here means raw_score &ge; that",
            )
        )
    diagnostics = "<br>".join(esc(line) for line in _diagnostics_line(header).split("\n"))
    rows.append(("fit", f"<span class='mono'>{diagnostics}</span>"))

    return "<dl class='grid-rows'>" + "".join(
        f"<div><dt>{esc(name)}</dt><dd>{value}</dd></div>" for name, value in rows
    ) + "</dl>"


def _models_table(curve: Curve) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{esc(model.model_id)}</td>"
        f"<td class='num'>{model.n_respondents}</td>"
        f"<td class='num'>{model.theta:+.2f}</td>"
        f"<td class='num'>{model.information:.2f}</td>"
        f"<td class='num'>{model.theta_se:.2f}</td>"
        "</tr>"
        for model in sorted(curve.models, key=lambda m: m.model_id)
    )
    return (
        "<div class='tablewrap'><table>"
        "<caption>Ordered by model id, deliberately not by &theta;: abilities are "
        "nuisance parameters in this fit, and the ranking claim this tool makes is the "
        "holdout rank correlation from <code>irtcheck validate</code>, not this "
        "column.</caption>"
        "<thead><tr><th>model</th><th class='num'>respondents</th>"
        "<th class='num'>&theta;</th><th class='num'>I(&theta;)</th>"
        "<th class='num'>&theta; SE</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _items_table(report: Report, ranks: dict[int, int | None]) -> str:
    heads = ["<th class='num'>rank</th>"]
    for column in COLUMNS:
        classes = " class='num'" if column.justify == "right" else ""
        label = esc(column.heading)
        key = SORTABLE.get(column.heading)
        if key:
            heads.append(
                f"<th{classes} data-key='{esc(key)}'>"
                f"<button type='button'>{label}</button></th>"
            )
        else:
            heads.append(f"<th{classes}>{label}</th>")

    body: list[str] = []
    for row in report.rows:
        rank = ranks.get(row.index)
        cells = [f"<td class='num rank'>{rank if rank is not None else '&mdash;'}</td>"]
        for column in COLUMNS:
            if column.heading == "flags":
                content = "".join(
                    f"<span class='chip {FLAG_CLASSES.get(flag, '')}'>{esc(flag)}</span>"
                    for flag in row.flags
                )
            else:
                content = esc(column.render(row))
            attrs = " class='num'" if column.justify == "right" else ""
            cells.append(f"<td{attrs}>{content}</td>")
        classes = " class='unrankable'" if not row.rankable else ""
        body.append(
            f"<tr{classes} data-id='{esc(row.item_id)}' data-a='{row.a_mean:.6f}' "
            f"data-b='{row.b_mean:.6f}' data-rankable='{1 if row.rankable else 0}'>"
            + "".join(cells)
            + "</tr>"
        )

    return (
        "<div class='tablewrap'><table id='items'>"
        f"<thead><tr>{''.join(heads)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _footer_lines(report: Report, curve: Curve) -> list[str]:
    header = report.header
    shown, total = len(report.rows), report.n_items_total
    lines = [
        f"Showing {shown} of {total} items, sorted by {esc(report.sort)}. Re-run with "
        "<code>--limit 0</code> for every item."
        if shown < total
        else f"All {total} items, sorted by {esc(report.sort)}."
    ]
    if header.insufficient_count:
        lines.append(
            f"{header.insufficient_count} insufficient-data items carry no rank and sort "
            "last: they are excluded from the ranking, not ranked low."
        )
    lines.append(
        f"Rank is by posterior mean discrimination over the "
        f"{total - header.insufficient_count} rankable items. Information is summed over "
        f"the {curve.n_pool} items <code>irtcheck select</code> can choose from."
    )
    provenance = f"irtcheck {esc(header.irtcheck_version)} &middot; fit written {esc(header.created)}"
    if report.source:
        provenance += f" &middot; {esc(report.source)}"
    lines.append(provenance)
    return lines


def render_html(
    fit: IrtFit,
    *,
    sort: str = DEFAULT_SORT,
    limit: int = 40,
    source: str | None = None,
) -> str:
    """The whole page, as one string. Nothing in it is fetched at open time."""
    report = build_report(fit, sort=sort, limit=limit, source=source)
    curve = build_curve(fit)
    header = report.header

    title = f"irtcheck — {header.n_items} items, {header.n_real_models} models"
    footer = "".join(f"<p>{line}</p>" for line in _footer_lines(report, curve))

    return (
        "<!doctype html>\n"
        "<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{esc(title)}</title>\n"
        f"<style>{STYLE}</style>\n"
        "</head>\n<body>\n<div class='wrap'>\n"
        "<header class='mast'>"
        "<div class='eyebrow'>irtcheck &middot; eval suite health check</div>"
        "<h1>Is this suite measuring the models you have?</h1>"
        "<p class='sub'>A 2PL fitted to a response matrix. Below: where this suite's "
        "measurement precision actually sits, which items carry it, and which items this "
        "much data cannot speak for either way.</p>"
        "</header>\n"
        f"{_facts(header)}\n"
        f"{_refusal_panel(report.refusal)}\n"
        "<h2>Test information against the abilities in this matrix</h2>"
        "<p class='sub'>The plot this tool exists for.</p>\n"
        f"{_plot_figure(curve)}\n"
        f"{_verdict_panel(curve, header)}\n"
        "<h2>The suite</h2>\n"
        f"{_header_rows(header)}\n"
        "<h2>Where the respondents sit</h2>"
        "<p class='sub'>The amber marks above, as numbers. &theta; SE is the ability "
        "standard error the whole suite achieves for that model under the "
        "&theta; ~ N(0, 1) this fit identifies.</p>\n"
        f"{_models_table(curve)}\n"
        "<h2>Items</h2>\n"
        f"{_items_table(report, _ranks(fit))}\n"
        "<p class='legend'><b>a</b> discrimination, how sharply the item separates around "
        "its difficulty &middot; <b>b</b> difficulty, the ability at which the item is a "
        "coin flip &middot; <b>HDI</b> 95% credible interval &middot; <b>n</b> responses "
        "&middot; <b>dead</b> confidently does not discriminate &middot; "
        "<b>inverted</b> confidently discriminates backwards &mdash; weaker "
        "respondents get it right more often &middot; "
        "<b>insufficient-data</b> cannot be told either way, so not ranked &middot; "
        "<b>ceiling</b>/<b>floor</b> everyone got it right, or everyone got it wrong "
        "&middot; <b>off-range</b> difficulty outside the ability range these respondents "
        "occupy.</p>\n"
        f"<footer>{footer}</footer>\n"
        f"</div>\n<script>{SCRIPT}</script>\n</body>\n</html>\n"
    )


def write_html(
    fit: IrtFit,
    path: str | Path,
    *,
    sort: str = DEFAULT_SORT,
    limit: int = 40,
    source: str | None = None,
) -> Path:
    """Render and write the page. Returns the path, for the caller to print."""
    path = Path(path)
    path.write_text(render_html(fit, sort=sort, limit=limit, source=source), encoding="utf-8")
    return path
