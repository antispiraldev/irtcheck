"""Terminal report: what the fit knows, and — louder — what it does not.

Reads a cached `IrtFit` and nothing else. No torch, no pyro, no `irtcheck.fit`
anywhere in this module's import path; see CLAUDE.md → The lazy torch boundary.

Three things here are deliberate and are the reason this is not a
table-printing exercise.

**The refusal is printed above the report, not below it.** At five to fifteen
real models most low-information items land in `insufficient-data`, and a
report that shows a confident-looking ranking of 400 items over a banner-free
header invites the user to act on an order the data does not support. When the
insufficient-data share crosses REFUSAL_SHARE the report leads with the
refusal and names `--respondent-key` — the cheapest way to raise respondent
count — before it shows a single parameter.

**`dead` and `insufficient-data` are counted and worded separately, never
summed.** `dead` is a finding about the suite: we are confident this item does
not discriminate. `insufficient-data` is a finding about the data we were
given: we cannot tell, because the `a` interval spans zero. A single "bad
items: 316" line would erase exactly the distinction the tool exists to make,
and would make the honest small-N answer look like a broken run.

**The header counts real models, not respondents.** `--respondent-key
model_id,prompt_variant` turns five models into sixty respondents; printing
sixty would misrepresent the one thing this tool is here to be honest about.
Both numbers are shown when they differ, with the ratio between them.

Columns are declared once, in COLUMNS, and both the rich table and the `--json`
payload are generated from that list. A column the table shows and the JSON
omits is therefore not expressible, which is the property `--json` needs to be
worth scripting against.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.box import SIMPLE_HEAD
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from irtcheck.artifact import (
    ALL_FLAGS,
    CEILING_THRESHOLD,
    DEAD_THRESHOLD,
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
    FLAG_OFF_RANGE,
    FLOOR_THRESHOLD,
    IrtFit,
)

SORT_KEYS = ("discrimination", "difficulty", "id")
DEFAULT_SORT = "discrimination"

# Share of items flagged insufficient-data at which the report stops leading
# with numbers and leads with the refusal instead. Conventions, not laws: half
# the suite unrankable makes the ranking misleading to read at all, a fifth
# makes it worth a caution line above the table.
REFUSAL_SHARE = 0.5
CAUTION_SHARE = 0.2

# Below this many *real models* the fit is thin however many pseudo-respondents
# were composed from them, because pseudo-respondents buy information about
# items and almost none about the ability scale. The spec leaves the exact
# number to be picked empirically (see docs/spec.md, Open questions); five is
# the bottom of the range the spec assumes and is used as a caution trigger,
# never as a hard refusal to produce output.
THIN_MODEL_COUNT = 5

# Fields a user is likely to have that would compose a finer respondent key,
# in the order worth suggesting. `prompt_variant` first: it is in the input
# contract and costs one re-run of the harness with a second prompt template.
RESPONDENT_KEY_SUGGESTIONS = ("prompt_variant", "temperature", "seed", "checkpoint", "quantization")

# Keys under which a fitter may record that a continuous `raw_score` was
# thresholded into a binary `correct`. CLAUDE.md requires the threshold to be
# printed in the report header rather than applied quietly, so the header looks
# for all of the plausible spellings and prints whichever it finds.
THRESHOLD_KEYS = ("score_threshold", "binarize_threshold", "threshold")

FLAG_STYLES = {
    FLAG_DEAD: "bold red",
    FLAG_INSUFFICIENT_DATA: "yellow",
    FLAG_CEILING: "cyan",
    FLAG_FLOOR: "cyan",
    FLAG_OFF_RANGE: "magenta",
}


class ReportError(ValueError):
    """A bad flag value. Raised before anything is printed."""


# -- rows --------------------------------------------------------------------


@dataclass(slots=True)
class ItemRow:
    """One item, flattened out of the parallel posterior arrays."""

    index: int
    item_id: str
    a_mean: float
    a_sd: float
    a_hdi_low: float
    a_hdi_high: float
    b_mean: float
    b_sd: float
    b_hdi_low: float
    b_hdi_high: float
    n_resp: int
    p_correct: float
    flags: list[str]

    @property
    def rankable(self) -> bool:
        """False for insufficient-data: the spec excludes those from ranking
        rather than ranking them anyway."""
        return FLAG_INSUFFICIENT_DATA not in self.flags

    @property
    def dead(self) -> bool:
        return FLAG_DEAD in self.flags


def item_rows(fit: IrtFit) -> list[ItemRow]:
    return [
        ItemRow(
            index=i,
            item_id=fit.item_ids[i],
            a_mean=fit.a.mean[i],
            a_sd=fit.a.sd[i],
            a_hdi_low=fit.a.hdi_low[i],
            a_hdi_high=fit.a.hdi_high[i],
            b_mean=fit.b.mean[i],
            b_sd=fit.b.sd[i],
            b_hdi_low=fit.b.hdi_low[i],
            b_hdi_high=fit.b.hdi_high[i],
            n_resp=fit.n_resp[i],
            p_correct=fit.p_correct[i],
            flags=list(fit.flags[i]),
        )
        for i in range(fit.n_items)
    ]


# -- columns -----------------------------------------------------------------


def _num(value: float, places: int = 2, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:{'+' if sign else ''}.{places}f}"


def _interval(low: float, high: float) -> str:
    return f"[{_num(low)}, {_num(high)}]"


@dataclass(frozen=True, slots=True)
class Column:
    """One table column and the JSON keys carrying the same information.

    `keys`/`values` exist so that `--json` cannot drift from the table: a
    column whose interval renders as one cell still contributes both of its
    endpoints to the JSON row, and tests assert that every column contributes
    at least one key to every emitted record.
    """

    heading: str
    justify: str
    render: Callable[[ItemRow], str]
    keys: tuple[str, ...]
    values: Callable[[ItemRow], tuple[Any, ...]]
    style: str | None = None


COLUMNS: tuple[Column, ...] = (
    Column(
        heading="item",
        justify="left",
        render=lambda r: r.item_id,
        keys=("item_id",),
        values=lambda r: (r.item_id,),
    ),
    Column(
        heading="a",
        justify="right",
        render=lambda r: _num(r.a_mean),
        keys=("a",),
        values=lambda r: (r.a_mean,),
    ),
    Column(
        heading="a 95% HDI",
        justify="right",
        render=lambda r: _interval(r.a_hdi_low, r.a_hdi_high),
        keys=("a_hdi_low", "a_hdi_high"),
        values=lambda r: (r.a_hdi_low, r.a_hdi_high),
        style="dim",
    ),
    Column(
        heading="b",
        justify="right",
        render=lambda r: _num(r.b_mean, sign=True),
        keys=("b",),
        values=lambda r: (r.b_mean,),
    ),
    Column(
        heading="b 95% HDI",
        justify="right",
        render=lambda r: _interval(r.b_hdi_low, r.b_hdi_high),
        keys=("b_hdi_low", "b_hdi_high"),
        values=lambda r: (r.b_hdi_low, r.b_hdi_high),
        style="dim",
    ),
    Column(
        heading="n",
        justify="right",
        render=lambda r: str(r.n_resp),
        keys=("n_resp",),
        values=lambda r: (r.n_resp,),
    ),
    Column(
        heading="p(correct)",
        justify="right",
        render=lambda r: _num(r.p_correct),
        keys=("p_correct",),
        values=lambda r: (r.p_correct,),
    ),
    Column(
        heading="flags",
        justify="left",
        render=lambda r: " ".join(r.flags),
        keys=("flags",),
        values=lambda r: (list(r.flags),),
    ),
)

# Not columns: nobody reads a posterior sd off a terminal table when the
# interval is next to it, but a script computing its own intervals wants them.
EXTRA_JSON_KEYS = ("a_sd", "b_sd", "rankable", "dead")


def row_json(row: ItemRow) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for column in COLUMNS:
        record.update(dict(zip(column.keys, column.values(row), strict=True)))
    record["a_sd"] = row.a_sd
    record["b_sd"] = row.b_sd
    record["rankable"] = row.rankable
    record["dead"] = row.dead
    return record


# -- header ------------------------------------------------------------------


@dataclass(slots=True)
class Header:
    """Everything the spec requires above the table, computed once.

    `n_real_models` and `n_respondents` are both carried deliberately: they are
    equal until someone passes `--respondent-key`, and after that only the
    first one is a count of things that answered independently.
    """

    n_real_models: int
    n_respondents: int
    has_pseudo_respondents: bool
    respondent_key: list[str]
    respondents_per_model: float
    n_items: int
    n_usable_items: int
    flag_counts: dict[str, int]
    flag_rates: dict[str, float]
    n_responses: int | None
    density: float | None
    model: dict[str, Any]
    diagnostics: dict[str, Any]
    score_threshold: float | None
    created: str
    irtcheck_version: str

    @property
    def dead_count(self) -> int:
        return self.flag_counts[FLAG_DEAD]

    @property
    def insufficient_count(self) -> int:
        return self.flag_counts[FLAG_INSUFFICIENT_DATA]

    @property
    def insufficient_share(self) -> float:
        return self.flag_rates[FLAG_INSUFFICIENT_DATA]

    @property
    def ceiling_rate(self) -> float:
        return self.flag_rates[FLAG_CEILING]

    @property
    def floor_rate(self) -> float:
        return self.flag_rates[FLAG_FLOOR]

    def to_json(self) -> dict[str, Any]:
        return {
            "n_real_models": self.n_real_models,
            "n_respondents": self.n_respondents,
            "has_pseudo_respondents": self.has_pseudo_respondents,
            "respondent_key": list(self.respondent_key),
            "respondents_per_model": self.respondents_per_model,
            "n_items": self.n_items,
            "n_usable_items": self.n_usable_items,
            "flag_counts": dict(self.flag_counts),
            "flag_rates": dict(self.flag_rates),
            "dead_count": self.dead_count,
            "insufficient_data_count": self.insufficient_count,
            "insufficient_data_share": self.insufficient_share,
            "ceiling_rate": self.ceiling_rate,
            "floor_rate": self.floor_rate,
            "n_responses": self.n_responses,
            "density": self.density,
            "model": dict(self.model),
            # Summarised, not raw: see summarise_diagnostics. The full ELBO
            # trace lives in the artifact, which is where a caller who wants it
            # should read it from.
            "diagnostics": summarise_diagnostics(self.diagnostics),
            "score_threshold": self.score_threshold,
            "created": self.created,
            "irtcheck_version": self.irtcheck_version,
            "dead_threshold": DEAD_THRESHOLD,
            "ceiling_threshold": CEILING_THRESHOLD,
            "floor_threshold": FLOOR_THRESHOLD,
        }


def _score_threshold(fit: IrtFit) -> float | None:
    """The cutoff a continuous `raw_score` was binarised at, if any.

    CLAUDE.md: thresholding is a documented escape hatch that has to print its
    threshold in the report header, not a quiet cast. The fitter owns the key
    it records this under, so look for every plausible spelling rather than
    silently printing nothing.
    """
    for source in (fit.diagnostics, fit.model):
        for key in THRESHOLD_KEYS:
            value = source.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    return None


def build_header(fit: IrtFit) -> Header:
    counts = {flag: len(fit.flagged(flag)) for flag in ALL_FLAGS}
    n_items = fit.n_items
    n_responses = len(fit.responses) if fit.responses is not None else None
    cells = fit.n_respondents * n_items
    return Header(
        n_real_models=fit.n_real_models,
        n_respondents=fit.n_respondents,
        has_pseudo_respondents=fit.has_pseudo_respondents,
        respondent_key=list(fit.respondent_key),
        respondents_per_model=(
            fit.n_respondents / fit.n_real_models if fit.n_real_models else 0.0
        ),
        n_items=n_items,
        n_usable_items=len(fit.usable_items()),
        flag_counts=counts,
        flag_rates={f: (c / n_items if n_items else 0.0) for f, c in counts.items()},
        n_responses=n_responses,
        density=(n_responses / cells if n_responses is not None and cells else None),
        model=dict(fit.model),
        diagnostics=dict(fit.diagnostics),
        score_threshold=_score_threshold(fit),
        created=fit.created,
        irtcheck_version=fit.irtcheck_version,
    )


# -- refusal -----------------------------------------------------------------


def suggested_respondent_key(current: list[str]) -> str:
    """A concrete `--respondent-key` the user could pass next.

    Advice with a field name in it gets acted on; "consider more respondents"
    does not. The suggestion extends whatever key produced this fit rather than
    replacing it, because model_id has to stay in for holdout to work.
    """
    fields = list(current) or ["model_id"]
    for candidate in RESPONDENT_KEY_SUGGESTIONS:
        if candidate not in fields:
            return ",".join([*fields, candidate])
    return ",".join(fields)


@dataclass(slots=True)
class Refusal:
    """Whether, and how loudly, to say "you need more respondents".

    level is one of "none", "caution", "refusal". Only "refusal" is printed
    above the header; "caution" is a single line under it. Neither is an error:
    the report still renders every item.
    """

    level: str
    headline: str = ""
    detail: list[str] = field(default_factory=list)
    next_step: str = ""
    suggested_respondent_key: str = ""

    @property
    def is_refusal(self) -> bool:
        return self.level == "refusal"

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "headline": self.headline,
            "detail": list(self.detail),
            "next_step": self.next_step,
            "suggested_respondent_key": self.suggested_respondent_key,
            "refusal_share_threshold": REFUSAL_SHARE,
            "caution_share_threshold": CAUTION_SHARE,
            "thin_model_count": THIN_MODEL_COUNT,
        }


def build_refusal(header: Header) -> Refusal:
    share = header.insufficient_share
    thin_models = header.n_real_models < THIN_MODEL_COUNT
    if share >= REFUSAL_SHARE:
        level = "refusal"
    elif share >= CAUTION_SHARE or thin_models:
        level = "caution"
    else:
        return Refusal(level="none", suggested_respondent_key=suggested_respondent_key(header.respondent_key))

    key = suggested_respondent_key(header.respondent_key)
    pct = f"{share:.0%}"
    models = f"{header.n_real_models} real model{'s' if header.n_real_models != 1 else ''}"

    if level == "refusal":
        headline = (
            f"Not enough respondents to rank these items: {header.insufficient_count} of "
            f"{header.n_items} items ({pct}) are insufficient-data."
        )
    else:
        headline = (
            f"Ranking is thin: {header.insufficient_count} of {header.n_items} items "
            f"({pct}) are insufficient-data."
        )

    # `dead` and `insufficient-data` are different claims and the panel has to
    # keep them apart even when one of the counts is zero — a rare `dead` count
    # at this respondent count is the expected result, not a sign of a bad run.
    if header.dead_count:
        contrast = (
            f"Separately, {header.dead_count} item"
            f"{'s' if header.dead_count != 1 else ''} "
            f"{'are' if header.dead_count != 1 else 'is'} flagged dead: there the fit is "
            "confident the item does not discriminate. That is a finding about the suite."
        )
    else:
        contrast = (
            "No items are flagged dead, which is expected rather than suspicious at this "
            "respondent count: 'dead' needs an interval narrow enough to sit wholly "
            f"inside (0, {DEAD_THRESHOLD}) — it has to clear zero too, or the item is "
            "insufficient-data instead. That caps the standard error at 0.089, which "
            "measures out at roughly a thousand respondents, not a hundred. Below that, "
            "an item that does not discriminate comes back insufficient-data."
        )

    detail = [
        "Their discrimination interval spans zero, so this fit cannot tell whether they "
        "separate stronger models from weaker ones. That is a statement about the data "
        "supplied, not a verdict on the items.",
        contrast,
        f"This fit has {models}. Item parameters sharpen roughly with the square root of "
        "respondent count, so the returns on adding respondents are large down here.",
    ]
    if thin_models:
        detail.append(
            f"Fewer than {THIN_MODEL_COUNT} real models is below the range this tool assumes; "
            "treat every number below as provisional."
        )
    if header.has_pseudo_respondents:
        detail.append(
            f"The {header.n_respondents} respondents already include pseudo-respondents "
            f"composed with --respondent-key {','.join(header.respondent_key)}; another axis "
            "would add more."
        )

    next_step = (
        "Cheapest next step — raise respondent count without running new models. Temperature "
        "samples, prompt variants, checkpoints and quantizations of one model each count as a "
        "separate respondent:\n"
        f"    irtcheck fit responses.jsonl -o suite.irt --respondent-key {key}"
    )
    return Refusal(
        level=level,
        headline=headline,
        detail=detail,
        next_step=next_step,
        suggested_respondent_key=key,
    )


# -- the report --------------------------------------------------------------


def sort_rows(rows: list[ItemRow], sort: str) -> list[ItemRow]:
    """Order items for display.

    Under `discrimination` — the ranking sort — insufficient-data items go last
    however large their point estimate is. Sorting them in among the rest would
    be ranking items the fit has just said it cannot rank, and the top of the
    table is exactly where a wide interval with a flattering mean would land.
    `difficulty` and `id` are plain orderings: they are not claims about signal.
    """
    if sort == "id":
        return sorted(rows, key=lambda r: r.item_id)
    if sort == "difficulty":
        return sorted(rows, key=lambda r: (r.b_mean, r.item_id))
    if sort == "discrimination":
        return sorted(rows, key=lambda r: (not r.rankable, -r.a_mean, r.item_id))
    raise ReportError(
        f"unknown --sort {sort!r}. Choose one of: {', '.join(SORT_KEYS)}."
    )


@dataclass(slots=True)
class Report:
    header: Header
    refusal: Refusal
    rows: list[ItemRow]  # sorted, limited: exactly what the table shows
    n_items_total: int
    sort: str
    limit: int
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "irtcheck_report_version": 1,
            "source": self.source,
            "sort": self.sort,
            "limit": self.limit,
            "items_total": self.n_items_total,
            "items_shown": len(self.rows),
            "columns": [key for column in COLUMNS for key in column.keys],
            "refusal": self.refusal.to_json(),
            "header": self.header.to_json(),
            "items": [row_json(row) for row in self.rows],
        }


def build_report(
    fit: IrtFit,
    *,
    sort: str = DEFAULT_SORT,
    limit: int = 40,
    source: str | None = None,
) -> Report:
    if limit < 0:
        raise ReportError("--limit cannot be negative. Use 0 for every item.")
    rows = sort_rows(item_rows(fit), sort)
    header = build_header(fit)
    return Report(
        header=header,
        refusal=build_refusal(header),
        rows=rows if limit == 0 else rows[:limit],
        n_items_total=fit.n_items,
        sort=sort,
        limit=limit,
        source=source,
    )


# -- rendering ---------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_refusal(refusal: Refusal) -> Panel:
    body = Text()
    body.append(refusal.headline, style="bold")
    for line in refusal.detail:
        body.append("\n\n")
        body.append(line)
    body.append("\n\n")
    body.append(refusal.next_step, style="bold cyan")
    style = "red" if refusal.is_refusal else "yellow"
    title = (
        "cannot rank these items yet" if refusal.is_refusal else "read the ranking with care"
    )
    return Panel(body, title=title, border_style=style, padding=(1, 2))


def render_header(header: Header) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="dim")
    grid.add_column(justify="left")

    if header.has_pseudo_respondents:
        respondents = Text()
        respondents.append(f"{header.n_real_models} real models", style="bold")
        respondents.append(
            f" → {header.n_respondents} respondents "
            f"({header.respondents_per_model:.1f} per model, "
            f"--respondent-key {','.join(header.respondent_key)})"
        )
        respondents.append(
            "\npseudo-respondents inform item parameters; they are not model rankings",
            style="dim italic",
        )
    else:
        respondents = Text(f"{header.n_real_models} real models", style="bold")
        respondents.append(
            f"  (one respondent each, --respondent-key {','.join(header.respondent_key)})",
            style="dim",
        )
    grid.add_row("respondents", respondents)

    items = Text(f"{header.n_items} items", style="bold")
    items.append(f"  ({header.n_usable_items} usable for ranking and selection)")
    if header.n_responses is not None:
        density = f", {header.density:.0%} dense" if header.density is not None else ""
        items.append(f"\n{header.n_responses} responses{density}", style="dim")
    grid.add_row("items", items)

    # Never one "bad items" number. These are different claims about different
    # things and the counts must not be added together.
    dead = Text()
    dead.append(f"{header.dead_count}", style="bold red")
    dead.append(
        f" ({_pct(header.flag_rates[FLAG_DEAD])})  confidently do not discriminate "
        f"(a interval entirely below {DEAD_THRESHOLD})",
    )
    grid.add_row("dead", dead)

    unknown = Text()
    unknown.append(f"{header.insufficient_count}", style="bold yellow")
    unknown.append(
        f" ({_pct(header.insufficient_share)})  cannot tell — a interval spans zero; "
        "excluded from ranking and selection"
    )
    grid.add_row("insufficient-data", unknown)

    rates = Text(
        f"ceiling {header.flag_counts[FLAG_CEILING]} ({_pct(header.ceiling_rate)}, "
        f"p ≥ {CEILING_THRESHOLD})   "
        f"floor {header.flag_counts[FLAG_FLOOR]} ({_pct(header.floor_rate)}, "
        f"p ≤ {FLOOR_THRESHOLD})"
    )
    grid.add_row("ceiling / floor", rates)

    off = header.flag_counts[FLAG_OFF_RANGE]
    grid.add_row(
        "off-range",
        Text(
            f"{off} ({_pct(header.flag_rates[FLAG_OFF_RANGE])})  difficulty outside the ability "
            "range these respondents occupy"
        ),
    )

    if header.score_threshold is not None:
        grid.add_row(
            "score threshold",
            Text(
                f"continuous scores were binarised at {header.score_threshold:g} — "
                "'correct' below means raw_score ≥ that",
                style="bold magenta",
            ),
        )

    grid.add_row("fit", Text(_diagnostics_line(header)))
    return Panel(grid, title="suite", border_style="blue", padding=(1, 2))


# Diagnostics keys that are collections rather than numbers. They are the fit's
# telemetry, they belong in the artifact, and a report is a summary — so the
# report states their shape and not their contents.
def summarise_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Diagnostics with every collection reduced to a description of its size.

    `elbo_history` is the reason this exists. It holds up to 2000 floats, and
    rendering the dict naively put every one of them in the report: 1127 lines
    of terminal output for a 30-item suite, 1056 of them ELBO values, with the
    item table — the thing a reader came for — below the fold. The JSON was
    worse, 2000 of 2654 lines.

    The filter is on *type*, not on the key name. Special-casing `elbo_history`
    would fix today's flood and leave the next diagnostics key that happens to
    be a list free to reintroduce it, which is the same bug with a different
    name. A report renders scalars; anything else is described.
    """
    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if value is None or key == "source":
            continue
        if isinstance(value, str | bytes):
            out[key] = value
        elif isinstance(value, Sequence):
            # Kept as a count and its ends: enough to see that a trace exists
            # and where it got to, without carrying the trace.
            out[f"{key}_points"] = len(value)
        elif isinstance(value, Mapping):
            out[key] = {
                k: v for k, v in value.items() if not isinstance(v, Sequence | Mapping)
            }
        else:
            out[key] = value
    return out


def _scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k}={_scalar(v)}" for k, v in value.items()) + "}"
    return str(value)


def _diagnostics_line(header: Header) -> str:
    model = [str(header.model.get("kind", "?")), f"{header.model.get('priors', '?')} priors"]
    identification = header.model.get("identification")
    if identification:
        model.append(str(identification))

    numbers = [
        f"{key}={_scalar(value)}" for key, value in summarise_diagnostics(header.diagnostics).items()
    ]

    lines = [" · ".join(model)]
    if numbers:
        lines.append(" · ".join(numbers))
    if header.model.get("synthetic"):
        lines.append("SYNTHETIC ARTIFACT — fabricated from known parameters, not fitted")
    lines.append(f"irtcheck {header.irtcheck_version}, written {header.created}")
    return "\n".join(lines)


def render_table(report: Report) -> Table:
    table = Table(box=SIMPLE_HEAD, header_style="bold", expand=False)
    for column in COLUMNS:
        table.add_column(column.heading, justify=column.justify, style=column.style)
    for row in report.rows:
        style = "dim" if not row.rankable else None
        cells = [
            _flag_text(row) if column.heading == "flags" else column.render(row)
            for column in COLUMNS
        ]
        table.add_row(*cells, style=style)
    return table


def _flag_text(row: ItemRow) -> Text:
    text = Text()
    for i, flag in enumerate(row.flags):
        if i:
            text.append(" ")
        text.append(flag, style=FLAG_STYLES.get(flag, ""))
    return text


def render_footer(report: Report) -> Text:
    shown, total = len(report.rows), report.n_items_total
    text = Text()
    if shown < total:
        text.append(
            f"showing {shown} of {total} items, sorted by {report.sort} "
            "(--limit 0 for all)\n",
            style="dim",
        )
    else:
        text.append(f"{total} items, sorted by {report.sort}\n", style="dim")
    if report.sort == "discrimination" and report.header.insufficient_count:
        text.append(
            f"{report.header.insufficient_count} insufficient-data items sort last: they are "
            "excluded from ranking, not ranked low.\n",
            style="dim",
        )
    if report.refusal.level == "caution":
        text.append(report.refusal.headline + " ", style="yellow")
        text.append(
            f"Raise respondent count with --respondent-key {report.refusal.suggested_respondent_key}.",
            style="yellow",
        )
    return text


def render(report: Report, console: Console) -> None:
    """Print the whole thing. Refusal first, then header, then the table."""
    blocks: list[Any] = []
    if report.refusal.is_refusal:
        blocks.append(render_refusal(report.refusal))
    blocks.append(render_header(report.header))
    blocks.append(render_table(report))
    blocks.append(render_footer(report))
    console.print(Group(*blocks))
