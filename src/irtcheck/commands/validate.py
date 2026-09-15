"""`irtcheck validate` — owned by wave1/select-validate.

The headline command. It refits the 2PL once per held-out model, which is the
slowest thing the tool does and the reason the artifact carries its own
responses: `irtcheck validate suite.irt` has to work with one argument rather
than depending on the user still having the original JSONL, unchanged, beside
the artifact.

Plain summary by default, `--json` for machines, per the spec's output rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from irtcheck.artifact import ArtifactError, IrtFit
from irtcheck.validate import (
    ValidateError,
    ValidationReport,
    default_fit_fn,
    leave_one_model_out,
    parse_sizes,
)


def run(
    *,
    artifact: Path,
    sizes: str = "25,50,100,200,400",
    as_json: bool = False,
    seed: int = 0,
    epochs: int = 2000,
    **_: Any,
) -> None:
    err = Console(stderr=True)
    out = Console()

    try:
        fit = IrtFit.load(artifact)
        wanted = parse_sizes(sizes)
        # Raises with the "written with --no-embed-responses" explanation when
        # the artifact cannot be refitted, which is the whole point of that
        # message existing.
        matrix = fit.matrix()
    except (ArtifactError, ValidateError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    def announce(model_id: str, position: int, total: int) -> None:
        err.print(f"[dim]Refitting without {model_id} ({position + 1}/{total})…[/dim]")

    try:
        report = leave_one_model_out(
            matrix,
            fit_fn=default_fit_fn(seed=seed, epochs=epochs),
            sizes=wanted,
            on_model=None if as_json else announce,
        )
    except ValidateError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    if as_json:
        print(json.dumps(report.to_dict() | {"source": str(artifact)}, indent=2))
        return
    _render(out, report)


def _fmt(value: float) -> str:
    return "—" if value != value else f"{value:+.3f}"


def _render(out: Console, report: ValidationReport) -> None:
    # Real models, not respondents: the count of things being ranked is what
    # makes or breaks a rank correlation, and it is not the respondent count.
    out.print(
        f"[bold]Leave-one-model-out[/bold] · {report.n_models} models · "
        f"{report.n_items} items · {report.n_respondents} respondents "
        f"(key {'+'.join(report.respondent_key)})"
    )

    table = Table(title=None, header_style="bold")
    table.add_column("n", justify="right")
    table.add_column("items", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("Kendall tau", justify="right")
    table.add_column("Spearman (theta)", justify="right")
    table.add_column("tau (theta)", justify="right")
    for result in report.results:
        table.add_row(
            str(result.size),
            f"{result.min_selected}{'*' if result.short else ''}",
            _fmt(result.spearman),
            _fmt(result.kendall),
            f"[dim]{_fmt(result.spearman_theta)}[/dim]",
            f"[dim]{_fmt(result.kendall_theta)}[/dim]",
        )
    out.print(table)

    out.print(
        "Correlations are between each model's rank on the anchor set — chosen by a fit "
        "that never saw it — and its rank on the full suite.\n"
        "[bold]Spearman and Kendall tau are plain accuracy over the anchor items[/bold], "
        "which is what you do with an anchor set once you have it. The dim columns "
        "re-estimate ability from the same responses with item parameters held fixed; "
        "the two diverge when an anchor set skews hard or easy."
    )
    if any(r.short for r in report.results):
        out.print(
            "[yellow]*[/yellow] fewer items were available than requested, even after padding "
            "with insufficient-data items — the rest are inverted, at ceiling or floor, "
            "unanswered, or lean backwards, and are never selected."
        )
    if report.n_models < 6:
        out.print(
            f"[yellow]{report.n_models} models is thin for a rank correlation[/yellow]: a "
            "single swapped pair moves tau a long way. Read the trend across n rather than "
            "any one number."
        )
