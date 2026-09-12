"""`irtcheck report` — owned by wave1/report.

CLI wiring only: load the artifact, hand it to irtcheck.report, print. The
signature of `run` is fixed by cli.py, which is frozen — it is called with
exactly the keywords declared there.

Nothing here imports torch, pyro or irtcheck.fit, directly or transitively.
Reporting on a cached fit has to work on a machine that has never installed
them; see CLAUDE.md → The lazy torch boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from irtcheck.artifact import ArtifactError, IrtFit
from irtcheck.html import write_html
from irtcheck.report import DEFAULT_SORT, ReportError, build_report, render


def run(
    *,
    artifact: Path,
    html: Path | None = None,
    as_json: bool = False,
    sort: str = DEFAULT_SORT,
    limit: int = 40,
) -> None:
    console = Console()
    errors = Console(stderr=True)

    try:
        fit = IrtFit.load(artifact)
    except ArtifactError as exc:
        errors.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(2) from exc

    try:
        report = build_report(fit, sort=sort, limit=limit, source=str(artifact))
    except ReportError as exc:
        errors.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(2) from exc

    if html is not None:
        write_html(fit, html, sort=sort, limit=limit, source=str(artifact))
        errors.print(f"[dim]wrote {html}[/]")  # stderr: --json stays pipeable

    if as_json:
        # print(), not console.print(): rich would wrap and highlight it, and
        # this stream is meant to be piped into jq.
        print(json.dumps(report.to_json(), indent=2))
        return

    render(report, console)
