"""`irtcheck fit` — owned by wave1/fit.

Read a response file, build the matrix, fit the 2PL, write the artifact. The
slow half of a deliberately two-stage tool: everything after this reads the
cached `.irt` and is instant.

Nothing here imports torch or pyro at module scope, and `irtcheck.fit` is
imported inside `run()`. cli.py already defers importing *this* module, so the
two together are what keep `irtcheck --help` and `irtcheck report` working on a
machine that has never installed torch. See CLAUDE.md → The lazy torch boundary.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from irtcheck.artifact import (
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
)
from irtcheck.io import read_any
from irtcheck.matrix import (
    DEFAULT_RESPONDENT_KEY,
    MatrixError,
    build_matrix,
    parse_respondent_key,
)
from irtcheck.records import RecordError

# Progress is redrawn every this many SVI steps. Often enough to look alive on
# a 4000-item suite, rarely enough that the terminal is not the bottleneck.
PROGRESS_EVERY = 25

# The --respondent-key default, in the spelling the flag takes. Restated here
# only so `run` has a signature that stands on its own when called directly;
# cli.py is what a user actually sees.
DEFAULT_KEY_SPEC = ",".join(DEFAULT_RESPONDENT_KEY)

# Adapter overrides, and the environment variable each one sets.
#
# The readers take `read(path)` and nothing else — that narrow signature is what
# lets io/__init__.py discover adapters by walking its own directory, which in
# turn is what let three adapters be written in parallel without touching a
# shared dispatch table. Widening it to thread options through would trade that
# away for four rarely-used values, so the flags set the environment variables
# the adapters already read instead.
#
# The variables came first: wave 1 shipped them because cli.py was frozen. They
# stay supported, and the flag wins when both are set.
ADAPTER_OVERRIDES = {
    "model_id": "IRTCHECK_LMEVAL_MODEL_ID",
    "metric": "IRTCHECK_LMEVAL_METRIC",
    "lmeval_filter": "IRTCHECK_LMEVAL_FILTER",
    "scorer": "IRTCHECK_INSPECT_SCORER",
}


@contextmanager
def _environment(values: dict[str, str]):
    """Apply `values` to os.environ for the duration of the block, then restore."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def run(
    *,
    responses: Path,
    output: Path,
    respondent_key: str = DEFAULT_KEY_SPEC,
    fmt: str | None = None,
    priors: str = "hierarchical",
    epochs: int = 2000,
    seed: int = 0,
    device: str = "cpu",
    embed_responses: bool = True,
    model_id: str | None = None,
    metric: str | None = None,
    lmeval_filter: str | None = None,
    scorer: str | None = None,
    **_: Any,
) -> None:
    """Entry point called by cli.fit.

    The keywords are exactly the ones the frozen cli.py passes, spelled out
    rather than taken as **kwargs so that calling `run()` with nothing raises
    TypeError — which is how test_contracts.py tells an implemented command
    from a stub that still has to name its owning brief.
    """
    # highlight=False: rich's automatic number highlighting inserts escape
    # codes inside an error message's numbers, which makes the messages harder
    # to read and impossible to grep.
    console = Console(stderr=True, highlight=False)

    overrides = {
        ADAPTER_OVERRIDES[name]: value
        for name, value in (
            ("model_id", model_id),
            ("metric", metric),
            ("lmeval_filter", lmeval_filter),
            ("scorer", scorer),
        )
        if value is not None
    }

    try:
        key = parse_respondent_key(respondent_key)
        # Scoped to the read, not set for the rest of the process. In the CLI
        # the difference is invisible because the process exits, but leaking
        # into os.environ made one test's --lmeval-filter poison every later
        # test in the same session.
        with _environment(overrides):
            matrix = build_matrix(read_any(responses, fmt=fmt), respondent_key=key)
    except (MatrixError, RecordError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Imported here, not at module scope: this line is the only reason this
    # process needs torch at all.
    from irtcheck.fit.fitter import FitError, fit_2pl

    _describe(console, matrix)

    epochs = int(epochs)
    started = time.perf_counter()
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} steps"),
            TextColumn("ELBO {task.fields[elbo]:>12,.1f}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("fitting 2PL", total=epochs, elbo=float("nan"))

            def on_step(step: int, elbo: float) -> None:
                if step % PROGRESS_EVERY == 0 or step == epochs - 1:
                    progress.update(task, completed=step + 1, elbo=elbo)

            fit = fit_2pl(
                matrix,
                priors=priors,
                epochs=epochs,
                seed=int(seed),
                device=device,
                embed_responses=bool(embed_responses),
                on_step=on_step,
            )
    except FitError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    fit.save(output)
    _summarise(console, fit, output, time.perf_counter() - started)


def _describe(console: Console, matrix) -> None:
    """What is about to be fitted, before the slow part starts.

    Respondent count leads, and it is the count of *real models*: it is the
    binding constraint on everything this tool can say, and a user watching
    "12 respondents (5 real models)" scroll past has been told the most
    important thing about their fit before it has finished.
    """
    pseudo = (
        f" ([bold]{matrix.n_real_models}[/bold] real models"
        f", key {'+'.join(matrix.respondent_key)})"
        if matrix.n_respondents != matrix.n_real_models
        else ""
    )
    console.print(
        f"[bold]{matrix.n_respondents}[/bold] respondents{pseudo}, "
        f"[bold]{matrix.n_items}[/bold] items, "
        f"{matrix.n_responses:,} responses ({matrix.density:.0%} of the grid)"
    )
    if matrix.n_real_models < 5:
        console.print(
            f"[yellow]note:[/yellow] {matrix.n_real_models} real model(s). Item "
            "parameters will be mostly prior, and most items will come back "
            "flagged insufficient-data. --respondent-key model_id,prompt_variant "
            "(or any field distinguishing runs) is the cheapest way to add "
            "respondents."
        )


def _summarise(console: Console, fit, output: Path, elapsed: float) -> None:
    counts = fit.diagnostics.get("flag_counts", {})
    size = output.stat().st_size
    console.print(
        f"wrote [bold]{output}[/bold] ({size / 1024:.0f} KiB) in {elapsed:.1f}s — "
        f"ELBO {fit.diagnostics['elbo_final']:,.1f} over "
        f"{fit.diagnostics['epochs']:,} epochs, seed {fit.diagnostics['seed']}"
    )
    console.print(
        f"  {counts.get(FLAG_INSUFFICIENT_DATA, 0)} insufficient-data · "
        f"{counts.get(FLAG_DEAD, 0)} dead · "
        f"{counts.get(FLAG_CEILING, 0)} ceiling · {counts.get(FLAG_FLOOR, 0)} floor"
    )
    undetermined = counts.get(FLAG_INSUFFICIENT_DATA, 0)
    if undetermined > fit.n_items // 2:
        console.print(
            f"[yellow]note:[/yellow] {undetermined} of {fit.n_items} items have a "
            "discrimination interval spanning zero — with this many respondents "
            "the data cannot tell whether they separate anyone. They are excluded "
            "from ranking and selection. More respondents is the fix."
        )
    console.print(f"next: [bold]irtcheck report {output}[/bold]")
