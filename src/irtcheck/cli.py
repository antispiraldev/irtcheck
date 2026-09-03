"""The command surface.

**This file is frozen after wave 0.** Every command and every flag is declared
here up front, and each one delegates to a module in irtcheck.commands/ owned
by exactly one brief. Four agents work wave 1 at once; if each of them had to
add their own command to this file, this file would be the one merge conflict
the whole arrangement was designed to avoid. Adding a flag later is a
deliberate, single-owner change, not something a wave-1 agent does in passing.

Commands import their implementation inside the function body, not at module
scope. `irtcheck report` on a machine without torch has to work — see
CLAUDE.md → The lazy torch boundary — and a top-level import here would drag
the fitter in on every invocation, including `--help`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from irtcheck import __version__

app = typer.Typer(
    name="irtcheck",
    help="Fit an IRT model to an eval response matrix and report which items carry signal.",
    no_args_is_help=True,
    add_completion=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"irtcheck {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    pass


@app.command()
def fit(
    responses: Annotated[
        Path, typer.Argument(help="Response file: JSONL, CSV, or a supported harness log.")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Where to write the .irt fit artifact.")
    ],
    respondent_key: Annotated[
        str,
        typer.Option(
            "--respondent-key",
            help=(
                "Comma-separated fields composing respondent identity, e.g. "
                "'model_id,prompt_variant'. Pseudo-respondents are the cheapest way "
                "to raise respondent count, which is the binding constraint."
            ),
        ),
    ] = "model_id",
    fmt: Annotated[
        str | None,
        typer.Option("--format", help="Force an input adapter instead of sniffing."),
    ] = None,
    priors: Annotated[
        str, typer.Option("--priors", help="'hierarchical' (partial pooling) or 'vague'.")
    ] = "hierarchical",
    epochs: Annotated[int, typer.Option("--epochs", help="SVI steps.")] = 2000,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed; fits are reproducible.")] = 0,
    device: Annotated[str, typer.Option("--device", help="'cpu' or 'cuda'.")] = "cpu",
    embed_responses: Annotated[
        bool,
        typer.Option(
            "--embed-responses/--no-embed-responses",
            help="Carry the response matrix in the artifact. Required by `validate`.",
        ),
    ] = True,
) -> None:
    """Fit a 2PL to a response matrix and cache the result."""
    from irtcheck.commands import fit as impl

    impl.run(
        responses=responses,
        output=output,
        respondent_key=respondent_key,
        fmt=fmt,
        priors=priors,
        epochs=epochs,
        seed=seed,
        device=device,
        embed_responses=embed_responses,
    )


@app.command()
def report(
    artifact: Annotated[Path, typer.Argument(help="A .irt file written by `irtcheck fit`.")],
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Also write a self-contained HTML report here."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON instead of a table.")
    ] = False,
    sort: Annotated[
        str, typer.Option("--sort", help="Order items by: discrimination, difficulty, or id.")
    ] = "discrimination",
    limit: Annotated[
        int, typer.Option("--limit", help="Show at most this many items; 0 for all.")
    ] = 40,
) -> None:
    """Show per-item parameters, flags, and suite-level diagnostics."""
    from irtcheck.commands import report as impl

    impl.run(artifact=artifact, html=html, as_json=as_json, sort=sort, limit=limit)


@app.command()
def select(
    artifact: Annotated[Path, typer.Argument(help="A .irt file written by `irtcheck fit`.")],
    count: Annotated[int, typer.Option("-n", "--count", help="Anchor set size.")],
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Write the item ids here as JSON.")
    ] = None,
    adaptive: Annotated[
        bool,
        typer.Option(
            "--adaptive",
            help=(
                "Select per respondent instead of one fixed set. Secondary: a "
                "regression suite needs the same items every run to stay comparable."
            ),
        ),
    ] = False,
) -> None:
    """Emit a fixed anchor set of N maximally informative items."""
    from irtcheck.commands import select as impl

    impl.run(artifact=artifact, count=count, output=output, adaptive=adaptive)


@app.command()
def validate(
    artifact: Annotated[Path, typer.Argument(help="A .irt file written by `irtcheck fit`.")],
    sizes: Annotated[
        str, typer.Option("--sizes", help="Comma-separated anchor sizes to sweep.")
    ] = "25,50,100,200,400",
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed for the refits.")] = 0,
    epochs: Annotated[
        int, typer.Option("--epochs", help="SVI steps per holdout refit.")
    ] = 2000,
) -> None:
    """Leave-one-model-out: does a small anchor set reproduce the full-suite ranking?"""
    from irtcheck.commands import validate as impl

    impl.run(artifact=artifact, sizes=sizes, as_json=as_json, seed=seed, epochs=epochs)


if __name__ == "__main__":
    app()
