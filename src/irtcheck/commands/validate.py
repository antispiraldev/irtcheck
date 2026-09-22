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

# Below this many real models, choosing items lost to sampling them even on
# synthetic data, *when every model is re-scored on the chosen set*. Given the
# true parameters the same selection won at every count, and placing the
# held-out model by ability — which leaves the other models' places alone —
# wins at every count too. Measured, not tuned: 10 and 25 models lost, 50 and
# 100 won clearly (docs/validation.md §5, studies/true_ability/). The sweep has
# no point between 25 and 50.
SELECTION_NEEDS_MODELS = 50


def why_random_wins(n_models: int) -> str:
    """The reason printed when random sets place held-out models as well or better."""
    if n_models < SELECTION_NEEDS_MODELS:
        return (
            f"With {n_models} models two things are in play. The fit's item estimates are noisy — "
            f"on synthetic data, scored this way, choosing beat sampling only from about "
            f"{SELECTION_NEEDS_MODELS} models. And this table re-scores every model on a set "
            "chosen from their own answers, which costs a chosen set more than a random one; the "
            "ability table below does not, and on synthetic data that is the larger part. If you "
            "compare models by plain accuracy on the set, drop the items `report` flags and draw "
            "the rest at random."
        )
    return (
        f"At {n_models} models synthetic data has choosing ahead, so this is likely "
        "something about this suite — it happened on HELM Lite at 95 models, and the cause "
        "is not measured. A random draw of the unflagged items is the safer set here."
    )


def ability_note(losing: list[tuple[int, float]]) -> str:
    """What the ability table says about the sizes where accuracy scoring lost."""
    won = [f"n={size} ({share:.0%})" for size, share in losing if share == share and share >= 0.5]
    if not won:
        return ""
    return (
        " Placed by ability instead, the same sets beat most random draws at "
        + ", ".join(won)
        + ": scoring every model on a set chosen from their own answers is part of what the "
        "first table measures."
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
            seed=seed,
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


def _places(value: float) -> str:
    return "—" if value != value else f"{value:.2f}"


def _render(out: Console, report: ValidationReport) -> None:
    # Real models, not respondents: the count of things being ranked is what
    # makes or breaks a placement, and it is not the respondent count.
    held = (
        "" if len(report.model_ids) == report.n_models else f" · {len(report.model_ids)} held out"
    )
    out.print(
        f"[bold]Leave-one-model-out[/bold] · {report.n_models} models{held} · "
        f"{report.n_items} items · {report.n_respondents} respondents "
        f"(key {'+'.join(report.respondent_key)})"
    )

    table = Table(title=None, header_style="bold")
    table.add_column("n", justify="right")
    table.add_column("items", justify="right")
    table.add_column("places off", justify="right")
    table.add_column("random", justify="right")
    table.add_column("beats random", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("random", justify="right")
    for result in report.results:
        beats = result.beats_random
        share = "—" if beats != beats else f"{beats:.0%}"
        if beats == beats and beats < 0.5:
            share = f"[yellow]{share}[/yellow]"
        table.add_row(
            str(result.size),
            f"{result.min_selected}{'*' if result.short else ''}",
            _places(result.mean_error),
            f"[dim]{_places(result.random_error)}[/dim]",
            share,
            _fmt(result.spearman),
            f"[dim]{_fmt(result.random_spearman)}[/dim]",
        )
    out.print(table)

    out.print(
        "For each model: choose the anchor set without it, score [bold]every[/bold] model on "
        "that set by plain accuracy, and see where the held-out model lands. "
        f"[bold]places off[/bold] is the mean distance from its full-suite place (0 is exact, "
        f"out of {report.n_models}); [bold]random[/bold] is the same for "
        f"{report.random_draws} random sets of the same size, and [bold]beats random[/bold] "
        "is the share of those draws the anchor set placed models better than. Spearman "
        "compares the held-out models' places with their full-suite places."
    )
    losing = [
        r.size for r in report.results if r.beats_random == r.beats_random and r.beats_random < 0.5
    ]
    if losing:
        sizes = ", ".join(f"n={n}" for n in losing)
        out.print(
            f"[yellow]At {sizes}, random item sets placed held-out models as well or better.[/yellow] "
            f"{why_random_wins(report.n_models)} The per-item flags from `report` do not "
            "depend on this."
            + ability_note(
                [(r.size, r.ability_beats_random) for r in report.results if r.size in losing]
            )
        )
    ability = Table(title=None, header_style="bold")
    ability.add_column("n", justify="right")
    ability.add_column("places off", justify="right")
    ability.add_column("random", justify="right")
    ability.add_column("beats random", justify="right")
    for result in report.results:
        beats = result.ability_beats_random
        share = "—" if beats != beats else f"{beats:.0%}"
        if beats == beats and beats < 0.5:
            share = f"[yellow]{share}[/yellow]"
        ability.add_row(
            str(result.size),
            _places(result.mean_ability_error),
            f"[dim]{_places(result.random_ability_error)}[/dim]",
            share,
        )
    out.print("\n[bold]Placed by ability[/bold] — the same sets, scored the way you would use one")
    out.print(ability)
    out.print(
        "Estimate the held-out model's ability from [bold]its own[/bold] answers to the set, using "
        "the item parameters of the fit that chose it, and place it among the other models' "
        "abilities in that same fit. Nothing about those models is recomputed on the set, so the "
        "choice of items cannot flatter or distort what the held-out model is compared against."
    )

    if any(r.short for r in report.results):
        out.print(
            "[yellow]*[/yellow] fewer items were available than requested, even after padding "
            "with insufficient-data items — the rest are inverted, at ceiling or floor, "
            "unanswered, or lean backwards, and are never selected."
        )
    if report.n_models < 6:
        out.print(
            f"[yellow]{report.n_models} models is thin for placement[/yellow]: a held-out model "
            "can only be a few places off, and one swap moves the mean a long way. Read the "
            "trend across n rather than any one number."
        )
