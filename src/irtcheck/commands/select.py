"""`irtcheck select` — owned by wave1/select-validate.

Emits a *fixed* anchor set. A regression suite needs the same items every run
or its scores are not comparable over time, so the same artifact and the same n
must always give the same ids: selection is deterministic, ties break toward
the lower item index, and there is no RNG anywhere in the path.

Stream discipline: the human summary goes to stderr and the item ids to stdout,
so `irtcheck select suite.irt -n 100 | xargs ...` works and still explains
itself on the terminal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from irtcheck.artifact import ArtifactError, IrtFit
from irtcheck.select import AnchorSet, SelectError, select_anchor


def run(
    *,
    artifact: Path,
    count: int,
    output: Path | None = None,
    adaptive: bool = False,
    **_: Any,
) -> None:
    err = Console(stderr=True)

    if adaptive:
        # Deliberately not built. Per-respondent selection demos well and is the
        # wrong default for a regression suite, so it is a wave-3 extension
        # (docs/build-plan.html, brief H) rather than a half-done flag that
        # silently returns something else than what the user asked for.
        err.print(
            "[red]--adaptive is not implemented.[/red] Adaptive selection picks a "
            "different item set per respondent, so scores stop being comparable "
            "across runs — it is a secondary feature (wave 3, brief H), and the "
            "fixed anchor set is what this command exists to produce.\n"
            "Drop --adaptive to get the fixed set."
        )
        raise typer.Exit(2)

    try:
        fit = IrtFit.load(artifact)
    except ArtifactError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    try:
        anchor = select_anchor(fit, count)
    except SelectError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    _summarise(err, fit, anchor)

    if output is not None:
        payload = anchor.to_dict() | {"source": str(artifact)}
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        err.print(f"Wrote {anchor.n_selected} item ids to [bold]{output}[/bold].")
    else:
        for item_id in anchor.item_ids:
            print(item_id)


def _summarise(err: Console, fit: IrtFit, anchor: AnchorSet) -> None:
    # Respondent count is the count of real models, never of pseudo-respondents:
    # reporting 60 when they are twelve temperature samples of five models
    # misrepresents the one thing this tool exists to be honest about.
    models = fit.n_real_models
    header = f"{models} model{'s' if models != 1 else ''}"
    if fit.has_pseudo_respondents:
        header += f" ({fit.n_respondents} respondents, key {'+'.join(fit.respondent_key)})"
    err.print(
        f"[bold]{anchor.n_selected}[/bold] of {anchor.n_usable} usable items "
        f"({fit.n_items} total) · {header}"
    )
    if anchor.shortfall:
        err.print(
            f"[yellow]Asked for {anchor.requested}, selected {anchor.n_selected}.[/yellow] "
            f"Only {anchor.n_usable} of {fit.n_items} items are eligible; the rest are "
            "flagged insufficient-data, inverted, ceiling or floor. Padding the set with items we "
            "said we could not read would be worse than a short one."
        )
    err.print(
        f"Mean ability standard error over these models: {anchor.mean_theta_se:.3f} "
        f"(1.000 with no items at all)."
    )
    if anchor.gains:
        err.print(
            f"Marginal gain: first item {anchor.gains[0]:.4f}, "
            f"last item {anchor.gains[-1]:.4f}. Where that flattens is where n stops buying "
            "precision."
        )
