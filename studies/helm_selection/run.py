"""Does choosing items beat sampling them, on real data?

HELM Lite, leave-one-model-out through `irtcheck.validate` itself, so every
number here is what `irtcheck validate` would print for that selector. Three
ways of choosing an anchor set, each against the same random sets:

    tool        select_item_ids(fit, n) — what `select` ships
    per-scen    n allocated across scenarios in proportion to their size, then
                the most informative confident items within each, then that
                scenario's padding items
    item-rest   top n by item-rest correlation over the held-out matrix — no IRT

and, for the record, the protocol `validate` used until this study: each
held-out model scored on its own set, rank correlation across the holdouts.

    PYTHONPATH=src .venv/bin/python studies/helm_selection/run.py \\
        --fit helm12.irt --cache ~/.cache/irtcheck-helm12 > studies/helm_selection/helm12.txt
    PYTHONPATH=src .venv/bin/python studies/helm_selection/run.py \\
        --fit helm95.irt --cache ~/.cache/irtcheck-helm95 --holdouts 24 --workers 3 \\
        > studies/helm_selection/helm95.txt

`helm12.irt` and `helm95.irt` are `irtcheck fit` on the matrices
`scripts/README.md` rebuilds. Holdout fits are cached per model; the first
twelve-model run is ~3 minutes, the 95-model one ~25 on three workers.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from irtcheck.artifact import IrtFit
from irtcheck.matrix import ResponseMatrix
from irtcheck.select import SelectError, padding_items, select_anchor, select_item_ids
from irtcheck.validate import AccuracyTable, full_suite_accuracy, leave_one_model_out

SIZES = (25, 50, 100, 200, 400)
OLD_PROTOCOL_DRAWS = 100


def cache_path(cache: Path, model_id: str) -> Path:
    return cache / (re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + ".irt")


def fit_holdout(job: tuple[str, str, str]) -> str:
    source, cache, model_id = job
    path = cache_path(Path(cache), model_id)
    if path.exists():
        return f"cached  {model_id}"
    os.environ["OMP_NUM_THREADS"] = "10"
    import torch

    torch.set_num_threads(10)
    from irtcheck.fit.fitter import fit_2pl

    matrix = IrtFit.load(source).matrix()
    fit = fit_2pl(matrix.drop_model(model_id), seed=0, epochs=2000)
    tmp = path.with_suffix(".tmp")
    fit.save(tmp)
    tmp.rename(path)
    return f"fitted  {model_id}"


def scenario(item_id: str) -> str:
    return re.sub(r"_id\d+$", "", item_id)


def per_scenario(fit: IrtFit, n: int) -> list[str]:
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, item in enumerate(fit.item_ids):
        groups[scenario(item)].append(i)
    share = {s: n * len(idx) / fit.n_items for s, idx in groups.items()}
    take = {s: int(np.floor(v)) for s, v in share.items()}
    for s in sorted(groups, key=lambda s: (-(share[s] - take[s]), s))[: n - sum(take.values())]:
        take[s] += 1
    confident = set(fit.usable_items())
    padding = set(padding_items(fit, exclude=list(confident)))
    chosen: list[int] = []
    for s in sorted(groups):
        picked: list[int] = []
        for tier in (confident, padding):
            candidates = [i for i in groups[s] if i in tier and fit.n_resp[i] > 0]
            if len(picked) < take[s] and candidates:
                try:
                    anchor = select_anchor(fit, take[s] - len(picked), candidates=candidates)
                    picked += anchor.item_indices
                except SelectError:
                    pass
        chosen += picked
    return [fit.item_ids[i] for i in chosen]


def item_rest(matrix: ResponseMatrix) -> list[str]:
    """Items by item-rest correlation over real models, best first; constant items last out."""
    table = AccuracyTable(matrix)
    models = len(table.model_ids)
    per_model = np.full((models, matrix.n_items), np.nan)
    for c in range(matrix.n_items):
        per_model[:, c] = table.accuracy(np.array([c]))
    corr = np.full(matrix.n_items, -np.inf)
    total = np.nansum(per_model, axis=1)
    count = np.sum(~np.isnan(per_model), axis=1)
    for c in range(matrix.n_items):
        x = per_model[:, c]
        ok = ~np.isnan(x)
        if ok.sum() < 3 or x[ok].std() == 0:
            continue
        rest = (total[ok] - x[ok]) / (count[ok] - 1)
        corr[c] = np.corrcoef(x[ok], rest)[0, 1]
    order = np.argsort(-corr, kind="stable")
    return [matrix.item_ids[c] for c in order if np.isfinite(corr[c])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--holdouts", type=int, default=0, help="0 holds out every model")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    matrix = IrtFit.load(args.fit).matrix()
    truth = full_suite_accuracy(matrix)
    models = sorted(truth)
    if args.holdouts:
        # Spread across the accuracy ranking, best to worst.
        by_accuracy = sorted(models, key=lambda m: -truth[m])
        at = np.linspace(0, len(models) - 1, args.holdouts).round().astype(int)
        held = [by_accuracy[i] for i in at]
    else:
        held = models
    jobs = [(str(args.fit), str(args.cache), m) for m in held]
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(fit_holdout, jobs):
            print(line, file=sys.stderr, flush=True)

    def cached_fit(held_out: ResponseMatrix) -> IrtFit:
        (missing,) = set(matrix.derives_from) - set(held_out.derives_from)
        return IrtFit.load(cache_path(args.cache, missing))

    rest_orders: dict[str, list[str]] = {}

    def by_item_rest(fit: IrtFit, n: int) -> list[str]:
        key = ",".join(sorted(set(fit.derives_from)))
        if key not in rest_orders:
            rest_orders[key] = item_rest(fit.matrix())
        return rest_orders[key][:n]

    table = AccuracyTable(matrix)
    selectors = {"tool": select_item_ids, "per-scen": per_scenario, "item-rest": by_item_rest}
    reports = {}
    old = {}
    for name, select_fn in selectors.items():
        chosen: dict[tuple[str, int], list[str]] = {}

        def recording(fit: IrtFit, n: int, select_fn=select_fn, chosen=chosen) -> list[str]:
            (missing,) = set(matrix.derives_from) - set(fit.derives_from)
            chosen[(missing, n)] = list(select_fn(fit, n))
            return chosen[(missing, n)]

        reports[name] = leave_one_model_out(
            matrix, fit_fn=cached_fit, sizes=SIZES, select_fn=recording, holdouts=held
        )
        old[name] = {
            n: spearmanr(
                [truth[m] for m in held],
                [
                    table.accuracy(table.columns(chosen[(m, n)]))[table.model_ids.index(m)]
                    for m in held
                ],
            ).statistic
            for n in SIZES
        }
    rng = np.random.default_rng(0)
    old["random"] = {}
    for n in SIZES:
        rhos = []
        for _ in range(OLD_PROTOCOL_DRAWS):
            own = [
                table.accuracy(rng.choice(matrix.n_items, n, replace=False))[
                    table.model_ids.index(m)
                ]
                for m in held
            ]
            rhos.append(spearmanr([truth[m] for m in held], own).statistic)
        old["random"][n] = float(np.mean(rhos))

    first = reports["tool"]
    print(
        f"HELM Lite: {first.n_models} models x {first.n_items} items, "
        f"{len(held)} held out, {first.random_draws} random sets per (holdout, size)."
    )
    print()
    print("Places off: mean |place on the anchor set - place on the full suite| of the held-out")
    print(
        f"model, every model scored on that set. Lower is better; 0 is exact, out of {first.n_models}."
    )
    print("'beats' is the share of random draws the selector placed held-out models better than.")
    print()
    print(f"{'n':>4} | {'random':>7} | " + " | ".join(f"{name:>16}" for name in selectors))
    for s, n in enumerate(SIZES):
        random_error = reports["tool"].results[s].random_error
        cells = []
        for name in selectors:
            r = reports[name].results[s]
            cells.append(f"{r.mean_error:6.2f} beats {r.beats_random:4.0%}")
        print(f"{n:>4} | {random_error:7.2f} | " + " | ".join(cells))
    print()
    print("Spearman, held-out places against full-suite places (random: mean over draws)")
    print(f"{'n':>4} | {'random':>7} | " + " | ".join(f"{name:>9}" for name in selectors))
    for s, n in enumerate(SIZES):
        cells = [f"{reports[name].results[s].spearman:+9.3f}" for name in selectors]
        print(f"{n:>4} | {reports['tool'].results[s].random_spearman:+7.3f} | " + " | ".join(cells))
    print()
    print("The protocol validate used before this study: each held-out model on its own set,")
    print(
        f"Spearman across the {len(held)} holdouts. Kept for the record; it mixes set difficulty in."
    )
    print(f"{'n':>4} | {'random':>7} | " + " | ".join(f"{name:>9}" for name in selectors))
    for n in SIZES:
        cells = [f"{old[name][n]:+9.3f}" for name in selectors]
        print(f"{n:>4} | {old['random'][n]:+7.3f} | " + " | ".join(cells))
    print()
    print(
        "tool anchor-set sizes, smallest per n:", [r.min_selected for r in reports["tool"].results]
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
