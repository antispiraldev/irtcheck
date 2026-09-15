"""What to fill a short anchor set with: the measurement behind `select`'s padding.

The sweep found that a short set ranks worse than a random one (docs/validation.md
§4). Its "padded" comparison was the ungated selection, which puts mis-keyed
items in, so it could not say what padding *should* take. This scores four
answers on every sweep fit where the confident pool cannot fill a set:

    short      the confident items alone, as 0.1.0 was tagged
    ungated    selection from every non-ceiling/floor item
    no-inv     the confident items, filled from items that are not inverted
    shipped    `select_anchor` as it is now: also never an item whose fitted
               slope is negative

    PYTHONPATH=src .venv/bin/python studies/min_respondents/padding.py \\
        --fits ~/.cache/irtcheck-study > studies/min_respondents/padding.txt

~3 minutes on 14 workers. Differences are paired within a fit. The `random`
and `classic` columns are read from results.json for the same fits, so they are
the baselines §4 already quotes rather than a second draw.
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from analyse import (
    ANCHOR_SIZES,
    INVERTED_FRACTION,
    NAME,
    fresh_population,
    ranks_fresh,
    safe_select,
)

from irtcheck.artifact import FLAG_CEILING, FLAG_FLOOR, FLAG_INVERTED, IrtFit
from irtcheck.synth import make_truth

VARIANTS = ("short", "ungated", "no-inv", "shipped")
BASELINES = ("random", "classic")
KEY = ("models", "items", "variants", "seed")


def measure(path: Path) -> list[dict]:
    cell = {k: int(v) for k, v in NAME.search(path.name).groupdict().items()}
    fit = IrtFit.load(path)
    truth = make_truth(
        n_models=cell["models"],
        n_items=cell["items"],
        variants_per_model=cell["variants"],
        inverted_fraction=INVERTED_FRACTION,
        seed=cell["seed"],
    )
    flags = [set(f) for f in fit.flags]
    edge = [bool({FLAG_CEILING, FLAG_FLOOR} & f) for f in flags]
    theta, responses = fresh_population(truth, cell["seed"], "same")
    rows = []
    for n in ANCHOR_SIZES:
        short = safe_select(fit, n, candidates=fit.usable_items())
        if len(short) >= n:
            continue
        taken = set(short)
        no_inv_pool = [
            i
            for i in range(fit.n_items)
            if i not in taken and not edge[i] and FLAG_INVERTED not in flags[i]
        ]
        sets = {
            "short": short,
            "ungated": safe_select(
                fit, n, candidates=[i for i in range(fit.n_items) if not edge[i]]
            ),
            "no-inv": short + safe_select(fit, n - len(short), candidates=no_inv_pool),
            "shipped": safe_select(fit, n),
        }
        row = dict(cell, n=n)
        for name, items in sets.items():
            row[name] = ranks_fresh(items, theta, responses) if items else float("nan")
            row[f"{name}_size"] = len(items)
            row[f"{name}_backwards"] = int((truth.a[items] < 0).sum()) if items else 0
        rows.append(row)
    return rows


def paired(rows: list[dict], a: str, b: str) -> str:
    d = np.array([r[a] - r[b] for r in rows if np.isfinite(r[a]) and np.isfinite(r[b])])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return f"{a} - {b}: {d.mean() * 1000:+.1f} ± {se * 1000:.1f} thousandths (n={len(d)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(measure, sorted(args.fits.glob("*.irt"))) for r in rs]
    sweep = {tuple(f[k] for k in KEY): f for f in json.loads(args.results.read_text())["fits"]}
    for r in rows:
        for b in BASELINES:
            r[b] = sweep[tuple(r[k] for k in KEY)][f"n{r['n']}_rho_{b}"]

    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["n"], r["items"], r["variants"], r["models"])].append(r)
    print("Spearman over 2,000 fresh models (same population), mean over fits where the")
    print("confident pool is short. `short` is averaged over fits that returned anything.")
    print()
    columns = VARIANTS + BASELINES
    print("   n items var models fits | short size | " + " ".join(f"{v:>8}" for v in columns))
    for key in sorted(groups):
        g = groups[key]
        shorts = [r["short"] for r in g if np.isfinite(r["short"])]
        cells = [f"{np.mean(shorts):8.3f}" if shorts else "       -"]
        cells += [f"{np.mean([r[v] for r in g]):8.3f}" for v in columns[1:]]
        size = np.mean([r["short_size"] for r in g])
        print(
            f"{key[0]:>4} {key[1]:>5} {key[2]:>3} {key[3]:>6} {len(g):>4} | {size:10.1f} | "
            + " ".join(cells)
        )
    print()
    print(f"{len(rows)} short (fit, n) cases")
    for v in VARIANTS:
        print(f"mis-keyed items across them, {v:>8}: {sum(r[f'{v}_backwards'] for r in rows)}")
    print(paired(rows, "shipped", "no-inv"))
    print(paired(rows, "shipped", "ungated"))
    print(paired(rows, "shipped", "short"))
    print(paired(rows, "shipped", "random"))
    print(paired(rows, "shipped", "classic"))
    worse = sum(r["shipped"] < r["random"] for r in rows)
    print(f"fits where the shipped set ranks below the mean random draw: {worse} of {len(rows)}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
