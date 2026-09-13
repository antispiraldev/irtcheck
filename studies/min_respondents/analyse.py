"""Measure what the tool delivers at each respondent count, against ground truth.

Stage two of the minimum-respondent study. Reads the artifacts `run.py` wrote,
rebuilds the universe each came from, and asks the question a refusal threshold
has to answer: **at this many models, is what the tool hands you better than
what you had without it?**

"What it hands you" is the anchor set, so that is what gets scored — on fresh
models the fit never saw, drawn from the same universe, ranked by plain
accuracy over the set exactly as the README tells people to use one. Four sets
of each size are compared:

    tool     select_anchor(fit, n)                    what irtcheck does today
    ungated  the same selection, but only ceiling/floor excluded
    random   n items drawn uniformly                  what you had without the tool
    filtered n items drawn from the non-ceiling/floor pool
    classic  top-n by item-rest correlation           classical test theory, no IRT
    oracle   top-n by true information over N(0, 1)   the ceiling

`ungated` exists because the gate is the thing under test. If it beats `tool`
at small N, the tool is refusing something it could have done.

`filtered` and `classic` exist because `ungated` beating `random` proves less
than it looks. Dropping items everyone gets right or wrong needs no model at
all, and item-rest correlation is what anyone with a spreadsheet would compute.
An IRT fit has to beat *those* to be earning its keep at a given respondent
count; beating a uniform draw that includes ceiling items is not the bar.

    .venv/bin/python studies/min_respondents/analyse.py --fits ~/.cache/irtcheck-study \\
        --json studies/min_respondents/results.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

from irtcheck.artifact import (
    DEAD_THRESHOLD,
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
    FLAG_INVERTED,
    IrtFit,
)
from irtcheck.select import SelectError, item_information, select_anchor
from irtcheck.synth import make_truth

NAME = re.compile(r"m(?P<models>\d+)_i(?P<items>\d+)_v(?P<variants>\d)_s(?P<seed>\d+)\.irt$")
ANCHOR_SIZES = (25, 50)
FRESH_MODELS = 2000
RANDOM_DRAWS = 40
# Population quadrature for the oracle: it chooses for the universe's ability
# distribution, not for the few models a fit happened to see.
QUADRATURE = norm.ppf((np.arange(41) + 0.5) / 41)
INVERTED_FRACTION = 0.02  # must match run.py


# Who the anchor set is later used on. `same` is the population the fit's models
# came from. `stronger` is the next generation — every one a standard deviation
# above today's average — which is the situation a regression suite exists for,
# and the one where choosing items for *where models will be* rather than for
# how they happen to correlate today could plausibly matter.
POPULATIONS = {"same": (0.0, 1.0), "stronger": (1.0, 0.7)}


def fresh_population(truth, seed: int, population: str = "same") -> tuple[np.ndarray, np.ndarray]:
    """Unseen models, and their responses to every item."""
    rng = np.random.default_rng(10_000 + seed)
    centre, spread = POPULATIONS[population]
    theta = rng.normal(centre, spread, size=FRESH_MODELS)
    p = 1.0 / (1.0 + np.exp(-truth.a[None, :] * (theta[:, None] - truth.b[None, :])))
    return theta, (rng.random(p.shape) < p).astype(np.int8)


def ranks_fresh(items: list[int], theta: np.ndarray, responses: np.ndarray) -> float:
    """Spearman between true ability and accuracy over `items`; nan for no items."""
    if not items:
        return float("nan")
    accuracy = responses[:, items].mean(axis=1)
    if np.all(accuracy == accuracy[0]):
        return 0.0
    return float(spearmanr(accuracy, theta).statistic)


def classical_order(fit: IrtFit, pool: list[int]) -> list[int]:
    """`pool` sorted by item-rest correlation, computed per real model.

    Pseudo-respondents are averaged into their model first, so three prompt
    variants do not count as three independent observations of an item's
    discrimination — the same rule `derives_from` enforces everywhere else.
    """
    models = sorted(set(fit.derives_from))
    index = {m: k for k, m in enumerate(models)}
    rows = np.asarray(fit.responses.rows)
    cols = np.asarray(fit.responses.cols)
    obs = np.asarray(fit.responses.obs, dtype=float)
    owner = np.array([index[fit.derives_from[r]] for r in rows])
    total = np.zeros((len(models), fit.n_items))
    count = np.zeros((len(models), fit.n_items))
    np.add.at(total, (owner, cols), obs)
    np.add.at(count, (owner, cols), 1.0)
    score = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    filled = np.where(np.isnan(score), np.nanmean(score, axis=0, keepdims=True), score)
    overall = filled.sum(axis=1)
    corr = np.zeros(fit.n_items)
    for i in pool:
        rest = overall - filled[:, i]
        x = filled[:, i]
        if x.std() > 0 and rest.std() > 0:
            corr[i] = float(np.corrcoef(x, rest)[0, 1])
    return sorted(pool, key=lambda i: (-corr[i], i))


def safe_select(fit: IrtFit, n: int, candidates=None) -> list[int]:
    try:
        return select_anchor(fit, n, candidates=candidates).item_indices
    except SelectError:
        return []


def measure(path: Path) -> dict:
    cell = {k: int(v) for k, v in NAME.search(path.name).groupdict().items()}
    fit = IrtFit.load(path)
    truth = make_truth(
        n_models=cell["models"],
        n_items=cell["items"],
        variants_per_model=cell["variants"],
        inverted_fraction=INVERTED_FRACTION,
        seed=cell["seed"],
    )
    true_a = truth.a
    flags = [set(f) for f in fit.flags]
    has = lambda flag: np.array([flag in f for f in flags])  # noqa: E731
    insufficient, dead, inverted = has(FLAG_INSUFFICIENT_DATA), has(FLAG_DEAD), has(FLAG_INVERTED)
    edge = has(FLAG_CEILING) | has(FLAG_FLOOR)
    usable = np.zeros(fit.n_items, dtype=bool)
    usable[fit.usable_items()] = True

    lo, hi = np.asarray(fit.a.hdi_low), np.asarray(fit.a.hdi_high)
    truly_live = true_a >= 0.8

    row: dict = {
        **cell,
        "share_insufficient": float(insufficient.mean()),
        "n_usable": int(usable.sum()),
        "n_dead": int(dead.sum()),
        "n_inverted": int(inverted.sum()),
        "a_coverage": float(((lo <= true_a) & (true_a <= hi)).mean()),
        # Claims the tool makes, scored. Each is a share of the items that
        # carry the claim, so an empty claim is None rather than a perfect 0.
        "usable_truly_flat": _share(true_a[usable] < DEAD_THRESHOLD),
        "usable_truly_backwards": _share(true_a[usable] < 0.0),
        "dead_truly_negligible": _share(np.abs(true_a[dead]) < DEAD_THRESHOLD),
        "inverted_truly_backwards": _share(true_a[inverted] < 0.0),
        # What the refusal costs: live items the tool would not let you use.
        "live_refused": _share(insufficient[truly_live & ~edge]),
    }
    # Where the planted mis-keyed items end up. Exactly one bucket each, in the
    # order the report would describe them.
    planted = true_a < 0
    buckets = {
        "inverted": inverted,
        "dead": dead,
        "edge": edge & ~inverted & ~dead,
        "insufficient": insufficient & ~edge,
        "usable": usable & ~dead,
    }
    for name, mask in buckets.items():
        row[f"planted_inverted_as_{name}"] = int((planted & mask).sum())
    row["planted_inverted"] = int(planted.sum())
    row["planted_inverted_fitted_negative"] = int((planted & (np.asarray(fit.a.mean) < 0)).sum())

    true_info = item_information(np.abs(true_a), truth.b, QUADRATURE).mean(axis=1)
    true_info[true_a < 0] = 0.0  # an oracle does not pick a mis-keyed item
    ungated_pool = [i for i in range(fit.n_items) if not edge[i]]
    classical = classical_order(fit, ungated_pool)
    sets: dict[int, dict[str, list[int]]] = {}
    for n in ANCHOR_SIZES:
        tool = safe_select(fit, n)
        sets[n] = {
            "tool": tool,
            "ungated": safe_select(fit, n, candidates=ungated_pool),
            "classic": classical[:n],
            "oracle": list(np.argsort(-true_info)[:n]),
        }
        row[f"n{n}_tool_size"] = len(tool)
        row[f"n{n}_tool_backwards"] = int((true_a[tool] < 0).sum()) if tool else 0
        row[f"n{n}_ungated_backwards"] = int((true_a[sets[n]["ungated"]] < 0).sum())

    for population in POPULATIONS:
        theta, responses = fresh_population(truth, cell["seed"], population)
        tag = "" if population == "same" else f"{population}_"
        row[f"{tag}rho_full_suite"] = ranks_fresh(list(range(fit.n_items)), theta, responses)
        rng = np.random.default_rng(20_000 + cell["seed"])
        for n in ANCHOR_SIZES:
            for name, items in sets[n].items():
                row[f"{tag}n{n}_rho_{name}"] = ranks_fresh(items, theta, responses)
            random_rho = [
                ranks_fresh(list(rng.choice(fit.n_items, size=n, replace=False)), theta, responses)
                for _ in range(RANDOM_DRAWS)
            ]
            row[f"{tag}n{n}_rho_random"] = float(np.mean(random_rho))
            row[f"{tag}n{n}_rho_random_p10"] = float(np.quantile(random_rho, 0.1))
            row[f"{tag}n{n}_rho_filtered"] = float(
                np.mean(
                    [
                        ranks_fresh(
                            list(
                                rng.choice(
                                    ungated_pool, size=min(n, len(ungated_pool)), replace=False
                                )
                            ),
                            theta,
                            responses,
                        )
                        for _ in range(RANDOM_DRAWS)
                    ]
                )
            )
    return row


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _share(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if mask.size else None


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["items"], row["variants"], row["models"])].append(row)
    out = []
    for (items, variants, models), members in sorted(groups.items()):
        summary: dict = {
            "items": items,
            "variants": variants,
            "models": models,
            "seeds": len(members),
        }
        for key in members[0]:
            if key in {"items", "variants", "models", "seed"}:
                continue
            values = [m[key] for m in members if m[key] is not None and not np.isnan(m[key])]
            summary[key] = float(np.mean(values)) if values else None
            if key.endswith("_rho_tool"):
                summary[key.replace("_rho_tool", "_tool_empty")] = int(
                    sum(m[key] is None or bool(np.isnan(m[key])) for m in members)
                )
        out.append(summary)
    return out


def print_table(summary: list[dict], n: int, tag: str = "") -> None:
    head = (
        "items var models | ins%  usable | flat% back% | "
        f"{tag}n{n}: size  tool ungated  random filtered classic oracle | live-refused | cover"
    )
    print(head)
    print("-" * len(head))
    fmt = lambda v, w=5, p=3: "—".rjust(w) if v is None else f"{v:{w}.{p}f}"  # noqa: E731
    for s in summary:
        print(
            f"{s['items']:>5} {s['variants']:>3} {s['models']:>6} | "
            f"{100 * s['share_insufficient']:4.0f} {s['n_usable']:7.1f} | "
            f"{fmt(s['usable_truly_flat'] and 100 * s['usable_truly_flat'], 5, 1)} "
            f"{fmt(s['usable_truly_backwards'] and 100 * s['usable_truly_backwards'], 5, 1)} | "
            f"{s[f'n{n}_tool_size']:9.1f} {fmt(s[f'{tag}n{n}_rho_tool'])} {fmt(s[f'{tag}n{n}_rho_ungated'], 7)} "
            f"{fmt(s[f'{tag}n{n}_rho_random'], 7)} {fmt(s[f'{tag}n{n}_rho_filtered'], 8)} {fmt(s[f'{tag}n{n}_rho_classic'], 7)} "
            f"{fmt(s[f'{tag}n{n}_rho_oracle'], 6)} | {fmt(s['live_refused'], 12)} | {fmt(s['a_coverage'])}"
        )


def print_paired(rows: list[dict], n: int, tag: str = "") -> None:
    """Per-seed paired differences, x1000, mean ± standard error.

    Paired because every method is scored on the same universe and the same
    fresh models, so the between-universe variance — which is most of it —
    cancels. `tool - classic` counts only fits where the tool filled the set:
    a short set is a different failure, reported by the size column above.
    """
    comparisons = (
        ("tool - classic", "tool", "classic", True),
        ("ungated - classic", "ungated", "classic", False),
        ("tool - ungated", "tool", "ungated", True),
        ("classic - filtered", "classic", "filtered", False),
    )
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["items"], row["variants"], row["models"])
        for label, x, y, needs_full in comparisons:
            if needs_full and row[f"n{n}_tool_size"] < n:
                continue
            a, b = row[f"{tag}n{n}_rho_{x}"], row[f"{tag}n{n}_rho_{y}"]
            if a is None or b is None or np.isnan(a) or np.isnan(b):
                continue
            groups[key][label].append(1000.0 * (a - b))
    print("items var models | " + " | ".join(f"{label:>19}" for label, *_ in comparisons))
    for key in sorted(groups):
        cells = []
        for label, *_ in comparisons:
            values = np.asarray(groups[key][label])
            if values.size == 0:
                cells.append(f"{'—':>19}")
            elif values.size == 1:
                cells.append(f"{values[0]:+6.1f}        k1".rjust(19))
            else:
                se = values.std(ddof=1) / np.sqrt(values.size)
                cells.append(f"{values.mean():+6.1f} ± {se:4.1f} k{values.size}".rjust(19))
        print(f"{key[0]:>5} {key[1]:>3} {key[2]:>6} | " + " | ".join(cells))


def print_planted(summary: list[dict]) -> None:
    names = ("inverted", "dead", "insufficient", "usable", "edge")
    print("items var models | planted | " + "  ".join(f"{n:>12}" for n in names) + " | fitted a<0")
    for s in summary:
        total = s["planted_inverted"]
        shares = "  ".join(f"{s[f'planted_inverted_as_{n}'] / total:12.2f}" for n in names)
        negative = s["planted_inverted_fitted_negative"] / total
        print(
            f"{s['items']:>5} {s['variants']:>3} {s['models']:>6} | {total:7.1f} | {shares} | {negative:10.2f}"
        )


def _rounded(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fits", type=Path, help="directory of artifacts written by run.py")
    source.add_argument("--from-json", type=Path, help="re-print tables from a results file")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.from_json:
        rows = json.loads(args.from_json.read_text())["fits"]
    else:
        paths = sorted(p for p in args.fits.glob("*.irt") if NAME.search(p.name))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [
                json.loads(json.dumps(r, default=_plain))
                for r in pool.map(measure, paths, chunksize=4)
            ]
    summary = aggregate(rows)
    for population in POPULATIONS:
        tag = "" if population == "same" else f"{population}_"
        for n in ANCHOR_SIZES:
            print(f"== fresh models: {population} {POPULATIONS[population]}, anchor size {n}")
            print_table(summary, n, tag)
            print()
            print(f"== paired differences x1000: {population}, anchor size {n}")
            print_paired(rows, n, tag)
            print()
    print("== where planted mis-keyed items end up (share of planted)")
    print_planted(summary)
    if args.json:
        # One line per fit: indented, the file is four times the size for no
        # reader's benefit, and rounding past 1e-5 is noise in a Spearman.
        lines = ",\n".join(json.dumps(_rounded(r), separators=(",", ":")) for r in rows)
        head = json.dumps(_rounded({"populations": POPULATIONS, "cells": summary}), indent=1)
        args.json.write_text(head[:-2] + ',\n "fits": [\n' + lines + "\n ]\n}\n")


if __name__ == "__main__":
    main()
