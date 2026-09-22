"""Place a new model by ability, against the reference models' full-suite places.

`validate` scores *every* model on the anchor set, so the reference models are
re-scored on items chosen from their own responses, which pulls the held-out
model toward the middle (see fresh.py). A user does not have to work that way.
The fit they already have gives both the item parameters and an ability for
every model they already ran, over the whole suite. A new model can then be
placed by the ability estimated from its anchor answers alone:

    theta_hat(k) = argmax_theta  log N(theta; 0, 1)
                                 + sum_{i in S} log P(y_ki | a_i, b_i, theta)

    place(k) = 1 + #{m != k : theta_m(full-suite fit) > theta_hat(k)}

Nothing re-scores the reference models on S, so no selection touches the
numbers k is compared against. Only k's own responses do, and they played no
part in choosing S. Places off is still measured against the true ability
ranking, and random sets are drawn and placed exactly the same way.

Reads the fits diagnose.py cached; same arguments.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.special import expit

from irtcheck.artifact import IrtFit
from irtcheck.select import padding_items, select_item_ids
from irtcheck.synth import synthetic_matrix
from irtcheck.validate import DEFAULT_SIZES

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=int, required=True)
parser.add_argument("--variants", type=int, default=1)
parser.add_argument("--items", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--holdouts", type=int, default=0, help="0 = every model")
parser.add_argument("--draws", type=int, default=200)
args = parser.parse_args()

matrix, truth = synthetic_matrix(
    n_models=args.models, n_items=args.items, variants_per_model=args.variants, seed=args.seed
)
models = sorted(set(truth.derives_from))
pairs = list(zip(truth.theta, truth.derives_from, strict=True))
theta_of = {m: float(np.mean([t for t, d in pairs if d == m])) for m in models}

# Dense responses, as AccuracyTable builds them: one row per respondent.
correct = np.zeros((matrix.n_respondents, matrix.n_items), dtype=np.float64)
answered = np.zeros_like(correct)
np.add.at(correct, (matrix.rows, matrix.cols), matrix.obs)
np.add.at(answered, (matrix.rows, matrix.cols), 1.0)
column = {item: c for c, item in enumerate(matrix.item_ids)}
rows_of = {m: [r for r, src in enumerate(matrix.derives_from) if src == m] for m in models}


def cache_path(model_id: str) -> Path:
    return args.cache / (re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + ".irt")


by_ability = sorted(models, key=lambda m: theta_of[m])
if args.holdouts and args.holdouts < len(models):
    spots = np.linspace(0, len(models) - 1, args.holdouts).round().astype(int)
    held = [by_ability[i] for i in dict.fromkeys(spots.tolist())]
else:
    held = models
gone = [m for m in held if not cache_path(m).exists()]
if gone:
    raise SystemExit(f"no cached fit for {gone[:3]}: run diagnose.py with the same arguments")
fits = {m: IrtFit.load(cache_path(m)) for m in held}


def map_theta(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """MAP ability under theta ~ N(0, 1), by bisection on a strictly decreasing score."""

    def score(t: float) -> float:
        return float(-t + (a * (y - expit(a * (t - b)))).sum())

    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def reference_places(fit: IrtFit) -> dict[str, float]:
    """Each reference model's ability from the fit it is already in."""
    theta = np.asarray(fit.theta.mean, dtype=float)
    per: dict[str, list[float]] = {}
    for value, src in zip(theta, fit.derives_from, strict=True):
        per.setdefault(src, []).append(float(value))
    return {m: float(np.mean(v)) for m, v in per.items()}


def place(
    model_id: str,
    fit: IrtFit,
    item_ids: list[str],
    ref: dict[str, float],
    where: dict[str, int],
) -> float:
    idx = [where[i] for i in item_ids]
    a = np.asarray(fit.a.mean, dtype=float)[idx]
    b = np.asarray(fit.b.mean, dtype=float)[idx]
    cols = np.array([column[i] for i in item_ids], dtype=np.int64)
    thetas = []
    for r in rows_of[model_id]:
        seen = answered[r, cols] > 0
        if seen.any():
            thetas.append(map_theta(a[seen], b[seen], correct[r, cols][seen]))
    estimate = float(np.mean(thetas))
    above = sum(1 for m, t in ref.items() if t > estimate)
    ties = sum(1 for m, t in ref.items() if t == estimate)
    return 1 + above + ties / 2


true_place = {
    m: 1 + sum(1 for other in models if other != m and theta_of[other] > theta_of[m])
    for m in models
}

rng = np.random.default_rng(0)
usable_rng = np.random.default_rng(2)
errors: dict[tuple[str, int], list[float]] = {}
draw_errors: dict[int, list[list[float]]] = {
    n: [[] for _ in range(args.draws)] for n in DEFAULT_SIZES
}

for k in held:
    fit = fits[k]
    ref = reference_places(fit)
    where = {item: i for i, item in enumerate(fit.item_ids)}
    answered_items = [i for i, c in zip(fit.item_ids, fit.n_resp, strict=True) if c > 0]
    for n in DEFAULT_SIZES:
        chosen = select_item_ids(fit, n)
        errors.setdefault(("select", n), []).append(
            abs(place(k, fit, chosen, ref, where) - true_place[k])
        )
        usable = fit.usable_items()
        pool = list(usable_rng.choice(usable, size=min(n, len(usable)), replace=False))
        if len(pool) < n:
            pad = padding_items(fit, exclude=pool)
            pool += list(usable_rng.choice(pad, size=min(n - len(pool), len(pad)), replace=False))
        errors.setdefault(("usable+random", n), []).append(
            abs(place(k, fit, [fit.item_ids[i] for i in pool], ref, where) - true_place[k])
        )
        for d in range(args.draws):
            pick = list(rng.choice(answered_items, size=min(n, len(answered_items)), replace=False))
            draw_errors[n][d].append(abs(place(k, fit, pick, ref, where) - true_place[k]))

print(
    f"{args.models} models x {args.variants} variants, {args.items} items, seed {args.seed}, "
    f"{len(held)} held out. Each held-out model is placed by the ability estimated from its own "
    f"answers to the set, against the reference models' abilities in the same fit; "
    f"{args.draws} random sets per size."
)
print("\nPlaces off vs the true ability ranking; 'beats' = share of random sets beaten")
print("    n |         select |  usable+random |  random")
for n in DEFAULT_SIZES:
    draws = [float(np.mean(d)) for d in draw_errors[n]]
    cells = []
    for name in ("select", "usable+random"):
        err = float(np.mean(errors[(name, n)]))
        beats = np.mean([1.0 if err < d else 0.5 if err == d else 0.0 for d in draws])
        cells.append(f"{err:6.2f}  {beats:5.0%}")
    print(f"  {n:3d} | " + " | ".join(f"{c:>14s}" for c in cells) + f" |  {np.mean(draws):6.2f}")
