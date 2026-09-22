"""Two fixes §5 left untried, scored through the unchanged leave_one_model_out loop.

Reads the holdout fits diagnose.py cached (fitting any that are missing), so run
it with the same --models/--variants/--items/--seed/--holdouts and --cache.

  select         select_item_ids, as shipped: top n by information at the point
                 estimates.
  filter+random  drop what the fit makes a definite claim about (dead, inverted,
                 ceiling, floor), draw n of the rest at random. Averaged over
                 --repeats draws per holdout.
  expected info  information averaged over the interval on a, a ~ N(a_hat, se):
                 E[a^2 P(1-P)] by Gauss-Hermite. The fix §5 names. Note it adds
                 se^2 to a^2 and so favours *uncertain* items.
  lower bound    information at the interval's lower end on a (the pessimistic
                 reading), which penalises uncertain items instead.

The last two use select's pool and padding rule; only the score changes.
Target is true theta, as in diagnose.py, so the select column matches §5's sweep.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit

import irtcheck.validate as V
from irtcheck.artifact import FLAG_CEILING, FLAG_DEAD, FLAG_FLOOR, FLAG_INVERTED, IrtFit
from irtcheck.select import ability_distribution, padding_items, select_item_ids
from irtcheck.synth import synthetic_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=int, required=True)
parser.add_argument("--variants", type=int, default=1)
parser.add_argument("--items", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--holdouts", type=int, default=0, help="0 = every model")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--repeats", type=int, default=20)
args = parser.parse_args()

matrix, truth = synthetic_matrix(
    n_models=args.models, n_items=args.items, variants_per_model=args.variants, seed=args.seed
)
models = sorted(set(truth.derives_from))
pairs = list(zip(truth.theta, truth.derives_from, strict=True))
theta_of = {m: float(np.mean([t for t, d in pairs if d == m])) for m in models}


def cache_path(model_id: str) -> Path:
    return args.cache / (re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + ".irt")


def fit_one(model_id: str) -> str:
    path = cache_path(model_id)
    if not path.exists():
        V.default_fit_fn(seed=0, epochs=2000)(matrix.drop_model(model_id)).save(path)
    return model_id


by_ability = sorted(models, key=lambda m: theta_of[m])
if args.holdouts and args.holdouts < len(models):
    spots = np.linspace(0, len(models) - 1, args.holdouts).round().astype(int)
    held = [by_ability[i] for i in dict.fromkeys(spots.tolist())]
else:
    held = models
args.cache.mkdir(parents=True, exist_ok=True)
todo = [m for m in held if not cache_path(m).exists()]
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for done, m in enumerate(pool.map(fit_one, todo), 1):
        print(f"  fitted without {m} ({done}/{len(todo)})", file=sys.stderr, flush=True)
fits = {m: IrtFit.load(cache_path(m)) for m in held}


def fit_fn(held_out):
    (missing,) = set(models) - set(held_out.derives_from)
    return fits[missing]


# -- the selectors ------------------------------------------------------------
DEFINITE = {FLAG_DEAD, FLAG_INVERTED, FLAG_CEILING, FLAG_FLOOR}
draw_rng = np.random.default_rng(1)


def filter_random(fit: IrtFit, n: int) -> list[str]:
    pool = [
        item
        for item, flags, answered in zip(fit.item_ids, fit.flags, fit.n_resp, strict=True)
        if answered > 0 and not DEFINITE.intersection(flags)
    ]
    return list(draw_rng.choice(pool, size=min(n, len(pool)), replace=False))


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9)  # probabilists' Hermite: N(0, 1)
GH_W = GH_W / GH_W.sum()


def scored_select(score_fn):
    def select(fit: IrtFit, n: int) -> list[str]:
        theta, weights = ability_distribution(fit)
        a = np.asarray(fit.a.mean, dtype=float)
        b = np.asarray(fit.b.mean, dtype=float)
        se = (np.asarray(fit.a.hdi_high) - np.asarray(fit.a.hdi_low)) / (2 * 1.96)
        score = score_fn(a, b, se, theta) @ weights
        first = sorted(fit.usable_items(), key=lambda i: -score[i])[:n]
        pad = sorted(padding_items(fit, exclude=first), key=lambda i: -score[i])
        return [fit.item_ids[i] for i in first + pad[: n - len(first)]]

    return select


def info_at(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    p = expit(a[:, None] * (theta[None, :] - b[:, None]))
    return a[:, None] ** 2 * p * (1 - p)


def expected_info(a, b, se, theta):
    return sum(w * info_at(a + x * se, b, theta) for x, w in zip(GH_X, GH_W, strict=True))


def lower_bound_info(a, b, se, theta):
    return info_at(np.maximum(a - 1.96 * se, 0.0), b, theta)


# -- placement: the unchanged loop, target true theta, scored by accuracy -----
V.full_suite_accuracy = lambda _m: theta_of  # noqa: E731 - a constant target


def run(select_fn, seed: int = 0) -> V.ValidationReport:
    return V.leave_one_model_out(
        matrix, fit_fn=fit_fn, select_fn=select_fn, holdouts=held, seed=seed
    )


reports = {
    "select": [run(select_item_ids)],
    "filter+random": [run(filter_random, seed=r) for r in range(args.repeats)],
    "expected info": [run(scored_select(expected_info))],
    "lower bound": [run(scored_select(lower_bound_info))],
}

print(
    f"{args.models} models x {args.variants} variants, {args.items} items, seed {args.seed}, "
    f"{len(held)} held out; filter+random averaged over {args.repeats} draws"
)
print("\nPlaces off vs true theta, scored by accuracy; 'beats' = share of 200 random draws beaten")
names = list(reports)
print("    n | " + " | ".join(f"{name:>14s}" for name in names) + " |  random")
for k, n in enumerate(V.DEFAULT_SIZES):
    cells = []
    for name in names:
        err = np.mean([r.results[k].mean_error for r in reports[name]])
        beats = np.mean([r.results[k].beats_random for r in reports[name]])
        cells.append(f"{err:6.2f}  {beats:5.0%}")
    rand = reports["select"][0].results[k].random_error
    print(f"  {n:3d} | " + " | ".join(f"{c:>14s}" for c in cells) + f" |  {rand:6.2f}")
