"""Why does select lose to random on a synthetic 2PL? Two diagnostics and an oracle.

Reads holdout fits from --cache, fitting (and caching) any that are missing.
--holdouts K holds out K models spread evenly across the true-ability ranking;
every model is still scored on every set.

  oracle      select_item_ids on the held-out fit with a, b and theta replaced by
              the truth (intervals pinched, flags recomputed by compute_flags).
              Oracle beats random -> estimation noise; oracle loses -> objective.
  true a      mean TRUE discrimination of the chosen items, against the mean
              FITTED a select saw when choosing them (winner's curse).
  spread      SD of the chosen items' TRUE difficulty, and how many models get
              every item right or every item wrong (saturated -> tied).
  info        mean true test information at the models' true abilities, and at
              the weakest-served model (min over models).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import irtcheck.validate as V
from irtcheck.artifact import IrtFit, Posterior, compute_flags
from irtcheck.select import item_information, select_item_ids
from irtcheck.synth import synthetic_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=int, required=True)
parser.add_argument("--variants", type=int, default=1)
parser.add_argument("--items", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--holdouts", type=int, default=0, help="0 = every model")
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

matrix, truth = synthetic_matrix(
    n_models=args.models, n_items=args.items, variants_per_model=args.variants, seed=args.seed
)
models = sorted(set(truth.derives_from))
pairs = list(zip(truth.theta, truth.derives_from, strict=True))
theta_of = {m: float(np.mean([t for t, d in pairs if d == m])) for m in models}
model_theta = np.array([theta_of[m] for m in models])
true_item = {item: i for i, item in enumerate(truth.item_ids)}
true_resp = {r: i for i, r in enumerate(truth.respondent_ids)}
table = V.AccuracyTable(matrix)


def cache_path(model_id: str) -> Path:
    return args.cache / (re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + ".irt")


def fit_one(model_id: str) -> str:
    path = cache_path(model_id)
    if not path.exists():
        V.default_fit_fn(seed=0, epochs=2000)(matrix.drop_model(model_id)).save(path)
    return model_id


def load(model_id: str) -> IrtFit:
    return IrtFit.load(cache_path(model_id))


def pinched(values) -> Posterior:
    v = [float(x) for x in values]
    return Posterior(
        mean=v, sd=[1e-3] * len(v), hdi_low=[x - 2e-3 for x in v], hdi_high=[x + 2e-3 for x in v]
    )


def oracle(fit: IrtFit) -> IrtFit:
    idx = [true_item[i] for i in fit.item_ids]
    a, b = pinched(truth.a[idx]), pinched(truth.b[idx])
    theta = pinched([truth.theta[true_resp[r]] for r in fit.respondent_ids])
    flags = compute_flags(a, b, list(fit.p_correct), theta.mean)
    return dataclasses.replace(fit, a=a, b=b, theta=theta, flags=flags)


_oracles: dict[int, IrtFit] = {}


def oracle_select(fit: IrtFit, n: int) -> list[str]:
    if id(fit) not in _oracles:
        _oracles.clear()
        _oracles[id(fit)] = oracle(fit)
    return select_item_ids(_oracles[id(fit)], n)


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
fits = {m: load(m) for m in held}


def fit_fn(held_out):
    (missing,) = set(models) - set(held_out.derives_from)
    return fits[missing]


# -- placement: the unchanged loop, target true theta, scored by accuracy -----
V.full_suite_accuracy = lambda _m: theta_of  # noqa: E731 - a constant target
placed = {
    "select": V.leave_one_model_out(matrix, fit_fn=fit_fn, holdouts=held, seed=0),
    "oracle": V.leave_one_model_out(
        matrix, fit_fn=fit_fn, select_fn=oracle_select, holdouts=held, seed=0
    ),
}

# -- diagnostics per holdout, averaged ----------------------------------------
rng = np.random.default_rng(0)
rows: dict[tuple[str, int], list[tuple[float, ...]]] = {}


def describe(fit: IrtFit, item_ids: list[str]) -> tuple[float, ...]:
    idx = np.array([true_item[i] for i in item_ids])
    fitted = {i: k for k, i in enumerate(fit.item_ids)}
    fitted_a = float(np.mean([fit.a.mean[fitted[i]] for i in item_ids]))
    acc = table.accuracy(table.columns(item_ids))
    saturated = int(np.sum((acc <= 0.0) | (acc >= 1.0)))
    info = item_information(truth.a[idx], truth.b[idx], model_theta).sum(axis=0)
    return (
        float(np.mean(truth.a[idx])),
        fitted_a,
        float(np.std(truth.b[idx])),
        saturated,
        float(np.mean(info)),
        float(np.min(info)),
    )


for m in held:
    fit = fits[m]
    orc = oracle(fit)
    answered = [i for i, c in zip(fit.item_ids, fit.n_resp, strict=True) if c > 0]
    for n in V.DEFAULT_SIZES:
        chosen = select_item_ids(fit, n)
        rows.setdefault(("select", n), []).append(describe(fit, chosen))
        rows.setdefault(("oracle", n), []).append(describe(fit, select_item_ids(orc, n)))
        for _ in range(20):
            pick = list(rng.choice(answered, size=len(chosen), replace=False))
            rows.setdefault(("random", n), []).append(describe(fit, pick))

print(
    f"{args.models} models x {args.variants} variants, {args.items} items, seed {args.seed}, "
    f"{len(held)} held out"
)
print(f"true model abilities span {model_theta.min():+.2f} .. {model_theta.max():+.2f}")
print("\nPlaces off vs true theta, scored by accuracy ('beats' = share of 200 random draws)")
print("    n |  select  beats |  oracle  beats |  random")
for s, o in zip(placed["select"].results, placed["oracle"].results, strict=True):
    print(
        f"  {s.size:3d} |  {s.mean_error:6.2f}  {s.beats_random:4.0%}  |  {o.mean_error:6.2f}  {o.beats_random:4.0%}"
        f"  |  {s.random_error:6.2f}"
    )

print("\nWhat the sets contain (mean over holdouts; random = 20 draws per holdout)")
print("    n  set     | true a  fitted a | sd(true b)  saturated models | info mean  info min")
for n in V.DEFAULT_SIZES:
    for who in ("select", "oracle", "random"):
        ta, fa, sb, sat, im, imin = np.mean(rows[(who, n)], axis=0)
        print(
            f"  {n:3d}  {who:7s} | {ta:6.2f}  {fa:8.2f} | {sb:10.2f}  {sat:16.2f} | {im:9.2f}  {imin:8.2f}"
        )
