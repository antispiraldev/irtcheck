"""Does select win when its sets are scored by estimated ability instead of accuracy?

2x2 on a synthetic matrix: {select, random} x {plain accuracy, MAP ability}.
MAP ability is theta with item parameters held fixed at the held-out fit's
means and a N(0, 1) prior — the `map_theta` that validate carried before the
fixed-set protocol, vectorised over respondents. Every arm runs through the
unchanged `leave_one_model_out` loop on the same cached holdout fits; only the
scoring table is swapped. Both selectors are scored the same way within an arm.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

import irtcheck.validate as V
from irtcheck.artifact import IrtFit
from irtcheck.synth import synthetic_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=int, required=True)
parser.add_argument("--variants", type=int, default=1)
parser.add_argument("--items", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
args = parser.parse_args()
args.cache.mkdir(parents=True, exist_ok=True)

matrix, truth = synthetic_matrix(
    n_models=args.models, n_items=args.items, variants_per_model=args.variants, seed=args.seed
)
models = sorted(set(truth.derives_from))
pairs = list(zip(truth.theta, truth.derives_from, strict=True))
theta_of = {m: float(np.mean([t for t, d in pairs if d == m])) for m in models}

# -- holdout fits, cached on disk so every arm sees the same ones -------------
base_fit = V.default_fit_fn(seed=0, epochs=2000)
current: dict[str, IrtFit] = {}


def fit_fn(held_out):
    (missing,) = set(models) - set(held_out.derives_from)
    path = args.cache / (re.sub(r"[^A-Za-z0-9._-]", "_", missing) + ".irt")
    if path.exists():
        fit = IrtFit.load(path)
    else:
        print(f"  fitting without {missing}", file=sys.stderr, flush=True)
        fit = base_fit(held_out)
        fit.save(path)
    current["fit"] = fit
    return fit


# -- scoring by MAP ability ----------------------------------------------------
class AbilityTable(V.AccuracyTable):
    """Per-model mean MAP theta on `cols`, item parameters from the current fit."""

    def _params(self):
        fit = current["fit"]
        index = {item: i for i, item in enumerate(fit.item_ids)}
        a = np.full(len(self.column), np.nan)
        b = np.full(len(self.column), np.nan)
        fa, fb = np.asarray(fit.a.mean, float), np.asarray(fit.b.mean, float)
        for item, c in self.column.items():
            if item in index:
                a[c], b[c] = fa[index[item]], fb[index[item]]
        return a, b

    def accuracy(self, cols):
        a_all, b_all = self._params()
        a, b = a_all[cols], b_all[cols]
        known = np.isfinite(a)
        a, b, cols = a[known], b[known], cols[known]
        answered = self._answered[:, cols] > 0
        y = self._correct[:, cols]
        lo = np.full(answered.shape[0], -12.0)
        hi = np.full(answered.shape[0], 12.0)
        for _ in range(50):  # bisection on a strictly decreasing score
            mid = 0.5 * (lo + hi)
            p = expit(a * (mid[:, None] - b))
            score = np.sum(answered * a * (y - p), axis=1) - mid
            up = score > 0
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        theta = 0.5 * (lo + hi)
        ok = answered.any(axis=1)
        n = len(self._per_model)
        total = np.bincount(self._owner[ok], weights=theta[ok], minlength=n)
        count = np.bincount(self._owner[ok], minlength=n)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def constant(values):
    def target(_matrix):
        return values

    return target


observed = V.full_suite_accuracy
obs = observed(matrix)
print(f"{args.models} models x {args.variants} variants, {args.items} items, seed {args.seed}")
rho = spearmanr([obs[m] for m in models], [theta_of[m] for m in models])[0]
print(f"Spearman(observed full-suite accuracy, true theta) = {rho:+.3f}")

scorers = {"accuracy": V.AccuracyTable, "MAP ability": AbilityTable}
targets = {"true theta": constant(theta_of), "observed full-suite accuracy": observed}
for target_name, target in targets.items():
    V.full_suite_accuracy = target
    results = {}
    for scorer_name, table in scorers.items():
        V.AccuracyTable = table
        results[scorer_name] = V.leave_one_model_out(matrix, fit_fn=fit_fn, seed=0)
    V.AccuracyTable = scorers["accuracy"]
    print(
        f"\ntarget: {target_name}   places off, out of {args.models}; 'beats' = share of 200 random draws"
    )
    print("        |  scored by accuracy               |  scored by MAP ability")
    print("    n   |  select  random  beats  Spearman  |  select  random  beats  Spearman (random)")
    acc, abl = results["accuracy"].results, results["MAP ability"].results
    for s, t in zip(acc, abl, strict=True):
        print(
            f"  {s.size:4d}  |  {s.mean_error:6.2f}  {s.random_error:6.2f}  {s.beats_random:4.0%}"
            f"   {s.spearman:+.3f}  |  {t.mean_error:6.2f}  {t.random_error:6.2f}  {t.beats_random:4.0%}"
            f"   {t.spearman:+.3f} ({t.random_spearman:+.3f})"
        )
    sys.stdout.flush()
V.full_suite_accuracy = observed
