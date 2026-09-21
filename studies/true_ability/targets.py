"""Does random beat select against true ability, or only against observed full-suite accuracy?

Same leave_one_model_out loop, same cached holdout fits; only the target changes.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.stats import spearmanr

import irtcheck.validate as V
from irtcheck.synth import synthetic_matrix

p = argparse.ArgumentParser()
p.add_argument("--models", type=int, required=True)
p.add_argument("--variants", type=int, default=1)
p.add_argument("--items", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
args = p.parse_args()

matrix, truth = synthetic_matrix(
    n_models=args.models, n_items=args.items, variants_per_model=args.variants, seed=args.seed
)

models = sorted(set(truth.derives_from))
theta_of = {
    m: float(np.mean([t for t, d in zip(truth.theta, truth.derives_from, strict=True) if d == m]))
    for m in models
}


def expected_acc(m):
    thetas = [t for t, d in zip(truth.theta, truth.derives_from, strict=True) if d == m]
    return float(np.mean([np.mean(1 / (1 + np.exp(-truth.a * (t - truth.b)))) for t in thetas]))


expacc_of = {m: expected_acc(m) for m in models}

observed = V.full_suite_accuracy


def constant(values):
    def target(_matrix):
        return values

    return target


targets = {
    "full-suite accuracy (what validate uses)": observed,
    "true ability theta": constant(theta_of),
    "true expected accuracy (noise-free)": constant(expacc_of),
}

base_fit = V.default_fit_fn(seed=0, epochs=2000)
cache: dict[int, object] = {}


def fit_fn(held_out):
    k = held_out.n_respondents, tuple(sorted(set(held_out.derives_from)))
    if k not in cache:
        print(f"  fitting without {sorted(set(models) - set(k[1]))}", file=sys.stderr, flush=True)
        cache[k] = base_fit(held_out)
    return cache[k]


obs = observed(matrix)


def rho(x, y):
    return float(spearmanr(x, y)[0])


print(f"{args.models} models x {args.variants} variants, {args.items} items, seed {args.seed}")
print(
    f"Spearman(observed full-suite acc, theta) = {rho([obs[m] for m in models], [theta_of[m] for m in models]):+.3f}"
)
print(
    f"Spearman(observed full-suite acc, expected acc) = {rho([obs[m] for m in models], [expacc_of[m] for m in models]):+.3f}"
)
print(
    f"Spearman(theta, expected acc) = {rho([theta_of[m] for m in models], [expacc_of[m] for m in models]):+.3f}"
)
for name, fn in targets.items():
    V.full_suite_accuracy = fn
    r = V.leave_one_model_out(matrix, fit_fn=fit_fn, seed=0)
    print(f"\ntarget: {name}   (places off, out of {r.n_models})")
    print("    n | select | random | beats | Spearman | random")
    for s in r.results:
        print(
            f"  {s.size:3d} | {s.mean_error:6.2f} | {s.random_error:6.2f} | {s.beats_random:4.0%} "
            f"| {s.spearman:+.3f}   | {s.random_spearman:+.3f}"
        )
    sys.stdout.flush()
V.full_suite_accuracy = observed
