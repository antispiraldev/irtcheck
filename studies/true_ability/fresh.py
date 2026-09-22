"""Is select choosing on the same responses it is then scored on?

A held-out fit chooses items using every other model's responses, and
`validate` then scores those same models on those same responses. An item
tends to look informative when the other models' answers to it happen to line
up with their order, so on the chosen items the reference models are spread
out by their own noise while the held-out model, whose answers played no part
in the choice, is not. That would pull the held-out model toward the middle.

Synthetic data can separate the two: choose from the original responses (the
cached holdout fits), then score every model on a *fresh* draw of responses
from the same true parameters. Choosing is unchanged; only the conditioning on
the scoring data is removed. Random sets are unaffected by construction.

Reads the fits diagnose.py cached; same arguments. Target is true theta.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import irtcheck.validate as V
from irtcheck.artifact import IrtFit
from irtcheck.matrix import build_matrix
from irtcheck.select import padding_items, select_item_ids
from irtcheck.synth import responses_from_truth, synthetic_matrix, synthetic_records

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=int, required=True)
parser.add_argument("--variants", type=int, default=1)
parser.add_argument("--items", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--holdouts", type=int, default=0, help="0 = every model")
parser.add_argument("--repeats", type=int, default=20)
args = parser.parse_args()

original, truth = synthetic_matrix(
    n_models=args.models, n_items=args.items, variants_per_model=args.variants, seed=args.seed
)
key = ("model_id", "prompt_variant") if "|" in truth.respondent_ids[0] else ("model_id",)
fresh = build_matrix(
    synthetic_records(truth, responses_from_truth(truth, seed=args.seed + 1000)), respondent_key=key
)
models = sorted(set(truth.derives_from))
pairs = list(zip(truth.theta, truth.derives_from, strict=True))
theta_of = {m: float(np.mean([t for t, d in pairs if d == m])) for m in models}


def cache_path(model_id: str) -> Path:
    return args.cache / (re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + ".irt")


by_ability = sorted(models, key=lambda m: theta_of[m])
if args.holdouts and args.holdouts < len(models):
    spots = np.linspace(0, len(models) - 1, args.holdouts).round().astype(int)
    held = [by_ability[i] for i in dict.fromkeys(spots.tolist())]
else:
    held = models
missing = [m for m in held if not cache_path(m).exists()]
if missing:
    raise SystemExit(f"no cached fit for {missing[:3]}: run diagnose.py with the same arguments")
fits = {m: IrtFit.load(cache_path(m)) for m in held}


def fit_fn(held_out):
    (gone,) = set(models) - set(held_out.derives_from)
    return fits[gone]


usable_rng = np.random.default_rng(2)


def usable_random(fit: IrtFit, n: int) -> list[str]:
    usable = fit.usable_items()
    first = list(usable_rng.choice(usable, size=min(n, len(usable)), replace=False))
    if len(first) < n:
        pad = padding_items(fit, exclude=first)
        first += list(usable_rng.choice(pad, size=min(n - len(first), len(pad)), replace=False))
    return [fit.item_ids[i] for i in first]


V.full_suite_accuracy = lambda _m: theta_of  # noqa: E731 - a constant target


def run(matrix, select_fn, seed: int = 0) -> V.ValidationReport:
    return V.leave_one_model_out(
        matrix, fit_fn=fit_fn, select_fn=select_fn, holdouts=held, seed=seed
    )


reports = {
    ("select", "original"): [run(original, select_item_ids)],
    ("select", "fresh"): [run(fresh, select_item_ids)],
    ("usable+random", "original"): [
        run(original, usable_random, seed=r) for r in range(args.repeats)
    ],
    ("usable+random", "fresh"): [run(fresh, usable_random, seed=r) for r in range(args.repeats)],
}

print(
    f"{args.models} models x {args.variants} variants, {args.items} items, seed {args.seed}, "
    f"{len(held)} held out. Sets chosen from the original responses; every model scored on"
    " either the original responses or a fresh draw from the same true parameters."
)
print("\nPlaces off vs true theta, scored by accuracy; 'beats' = share of 200 random draws beaten")
names = list(reports)
print("    n | " + " | ".join(f"{a + ' / ' + b:>24s}" for a, b in names) + " | random orig / fresh")
for k, n in enumerate(V.DEFAULT_SIZES):
    cells = []
    for name in names:
        err = np.mean([r.results[k].mean_error for r in reports[name]])
        beats = np.mean([r.results[k].beats_random for r in reports[name]])
        cells.append(f"{err:6.2f}  {beats:5.0%}")
    orig = reports[("select", "original")][0].results[k].random_error
    new = reports[("select", "fresh")][0].results[k].random_error
    print(f"  {n:3d} | " + " | ".join(f"{c:>24s}" for c in cells) + f" |  {orig:6.2f} / {new:6.2f}")
