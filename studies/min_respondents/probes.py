"""The three follow-up measurements docs/validation.md §4 quotes beside the sweep.

Each one was a question the sweep raised rather than answered, so each reads
the sweep's artifacts where it can and fits afresh only where it has to.

    PYTHONPATH=src .venv/bin/python studies/min_respondents/probes.py --fits ~/.cache/irtcheck-study

1. `spread`  — why intervals fail below six models: fitted ability spread
               against true, seed 0.
2. `barrier` — why mis-keyed items stay positive: |logit p| of planted items
               by the sign their fitted `a` ended on, at 40 and 100 models.
3. `misses`  — which side the interval misses truth on at many respondents,
               averaged over the six seeds of four sweep cells.
4. `epochs`  — whether the coverage loss at many respondents is convergence:
               two cells refitted at 2,000 and 8,000 epochs. ~5 minutes.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logit

from irtcheck.artifact import FLAG_CEILING, FLAG_FLOOR, IrtFit
from irtcheck.synth import make_truth

NAME = re.compile(r"m(\d+)_i(\d+)_v(\d)_s(\d+)\.irt$")
INVERTED_FRACTION = 0.02  # must match run.py


def truth_for(models: int, items: int, variants: int, seed: int):
    return make_truth(
        n_models=models,
        n_items=items,
        variants_per_model=variants,
        inverted_fraction=INVERTED_FRACTION,
        seed=seed,
    )


def spread(fits: Path) -> None:
    print("== fitted ability spread vs true (800 items, 1 variant, seed 0)")
    for models in (3, 4, 5, 6, 8, 15):
        fit = IrtFit.load(fits / f"m{models:03d}_i0800_v1_s0.irt")
        truth = truth_for(models, 800, 1, 0)
        print(
            f"{models:>4} models: fitted sd {np.std(fit.theta.mean):.2f}  true sd {np.std(truth.theta):.2f}"
        )


def barrier(fits: Path) -> None:
    negative: list[float] = []
    positive: list[float] = []
    for path in sorted(fits.glob("m040_*.irt")) + sorted(fits.glob("m100_*.irt")):
        models, items, variants, seed = (int(x) for x in NAME.search(path.name).groups())
        fit = IrtFit.load(path)
        truth = truth_for(models, items, variants, seed)
        for i in np.flatnonzero(truth.a < 0):
            if {FLAG_CEILING, FLAG_FLOOR} & set(fit.flags[i]):
                continue
            value = abs(float(logit(np.clip(fit.p_correct[i], 1e-3, 1 - 1e-3))))
            (negative if fit.a.mean[i] < 0 else positive).append(value)
    print("== planted mis-keyed items at 40 and 100 models, excluding ceiling/floor")
    print(f"fitted a < 0: median |logit p| {np.median(negative):.2f}  (n={len(negative)})")
    print(f"fitted a > 0: median |logit p| {np.median(positive):.2f}  (n={len(positive)})")


def misses(fits: Path) -> None:
    print("== where a's interval misses truth (800 items, six seeds, ceiling/floor excluded)")
    for models, variants in ((30, 1), (100, 1), (30, 3), (100, 3)):
        above, below = [], []
        for seed in range(6):
            fit = IrtFit.load(fits / f"m{models:03d}_i0800_v{variants}_s{seed}.irt")
            truth = truth_for(models, 800, variants, seed)
            readable = np.array([not ({FLAG_CEILING, FLAG_FLOOR} & set(f)) for f in fit.flags])
            above.append((truth.a > np.asarray(fit.a.hdi_high))[readable].mean())
            below.append((truth.a < np.asarray(fit.a.hdi_low))[readable].mean())
        print(
            f"{models:>4} models x {variants} variants: truth above interval {np.mean(above):.3f}, "
            f"below {np.mean(below):.3f}"
        )


def _refit(cell: tuple[int, int, int]) -> str:
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch

    torch.set_num_threads(4)
    from irtcheck.fit.fitter import fit_2pl
    from irtcheck.synth import synthetic_matrix

    models, variants, epochs = cell
    matrix, truth = synthetic_matrix(
        n_models=models,
        n_items=800,
        variants_per_model=variants,
        inverted_fraction=INVERTED_FRACTION,
        seed=0,
    )
    fit = fit_2pl(matrix, seed=0, epochs=epochs)
    lo, hi = np.asarray(fit.a.hdi_low), np.asarray(fit.a.hdi_high)
    readable = np.array([not ({FLAG_CEILING, FLAG_FLOOR} & set(f)) for f in fit.flags])
    covered = ((lo <= truth.a) & (truth.a <= hi))[readable].mean()
    return (
        f"{models:>4} models x {variants} variants, {epochs:>5} epochs: coverage {covered:.3f}, "
        f"ELBO gain over last 10% {fit.diagnostics['elbo_improvement_last_10pct']:+.1f}"
    )


def epochs() -> None:
    print("== coverage of a at 2,000 vs 8,000 epochs (800 items, seed 0, ceiling/floor excluded)")
    cells = [(30, 1, 2000), (30, 1, 8000), (100, 3, 2000), (100, 3, 8000)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for line in pool.map(_refit, cells):
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--skip-epochs", action="store_true")
    args = parser.parse_args()
    spread(args.fits)
    barrier(args.fits)
    misses(args.fits)
    if not args.skip_epochs:
        epochs()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
