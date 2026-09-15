"""Fit a sweep of synthetic suites across respondent counts and save every artifact.

Stage one of the minimum-respondent study (docs/spec.md, Open questions). It
fits and saves; `analyse.py` reads the artifacts and measures. The split is so
that a new question about the fits — a different gate, a different anchor size
— costs seconds rather than a re-run of every fit.

    .venv/bin/python studies/min_respondents/run.py --out ~/.cache/irtcheck-study

Each fit is pinned to one thread and fits run in parallel processes: a 2PL on a
few hundred items is too small for intra-op threading to pay, and N processes
each spawning N threads is the oversubscription that made a run crawl before.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MODELS = (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100)
ITEMS = (200, 800)
VARIANTS = (1, 3)
SEEDS = tuple(range(6))

# Real suites carry a few mis-keyed items, and whether a selection gate keeps
# them out is one of the things being measured. Every other constant is
# synth.py's default, so these universes are the ones the rest of the test
# suite already reasons about.
INVERTED_FRACTION = 0.02


def artifact_name(models: int, items: int, variants: int, seed: int) -> str:
    return f"m{models:03d}_i{items:04d}_v{variants}_s{seed}.irt"


def fit_one(out: str, models: int, items: int, variants: int, seed: int) -> tuple[str, float]:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)

    from irtcheck.fit.fitter import fit_2pl
    from irtcheck.synth import synthetic_matrix

    path = Path(out) / artifact_name(models, items, variants, seed)
    if path.exists():
        return path.name, 0.0
    matrix, _ = synthetic_matrix(
        n_models=models,
        n_items=items,
        variants_per_model=variants,
        inverted_fraction=INVERTED_FRACTION,
        seed=seed,
    )
    started = time.time()
    fit = fit_2pl(matrix, seed=seed)
    fit.save(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    return path.name, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(MODELS, ITEMS, VARIANTS, SEEDS))
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_one, str(args.out), *cell) for cell in grid]
        for done, future in enumerate(as_completed(futures), start=1):
            name, seconds = future.result()
            print(f"[{done:>3}/{len(grid)}] {name} {seconds:5.1f}s", flush=True)
    print(f"{len(grid)} fits in {time.time() - started:.0f}s")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
