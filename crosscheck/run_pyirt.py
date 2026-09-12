"""Fit both datasets with py-irt 0.7.1 and record the estimates.

**Runs under the Python 3.11 venv, not the project venv.** Launch it with

    python crosscheck/run_pyirt.py

and it re-executes itself under the 3.11 interpreter setup.sh created, because
py-irt cannot be installed alongside this project: every release from 0.4 on
declares `Requires-Python >=3.9,<3.12`, and on 3.12 an unpinned
`pip install py-irt` silently resolves to the abandoned 0.1.1 fork. See
../CLAUDE.md and check_pyirt_resolution.sh.

Deliberately stdlib + numpy + py_irt only, so it stays importable in an
interpreter that has no irtcheck installed.

## What py-irt's 2PL actually is

Same likelihood form as ours, `sigmoid(a[i] * (theta[j] - b[i]))`, so its `b`
is our `b` and needs no slope-intercept conversion. Three differences that do
matter, all of them read off py_irt/models/two_param_logistic.py:

  - `a` is **LogNormal**, therefore strictly positive. It cannot reach the
    mirrored mode and it cannot produce an interval containing zero — which is
    precisely the property fit/model.py rejects for this tool, because it makes
    `insufficient-data` unreachable.

  - `export()` returns `exp(loc_slope)`, the **median** of the fitted LogNormal
    guide, not its mean. For a LogNormal(mu, s) the mean is exp(mu + s^2/2), so
    the number py-irt hands you is smaller than the posterior mean by a factor
    of exp(s^2/2). Both are recorded below; the comparison uses the median and
    says so, since that is what a py-irt user gets.

  - with `priors="hierarchical"` the ability distribution is
    `Normal(mu_theta, 1/u_theta)` with **both hyperparameters learned**, so the
    latent location and scale are not identified at all. Its output has to be
    standardised before comparison; conventions.standardise_theta_metric does
    it and reports the factor. With `priors="vague"` theta is a fixed N(0,1),
    the same ruler we use — but then `b ~ Normal(0, 0.1)` and
    `a ~ LogNormal(0, 0.1)`, which are not vague at all but very tight, so that
    run is expected to shrink hard toward a=1, b=0.

Both prior settings are run, because neither alone is an honest comparison.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CROSSCHECK = Path(__file__).resolve().parent
# Resolved without importing crosscheck.common, because this module has to stay
# importable by an interpreter that has never seen this project.
PYIRT_VENV = (
    Path(
        os.environ.get(
            "IRTCHECK_CROSSCHECK_HOME", Path.home() / ".cache" / "irtcheck-crosscheck"
        )
    )
    / "pyirt311"
)
DATASET_NAMES = ("dense", "sparse")
EPOCHS = 2000
SEED = 0


def reexec_under_311() -> None:
    """Hand the script to the 3.11 interpreter, once.

    The guard variable stops an infinite respawn if the target interpreter
    somehow cannot import py_irt either.
    """
    if os.environ.get("IRTCHECK_PYIRT_CHILD") == "1":
        return
    try:
        import py_irt  # noqa: F401

        return
    except ImportError:
        pass

    python = PYIRT_VENV / "bin" / "python"
    if not python.exists():
        sys.exit(
            "py-irt is not available: no Python 3.11 venv at\n"
            f"  {PYIRT_VENV}\n"
            "py-irt declares Requires-Python >=3.9,<3.12 so it cannot live in the project\n"
            "environment. Run crosscheck/setup.sh, which installs a standalone CPython\n"
            "3.11 with `uv python install 3.11` and pins py-irt==0.7.1 into a venv on it.\n"
            "Neither step needs root."
        )
    env = dict(os.environ, IRTCHECK_PYIRT_CHILD="1", PYTHONPATH=str(CROSSCHECK.parent))
    raise SystemExit(subprocess.run([str(python), str(Path(__file__))], env=env).returncode)


reexec_under_311()

import numpy as np  # noqa: E402
import py_irt.anchor_utils  # noqa: E402
import py_irt.dataset  # noqa: E402
import py_irt.initializers  # noqa: E402
import pyro  # noqa: E402
from py_irt.config import IrtConfig  # noqa: E402
from py_irt.dataset import Dataset  # noqa: E402
from py_irt.training import IrtModelTrainer  # noqa: E402

# py-irt's `verbose=False` silences the trainer's own table but not the three
# module-level rich Consoles its dataset loader and initialisers log through,
# which between them emit a screenful per fit. Quieten them so the lines this
# script prints are legible.
for _module in (py_irt.dataset, py_irt.initializers, py_irt.anchor_utils):
    getattr(_module, "console", None) and setattr(_module.console, "quiet", True)


def fit_one(dataset: str, priors: str) -> dict:
    work = CROSSCHECK / "work" / dataset
    data_path = work / "pyirt.jsonl"
    if not data_path.exists():
        sys.exit(f"missing {data_path}; run `python crosscheck/export_inputs.py` first")

    parsed = Dataset.from_jsonlines(data_path)
    config = IrtConfig(
        model_type="2pl",
        priors=priors,
        epochs=EPOCHS,
        # py-irt's own defaults. Changing them to match ours would make the
        # comparison a comparison of optimisers we tuned identically rather than
        # of two packages as they ship.
        lr=0.1,
        lr_decay=0.9999,
        # py-irt's reflection fix: pin the difficulty of the four hardest and
        # four easiest items to +/-3 before the first step, so the optimiser
        # cannot settle into the mirror. Ours does the same job by starting
        # every a at +1 and checking the sign afterwards.
        initializers=["difficulty_sign"],
        seed=SEED,
        deterministic=True,
    )
    pyro.set_rng_seed(SEED)
    pyro.clear_param_store()
    trainer = IrtModelTrainer(data_path=data_path, dataset=parsed, config=config, verbose=False)
    trainer.train(epochs=EPOCHS, device="cpu")

    params = trainer.last_params
    store = pyro.get_param_store()
    loc_slope = store["loc_slope"].detach().cpu().numpy()
    scale_slope = store["scale_slope"].detach().cpu().numpy()

    n_items = len(params["item_ids"])
    item_ids = [params["item_ids"][i] for i in range(n_items)]
    n_subjects = len(params["subject_ids"])
    respondent_ids = [params["subject_ids"][i] for i in range(n_subjects)]

    a_median = np.exp(loc_slope)
    # The LogNormal mean, for the record: exp(mu + s^2/2). The gap between this
    # and the median is the size of the bias a reader takes on by treating
    # py-irt's `disc` as a posterior mean.
    a_mean = np.exp(loc_slope + 0.5 * scale_slope**2)

    theta = np.asarray(params["ability"], dtype=float)
    b = np.asarray(params["diff"], dtype=float)

    detail = {
        "estimator": "SVI / Trace_ELBO (py-irt's own guide and optimiser)",
        "priors": priors,
        "epochs": EPOCHS,
        "seed": SEED,
        "py_irt_version": _pyirt_version(),
        "python": sys.version.split()[0],
        "a_reported_is": "exp(loc_slope), the LogNormal guide MEDIAN, which is what export() returns",
        "a_lognormal_mean": [float(x) for x in a_mean],
        "a_lognormal_sigma": [float(x) for x in scale_slope],
        "median_to_mean_ratio_max": float(np.max(a_mean / np.maximum(a_median, 1e-12))),
        "theta_fitted_sd": float(np.std(theta)),
        "theta_fitted_mean": float(np.mean(theta)),
    }
    if priors == "hierarchical":
        detail["theta_metric_is_free"] = True
        detail["note"] = (
            "theta ~ Normal(mu_theta, 1/u_theta) with both learned: location and scale "
            "are unidentified and must be standardised before comparison"
        )
    else:
        detail["theta_metric_is_free"] = False
        detail["note"] = (
            "py-irt's 'vague' priors are b ~ N(0, 0.1) and a ~ LogNormal(0, 0.1) — "
            "tight, not vague; expect heavy shrinkage toward a=1, b=0"
        )

    payload = {
        "name": f"pyirt-{priors}",
        "dataset": dataset,
        "item_ids": item_ids,
        "a": [float(x) for x in a_median],
        "b": [float(x) for x in b],
        "respondent_ids": respondent_ids,
        "theta": [float(x) for x in theta],
        "parameterisation": "a * (theta - b)",
        "a_positive": True,
        "theta_metric": (
            "theta ~ Normal(mu_theta, 1/u_theta), both learned — NOT identified"
            if priors == "hierarchical"
            else "theta ~ N(0, 1) fixed prior"
        ),
        "detail": detail,
    }
    out = work / f"est-pyirt-{priors}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{dataset}/pyirt-{priors}: {n_items} items, {n_subjects} subjects, "
        f"a median in [{a_median.min():.3f}, {a_median.max():.3f}], "
        f"fitted sd(theta)={np.std(theta):.3f} -> {out.name}"
    )
    return payload


def _pyirt_version() -> str:
    from importlib.metadata import version

    return version("py-irt")


def main() -> None:
    print(f"py-irt {_pyirt_version()} on Python {sys.version.split()[0]}")
    for dataset in DATASET_NAMES:
        for priors in ("vague", "hierarchical"):
            fit_one(dataset, priors)


if __name__ == "__main__":
    main()
