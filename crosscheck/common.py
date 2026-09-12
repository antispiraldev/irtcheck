"""Paths, the two datasets every estimator is run on, and the interchange format.

Nothing in here imports torch, pyro, py_irt or irtcheck's fitter. It is the one
module all five runners share, and two of them (`run_pyirt.py`, and R by way of
`run_mirt.py`) execute in interpreters that have never heard of this project.
So the interchange is boring on purpose: JSON for the matrix, wide CSV for R,
JSON Lines for py-irt, and one JSON file per estimator holding nothing but
point estimates and the convention they are expressed in.

## The two regimes, and why there are two

A cross-check that only runs where every estimator is well-determined proves
the code is right and says nothing about the tool; a cross-check that only runs
in the tool's real regime cannot distinguish a bug from a prior. So:

  - `dense`  — 300 respondents x 60 items. Every item parameter is pinned by
    the data. Here a disagreement between three independent implementations is
    a bug in one of them, and that is what makes this the bug-detection regime.

  - `sparse` — 12 respondents x 200 items, 10% of cells missing. This is what
    irtcheck is actually pointed at. Estimators are *expected* to diverge here,
    and the interesting question is not agreement but whether our intervals are
    honest about how little is known.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CROSSCHECK = Path(__file__).resolve().parent
WORK = CROSSCHECK / "work"
RESULTS = CROSSCHECK / "results"

# Where setup.sh puts the two reference toolchains. Outside the repository,
# because neither can live in the project environment: py-irt refuses to install
# on 3.12 at all, and R is not a Python package.
REFERENCE_HOME = Path(
    os.environ.get("IRTCHECK_CROSSCHECK_HOME", Path.home() / ".cache" / "irtcheck-crosscheck")
)

# Both datasets come out of irtcheck.synth, which is the only place ground truth
# exists in this repo (CLAUDE.md, "Testing a stochastic fit"). The seeds are
# fixed so a rerun on another machine compares like with like.
DATASETS: dict[str, dict[str, Any]] = {
    "dense": {
        "n_models": 300,
        "n_items": 60,
        "variants_per_model": 1,
        "missing": 0.0,
        "seed": 11,
        "why": "every item parameter is identified; disagreement here is a bug",
    },
    "sparse": {
        "n_models": 12,
        "n_items": 200,
        "variants_per_model": 1,
        "missing": 0.10,
        "seed": 23,
        "why": "irtcheck's real regime; divergence is expected and intervals are the claim",
    },
}


@dataclass(slots=True)
class Estimates:
    """One estimator's point estimates, plus the convention they arrive in.

    `a_positive` records whether the estimator constrains discrimination to be
    positive. It is not decoration: an estimator that does cannot land in the
    mirrored mode, so it is the fixed point the reflection check is made
    against, and an estimator that does not (ours) has to be checked.
    """

    name: str
    dataset: str
    item_ids: list[str]
    a: np.ndarray
    b: np.ndarray
    respondent_ids: list[str]
    theta: np.ndarray
    parameterisation: str
    a_positive: bool
    theta_metric: str
    detail: dict[str, Any]

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "dataset": self.dataset,
                    "item_ids": self.item_ids,
                    "a": [float(x) for x in self.a],
                    "b": [float(x) for x in self.b],
                    "respondent_ids": self.respondent_ids,
                    "theta": [float(x) for x in self.theta],
                    "parameterisation": self.parameterisation,
                    "a_positive": self.a_positive,
                    "theta_metric": self.theta_metric,
                    "detail": self.detail,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def from_json(cls, path: Path) -> Estimates:
        d = json.loads(Path(path).read_text())
        return cls(
            name=d["name"],
            dataset=d["dataset"],
            item_ids=d["item_ids"],
            a=np.asarray(d["a"], dtype=float),
            b=np.asarray(d["b"], dtype=float),
            respondent_ids=d["respondent_ids"],
            theta=np.asarray(d["theta"], dtype=float),
            parameterisation=d["parameterisation"],
            a_positive=d["a_positive"],
            theta_metric=d["theta_metric"],
            detail=d.get("detail", {}),
        )


def work_dir(dataset: str) -> Path:
    return WORK / dataset


def estimates_path(dataset: str, name: str) -> Path:
    return work_dir(dataset) / f"est-{name}.json"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_matrix(dataset: str) -> dict[str, Any]:
    """The dense 0/1/NaN rectangle plus ids, as written by export_inputs.py."""
    payload = read_json(work_dir(dataset) / "matrix.json")
    responses = np.asarray(payload["responses"], dtype=float)
    payload["responses"] = responses
    return payload


def load_truth(dataset: str) -> dict[str, Any]:
    payload = read_json(work_dir(dataset) / "truth.json")
    for key in ("a", "b", "theta"):
        payload[key] = np.asarray(payload[key], dtype=float)
    return payload
