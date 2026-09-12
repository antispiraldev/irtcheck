"""Drive fit_mirt.R and convert its output into the shared estimate format.

    python crosscheck/run_mirt.py

Finds Rscript by trying, in order: `$IRTCHECK_RSCRIPT`, the micromamba
environment setup.sh creates, and whatever is on `PATH`. Exits with a message
naming the missing piece rather than a traceback if none of them has mirt —
"the reference was not available" is a legitimate result of this exercise and
has to be reported as one, not faked.

The conversion from mirt's slope-intercept form to ours happens here, in
`conventions.slope_intercept_to_difficulty`, and is checked against mirt's own
`IRTpars=TRUE` output before anything else is compared.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from crosscheck.common import DATASETS, REFERENCE_HOME, Estimates, estimates_path, work_dir
from crosscheck.conventions import slope_intercept_to_difficulty

MAMBA_ENV = REFERENCE_HOME / "r-mirt"


def find_rscript() -> str | None:
    """Rscript with mirt importable, or None."""
    candidates = [
        os.environ.get("IRTCHECK_RSCRIPT"),
        str(MAMBA_ENV / "bin" / "Rscript"),
        shutil.which("Rscript"),
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-e", 'suppressPackageStartupMessages(library(mirt)); cat("ok")'],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate
    return None


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    return float("nan") if value in ("NA", "", "NaN") else float(value)


def run(dataset: str, rscript: str) -> Estimates:
    out = work_dir(dataset)
    items_csv = out / "mirt-items.csv"
    result = subprocess.run(
        [rscript, str(Path(__file__).with_name("fit_mirt.R")), str(out / "wide.csv"), str(items_csv)],
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"mirt failed on {dataset} (exit {result.returncode})")

    items = read_csv_dicts(items_csv)
    thetas = read_csv_dicts(out / "mirt-items-theta.csv")
    meta = {row["key"]: row["value"] for row in read_csv_dicts(out / "mirt-items-meta.csv")}

    item_ids = [row["item_id"] for row in items]
    a1 = np.array([_float(row["a1"]) for row in items])
    d = np.array([_float(row["d"]) for row in items])
    mirt_a = np.array([_float(row["mirt_a"]) for row in items])
    mirt_b = np.array([_float(row["mirt_b"]) for row in items])

    a, b = slope_intercept_to_difficulty(a1, d)

    # mirt as its own witness. If our b = -d/a1 does not reproduce the a/b mirt
    # reports from the same fit, our understanding of the slope-intercept form
    # is wrong and every comparison downstream of this is meaningless — so it is
    # a hard failure here rather than a soft note in a report nobody reads.
    finite = np.isfinite(b) & np.isfinite(mirt_b)
    conv_a = float(np.max(np.abs(a[finite] - mirt_a[finite])))
    conv_b = float(np.max(np.abs(b[finite] - mirt_b[finite])))
    if not (conv_a < 1e-6 and conv_b < 1e-6):
        raise SystemExit(
            "our slope-intercept conversion disagrees with mirt's own IRTpars output "
            f"(max|da|={conv_a:.3g}, max|db|={conv_b:.3g}). b = -d/a1 is wrong or mirt "
            "changed its parameterisation; do not trust anything downstream."
        )
    print(f"{dataset}/mirt: b = -d/a1 reproduces mirt's IRTpars to {max(conv_a, conv_b):.3g}")

    est = Estimates(
        name="mirt",
        dataset=dataset,
        item_ids=item_ids,
        a=a,
        b=b,
        respondent_ids=[row["respondent_id"] for row in thetas],
        theta=np.array([_float(row["theta"]) for row in thetas]),
        parameterisation="fitted as a1*theta + d; converted here with b = -d/a1",
        # Measured, not assumed. mirt's default 2PL places no positivity
        # constraint on the slope, and on these datasets it returns negative
        # ones: 2 of 59 items on dense, 22 of 183 on sparse. Only py-irt
        # constrains `a` positive, by drawing it from a LogNormal.
        a_positive=False,
        theta_metric="theta ~ N(0, 1) fixed quadrature distribution, 61 equally spaced nodes on [-6, 6]",
        detail={
            "estimator": "marginal maximum likelihood, EM over a fixed normal quadrature grid",
            "n_negative_slopes": int((a1 < 0).sum()),
            "theta_estimator": "EAP",
            "slope_intercept_conversion_max_abs_err": max(conv_a, conv_b),
            "a1": [float(x) for x in a1],
            "d": [float(x) for x in d],
            "se_a1": [_float(row["se_a1"]) for row in items],
            "se_d": [_float(row["se_d"]) for row in items],
            **meta,
        },
    )
    est.to_json(estimates_path(dataset, est.name))
    return est


def main() -> None:
    rscript = find_rscript()
    if rscript is None:
        print(
            "mirt is not available: no Rscript on PATH, at $IRTCHECK_RSCRIPT, or in\n"
            f"  {MAMBA_ENV}\n"
            "with the mirt package importable. Run crosscheck/setup.sh, which installs\n"
            "R and mirt from conda-forge into that prefix without needing root.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    print(f"using {rscript}")
    for dataset in DATASETS:
        run(dataset, rscript)


if __name__ == "__main__":
    main()
