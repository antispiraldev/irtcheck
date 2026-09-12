#!/usr/bin/env bash
# The whole cross-check, in order.
#
#   bash crosscheck/run_all.sh
#
# Run crosscheck/setup.sh first if you want the two external references; without
# them the scipy reference and the grid-posterior check still run, and
# compare.py lists the missing ones as MISSING rather than quietly leaving them
# out. About 15 minutes end to end with everything present.
#
# Each step writes into crosscheck/work/ and is independently re-runnable, so a
# failure in one reference does not cost the others.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${IRTCHECK_PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO"

if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY. Set IRTCHECK_PYTHON, or create the project venv:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

step() {
  echo
  echo "############################################################"
  echo "# $1"
  echo "############################################################"
}

failed=()
run() {
  local label="$1"; shift
  step "$label"
  if ! "$@"; then
    echo "!! $label FAILED (continuing; compare.py will report what is missing)"
    failed+=("$label")
  fi
}

step "0. the convention logic and the vacuity guards"
"$PY" -m pytest "$HERE/test_conventions.py" -q || failed+=("tests")

run "1. generate the shared datasets"            "$PY" -m crosscheck.export_inputs
run "2. irtcheck's own 2PL (hierarchical+vague)" "$PY" -m crosscheck.run_ours
run "3. scipy marginal ML (no external deps)"    "$PY" -m crosscheck.run_mml
run "4. R / mirt"                                "$PY" -m crosscheck.run_mirt
run "5. py-irt 0.7.1 on Python 3.11"             "$PY" -m crosscheck.run_pyirt
run "6. SVI intervals vs a grid posterior"       "$PY" -m crosscheck.run_gridpost

step "7. reconcile conventions and report"
"$PY" -m crosscheck.compare
compare_status=$?

echo
if [ ${#failed[@]} -gt 0 ]; then
  echo "steps that failed: ${failed[*]}"
fi
echo "results are in $HERE/results/"
exit $compare_status
