#!/usr/bin/env bash
# Reproduce the finding behind CLAUDE.md's "py-irt is not a dependency".
#
#   bash crosscheck/check_pyirt_resolution.sh
#
# The claim is that on Python 3.12+ an unpinned `pip install py-irt` silently
# resolves to 0.1.1 — an abandoned 2020 fork — rather than failing. "Silently"
# is the part that matters: nothing in pip's output says a 5-year-old version
# was substituted for the current one, so a requirements file naming `py-irt`
# would have installed the wrong package and kept working.
#
# This script makes that reproducible rather than remembered. It resolves
# without installing, so it is cheap and leaves nothing behind.
set -uo pipefail

PY312="${PYTHON312:-$(command -v python3.12 || command -v python3)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== interpreter =="
"$PY312" -VV

"$PY312" -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q --upgrade pip

echo
echo "== 1. an unpinned \`pip install py-irt\` on this interpreter resolves to: =="
"$TMP/venv/bin/pip" install --dry-run --quiet --report "$TMP/report.json" py-irt >/dev/null 2>&1
"$TMP/venv/bin/python" - "$TMP/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
for entry in report["install"]:
    meta = entry["metadata"]
    if meta["name"].replace("_", "-").lower() == "py-irt":
        print(f"   py-irt {meta['version']}  (Requires-Python: {meta.get('requires_python')})")
        break
else:
    print("   py-irt did not appear in the resolution at all")
PY

echo
echo "== 2. and the current release cannot be installed here at all: =="
"$TMP/venv/bin/pip" install --dry-run 'py-irt==0.7.1' 2>&1 | grep -E "^ERROR: (Could not find|No matching)" | sed 's/^/   /'

echo
echo "== 3. what 0.1.1's 2PL fit() actually does with its result =="
"$TMP/venv/bin/pip" download -q --no-deps --no-binary :all: -d "$TMP/src" 'py-irt==0.1.1' >/dev/null 2>&1
tar -xzf "$TMP"/src/py-irt-0.1.1.tar.gz -C "$TMP/src"
awk '/def fit\(self, models, items, responses, num_epochs\)/,/def fit_MCMC/' \
  "$TMP/src/py-irt-0.1.1/py_irt/models/two_param_logistic.py" | head -20 | sed 's/^/   /'
echo
echo "   Note the last statement: it builds \`values\`, a list of parameter NAMES,"
echo "   and the function ends. No return, no export, no read of the param store."
echo "   fit() trains the model and discards every estimate. This is the version"
echo "   a 3.12 interpreter gives you, which is why the 2PL in src/irtcheck/fit/"
echo "   is written directly against Pyro instead."
