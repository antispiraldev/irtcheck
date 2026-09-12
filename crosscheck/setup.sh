#!/usr/bin/env bash
# Install both reference implementations. Needs no root.
#
#   bash crosscheck/setup.sh
#
# Everything lands under $IRTCHECK_CROSSCHECK_HOME (default
# ~/.cache/irtcheck-crosscheck), outside the repository, because neither
# reference belongs in the project environment:
#
#   pyirt311/   a standalone CPython 3.11 venv with py-irt==0.7.1. py-irt
#               declares Requires-Python >=3.9,<3.12 and CANNOT be installed
#               alongside this project — see check_pyirt_resolution.sh for what
#               happens if you try. The interpreter comes from `uv python
#               install 3.11`, which downloads a relocatable build into the
#               user's own directory; no system package, no root.
#
#   r-mirt/     R 4.3 plus the mirt package, from conda-forge via micromamba.
#               `apt install r-base r-cran-mirt` needs root; micromamba is a
#               single static binary that does not. About 600 MB unpacked.
#
# Both steps are skipped if already present, so this is safe to re-run.
set -euo pipefail

HOME_DIR="${IRTCHECK_CROSSCHECK_HOME:-$HOME/.cache/irtcheck-crosscheck}"
PYIRT_VENV="$HOME_DIR/pyirt311"
R_PREFIX="$HOME_DIR/r-mirt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HOME_DIR"

echo "installing crosscheck references into $HOME_DIR"

# --- py-irt on Python 3.11 ---------------------------------------------------
if [ -x "$PYIRT_VENV/bin/python" ] && "$PYIRT_VENV/bin/python" -c "import py_irt" 2>/dev/null; then
  echo "[1/2] py-irt: already installed at $PYIRT_VENV"
else
  echo "[1/2] py-irt: installing"
  PY311="$(command -v python3.11 || true)"
  if [ -z "$PY311" ]; then
    if ! command -v uv >/dev/null; then
      echo "  need either python3.11 on PATH or uv (https://docs.astral.sh/uv/)." >&2
      echo "  With uv: uv python install 3.11. Without root and without uv, install" >&2
      echo "  a 3.11 by any means and re-run; nothing else about this step is special." >&2
      exit 1
    fi
    uv python install 3.11
    PY311="$(uv python find 3.11)"
  fi
  echo "  using $PY311 ($("$PY311" -V 2>&1))"
  "$PY311" -m venv "$PYIRT_VENV"
  "$PYIRT_VENV/bin/pip" install -q --upgrade pip
  "$PYIRT_VENV/bin/pip" install -q -r "$HERE/requirements-pyirt.txt"
  "$PYIRT_VENV/bin/python" -c "
import py_irt, sys
from importlib.metadata import version
print(f'  installed py-irt {version(\"py-irt\")} on Python {sys.version.split()[0]}')
"
fi

# --- R and mirt --------------------------------------------------------------
if [ -x "$R_PREFIX/bin/Rscript" ] && \
   "$R_PREFIX/bin/Rscript" -e 'suppressPackageStartupMessages(library(mirt))' 2>/dev/null; then
  echo "[2/2] mirt: already installed at $R_PREFIX"
else
  echo "[2/2] mirt: installing R + mirt from conda-forge (a few hundred MB)"
  MAMBA="$HOME_DIR/bin/micromamba"
  if [ ! -x "$MAMBA" ]; then
    mkdir -p "$HOME_DIR/bin"
    echo "  fetching micromamba"
    curl -sSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
      | tar -xj -C "$HOME_DIR" bin/micromamba
  fi
  MAMBA_ROOT_PREFIX="$HOME_DIR/mamba" "$MAMBA" create -y -p "$R_PREFIX" \
    -c conda-forge r-base=4.3 r-mirt
  "$R_PREFIX/bin/Rscript" -e '
    cat("  installed", R.version.string, "with mirt",
        as.character(packageVersion("mirt")), "\n")' 2>/dev/null \
    | grep installed
fi

echo
echo "done. Now run:"
echo "  bash crosscheck/run_all.sh"
echo
echo "run_mirt.py and run_pyirt.py find these prefixes on their own. To point"
echo "them somewhere else, set IRTCHECK_RSCRIPT=/path/to/Rscript, or move the"
echo "3.11 venv and set IRTCHECK_CROSSCHECK_HOME."
