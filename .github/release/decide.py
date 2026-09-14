"""Decide what the release workflow builds and where it publishes it.

Writes `version` and `target` to $GITHUB_OUTPUT, or exits non-zero with the
reason. Kept as a file rather than inline YAML so it can be run, and tested,
without a runner: `tests/test_release_scripts.py` does both.

    push of tag vX.Y.Z             -> target pypi, version X.Y.Z; tag must match
    dispatch, target pypi          -> same, and the ref must be a tag
    dispatch, target testpypi      -> version X.Y.Z.dev<run id><attempt>, any ref
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[2] / "src" / "irtcheck" / "__init__.py"


def packaged_version(init: Path = INIT) -> str:
    found = re.findall(r'^__version__ = "([^"]+)"$', init.read_text(), re.M)
    if len(found) != 1:
        sys.exit(f"{init} must assign __version__ exactly once; found {len(found)}")
    return found[0]


def decide(env: dict[str, str], packaged: str) -> tuple[str, str]:
    event = env["EVENT"]
    target = "pypi" if event == "push" else env.get("TARGET") or "testpypi"

    if target == "testpypi":
        # PEP 440 dev releases sort before the release they precede, so a
        # rehearsal of 0.2.0 can never be mistaken for, or shadow, 0.2.0 itself.
        return f"{packaged}.dev{env['RUN_ID']}{env['RUN_ATTEMPT']}", target

    if target != "pypi":
        sys.exit(f"unknown target {target!r}")
    ref = env["GIT_REF"]
    if env.get("REF_TYPE") != "tag":
        sys.exit(
            f"publishing to PyPI needs a tag, and this run is on the {env.get('REF_TYPE')} "
            f"{ref!r}. Choose the tag under 'Use workflow from', or push one."
        )
    if not ref.startswith("v"):
        sys.exit(f"tag {ref!r} does not start with 'v'")
    if ref[1:] != packaged:
        sys.exit(
            f"tag {ref!r} says version {ref[1:]!r} but src/irtcheck/__init__.py says "
            f"{packaged!r}. PyPI filenames cannot be reused, so publishing this would "
            f"spend a version number on an unfixable release. Fix __init__.py, or move the tag."
        )
    return packaged, target


def main() -> None:
    version, target = decide(dict(os.environ), packaged_version())
    print(f"target {target}, version {version}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.write(f"version={version}\ntarget={target}\n")


if __name__ == "__main__":
    main()
