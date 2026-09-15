"""Rewrite `__version__` in src/irtcheck/__init__.py — for TestPyPI rehearsals only.

The one line hatch reads the version from (see pyproject.toml). Fails rather
than writing if the line is not there exactly once, so a refactor of
`__init__.py` breaks the rehearsal loudly instead of building the old version.

    python .github/release/set_version.py 0.1.0.dev123451
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[2] / "src" / "irtcheck" / "__init__.py"
PATTERN = re.compile(r'^__version__ = "[^"]+"$', re.M)


def set_version(version: str, init: Path = INIT) -> None:
    text = init.read_text()
    rewritten, count = PATTERN.subn(f'__version__ = "{version}"', text)
    if count != 1:
        sys.exit(f"{init} must assign __version__ exactly once; found {count}")
    init.write_text(rewritten)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+(\.\d+)*(\.dev\d+)?", sys.argv[1]):
        sys.exit("usage: set_version.py <PEP 440 version, e.g. 0.1.0.dev123451>")
    set_version(sys.argv[1])
    print(f"__version__ = {sys.argv[1]!r}")
