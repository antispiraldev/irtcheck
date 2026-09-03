"""The merge-time race check.

Every other test in this suite fails on a single bad branch. These fail on a
combination of branches that were each green on their own — the class of bug
that only appears once two PRs have merged, when nothing is left to report an
error. That is why the CI workflow runs this as its own job, ahead of the
others, with its own name in the PR checks list.

Four agents work a wave concurrently. The specific collisions this catches:

  - two branches both bump SCHEMA_VERSION to the same number, and the second
    merge silently wins;
  - two commands declare the same flag with different meanings;
  - a module grows a top-level `import torch`, quietly breaking the promise
    that `report` and `select` run without it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from irtcheck.artifact import SCHEMA_VERSION, IrtFit
from irtcheck.cli import app
from irtcheck.synth import synthetic_fit

SRC = Path(__file__).resolve().parents[1] / "src" / "irtcheck"
EXPECTED_COMMANDS = {"fit", "report", "select", "validate"}

# Only this package may import torch or pyro. See CLAUDE.md → The lazy torch
# boundary, and the tests-light CI job that enforces it from the outside.
HEAVY = {"torch", "pyro", "pyro_ppl"}
HEAVY_ALLOWED_PREFIX = "fit/"


def source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


# -- schema ------------------------------------------------------------------


def test_schema_version_is_defined_exactly_once():
    """Two branches both bumping this is the #83/#84 shape: individually green,
    broken once combined."""
    hits = []
    for path in source_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.match(r"^SCHEMA_VERSION\s*[:=]", line):
                hits.append(f"{path.relative_to(SRC)}:{number}")
    assert hits == ["artifact.py:1"] or len(hits) == 1, (
        f"SCHEMA_VERSION is assigned in {len(hits)} places: {hits}. "
        "It must have exactly one definition."
    )


def test_schema_version_is_a_positive_int():
    assert isinstance(SCHEMA_VERSION, int) and SCHEMA_VERSION > 0


def test_an_artifact_round_trips(tmp_path):
    """Deliberately duplicated from test_artifact.py. This job is the one that
    must fail loudly on a bad merge, so it does not depend on another file's
    coverage surviving a refactor."""
    fit, _ = synthetic_fit(n_models=5, n_items=25, seed=0)
    first = fit.save(tmp_path / "a.irt")
    second = IrtFit.load(first).save(tmp_path / "b.irt")
    assert first.read_bytes() == second.read_bytes()


# -- CLI surface -------------------------------------------------------------


def test_every_command_is_registered():
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    assert names == EXPECTED_COMMANDS


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_every_command_answers_help(command):
    result = CliRunner().invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_root_help_and_version():
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0 and "irtcheck" in version.output


def test_no_two_commands_disagree_about_a_flag():
    """`--seed` may appear on several commands, but it must mean the same
    thing. Two agents giving one flag two meanings is invisible in review of
    either branch alone."""
    meanings: dict[str, set[str]] = {}
    for command in app.registered_commands:
        name = command.name or command.callback.__name__
        for param in command.callback.__annotations__.values():
            metadata = getattr(param, "__metadata__", ())
            for option in metadata:
                for decl in getattr(option, "param_decls", ()) or ():
                    if decl.startswith("--"):
                        meanings.setdefault(decl, set()).add(
                            f"{name}:{(getattr(option, 'help', '') or '')[:40]}"
                        )
    for flag, uses in meanings.items():
        helps = {u.split(":", 1)[1] for u in uses}
        assert len(helps) == 1, f"{flag} is documented differently across commands: {uses}"


def test_unimplemented_commands_name_their_owning_brief():
    """A stub must say who owns it, so a wave-1 agent picking up work knows
    whether the gap is theirs to fill."""
    from irtcheck.commands import report

    with pytest.raises(NotImplementedError, match="wave1/report"):
        report.run()


# -- the lazy torch boundary -------------------------------------------------


def test_only_the_fit_package_imports_torch_or_pyro():
    offenders = []
    for path in source_files():
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith(HEAVY_ALLOWED_PREFIX):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # A lazy import inside a function is fine; only module scope counts.
            if isinstance(node, ast.Module):
                continue
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in HEAVY:
                    offenders.append(f"{relative}: import {name}")
    assert not offenders, (
        "these modules import torch/pyro at module scope, which breaks the promise "
        "that `report`, `select` and the HTML output run without them:\n  "
        + "\n  ".join(offenders)
    )


def test_importing_the_cli_does_not_pull_in_torch():
    """`irtcheck --help` on a machine without torch has to work. Run in a
    subprocess because the test session may already have imported it."""
    code = (
        "import sys; import irtcheck.cli; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
