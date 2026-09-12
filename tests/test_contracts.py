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
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer.main
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


def command_flags() -> dict[str, dict[str, tuple[str, bool, str]]]:
    """{command: {flag: (type name, is a boolean switch, help text)}}, from click.

    This walks the real, parsed CLI surface. The version of this test written
    in wave 0 read `command.callback.__annotations__` and looked for
    `__metadata__` on each value, and it never found a single flag — so it
    passed vacuously through the whole of wave 1, which is exactly the period
    it existed to police. Two independent reasons it could not work:

      - `cli.py` has `from __future__ import annotations`, so every annotation
        is a *string* and has no `__metadata__` at all. Resolving them needs
        `typing.get_type_hints(..., include_extras=True)`.
      - even resolved, typer's `OptionInfo.param_decls` is usually empty here:
        in the Annotated style a lone positional argument to `typer.Option` is
        taken as `default`, and the flag name is derived from the parameter
        name instead. `typer.Option("-o", "--output")` records `--output`;
        `typer.Option("--respondent-key")` records nothing.

    Asking click for the command it actually built sidesteps both.
    """
    group = typer.main.get_command(app)
    out: dict[str, dict[str, tuple[str, bool, str]]] = {}
    for name, command in group.commands.items():
        out[name] = {
            opt: (
                getattr(param.type, "name", str(param.type)),
                bool(getattr(param, "is_flag", False) or getattr(param, "secondary_opts", [])),
                (param.help or "").strip(),
            )
            for param in command.params
            for opt in param.opts
            if opt.startswith("--")
        }
    return out


def test_the_flag_inventory_is_not_empty():
    """Guards the guard: the test below is only meaningful if it sees flags.

    Its predecessor silently saw none for the whole of wave 1. Any future
    change to how the surface is introspected fails here, loudly, instead of
    quietly reducing the next test to a no-op.
    """
    flags = command_flags()
    assert set(flags) == EXPECTED_COMMANDS
    assert all(len(v) >= 3 for v in flags.values()), flags
    assert "--respondent-key" in flags["fit"]


def test_no_two_commands_disagree_about_a_flag():
    """`--seed` may appear on several commands, but it must mean the same thing.
    Two agents giving one flag two meanings is invisible in review of either
    branch alone.

    The check is on *type and arity*, not on help text. Once this test started
    working it immediately flagged four cases — `--output`, `--seed`,
    `--epochs`, `--json` — and all four are legitimate: same role, different
    object. `select --output` writes item ids, `fit --output` writes an
    artifact; `validate --epochs` is "SVI steps per holdout refit" because
    there are several fits. Demanding identical prose would force those to be
    reworded worse, so the rule is the one that catches a real collision: the
    same flag taking an int on one command and a path on another, or being a
    switch here and a value there. A wording difference is specialisation; a
    type difference means two agents meant different things.
    """
    signatures: dict[str, dict[str, tuple[str, bool]]] = {}
    for command, flags in command_flags().items():
        for flag, (type_name, is_switch, _help) in flags.items():
            signatures.setdefault(flag, {})[command] = (type_name, is_switch)

    for flag, uses in signatures.items():
        distinct = set(uses.values())
        assert len(distinct) == 1, (
            f"{flag} has a different type or arity across commands: "
            + "; ".join(
                f"{c}={t}{' (switch)' if s else ''}" for c, (t, s) in sorted(uses.items())
            )
        )


def test_every_flag_is_documented():
    """A flag with no help text is undiscoverable in `--help`, which is the
    same failure as shipping it as an environment variable."""
    undocumented = [
        f"{command} {flag}"
        for command, flags in command_flags().items()
        for flag, (_type, _switch, help_text) in flags.items()
        if not help_text and flag != "--help"
    ]
    assert not undocumented, f"flags with no help text: {undocumented}"


def test_unimplemented_commands_name_their_owning_brief():
    """A stub must say who owns it, so a wave-1 agent picking up work knows
    whether the gap is theirs to fill.

    Written against whichever commands are *still* stubs rather than against
    one named command. Each wave-1 brief replaces exactly one of these stubs,
    and a test that named `report` would have to be edited by the report agent,
    then by the fit agent, then by select-validate — three concurrent branches
    all touching this file, which is the one file in the repo that exists to
    catch what happens when concurrent branches collide.

    An implemented `run` raises TypeError for the arguments cli.py would have
    passed it; that is not a stub message and nothing is asserted about it.
    """
    for name in sorted(EXPECTED_COMMANDS):
        module = importlib.import_module(f"irtcheck.commands.{name}")
        try:
            module.run()
        except NotImplementedError as exc:
            assert re.search(r"wave\d+/[a-z-]+", str(exc)), (
                f"the `{name}` stub does not name the brief that owns it: {exc}"
            )
        except TypeError:
            pass  # implemented, and its signature is checked by its own tests


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
