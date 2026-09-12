"""The four adapter overrides, as flags rather than environment variables.

Wave 1 shipped these as `IRTCHECK_*` variables because `cli.py` was frozen
while four agents worked against it. A variable that appears in no `--help` is
not an interface a user can find, so they are flags now — and the variables
still work, because someone's scripts may already set them.

These tests exist because the plumbing is indirect: the flag sets the variable
that the adapter reads. Nothing about that is visible in either file alone, so
an assertion that the value actually arrives is the only thing holding the two
ends together.
"""

from __future__ import annotations

import json
import os

import pytest
import typer.main
from typer.testing import CliRunner

from irtcheck.cli import app
from irtcheck.commands.fit import ADAPTER_OVERRIDES

runner = CliRunner()


def command_of(name: str):
    """The built click command for `name` — the real, parsed CLI surface."""
    commands = typer.main.get_command(app).commands
    assert name in commands, f"no command named {name!r}"
    return commands[name]


def flags_of(name: str) -> set[str]:
    return {opt for param in command_of(name).params for opt in param.opts if opt.startswith("--")}


def write_one_response(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"model_id": "m", "item_id": "i", "correct": 1}) + "\n")
    return path


def spy_on(variable: str, seen: dict[str, str | None]):
    """A read_any stand-in that records the variable and stops before fitting."""

    def spy(path, *, fmt=None):
        seen["value"] = os.environ.get(variable)
        raise RuntimeError("stop before fitting")

    return spy


def test_every_override_is_a_flag_on_fit():
    """Each environment variable has a corresponding flag on `fit`.

    Read off the built click command, which is the actual parsed surface. Two
    nearer-looking sources are both wrong: rendered `--help` text is truncated
    to a terminal width CliRunner does not control, and typer's `OptionInfo`
    carries an empty `param_decls` here, because in the Annotated style a lone
    positional argument to `typer.Option` is taken as `default` and the flag
    name is derived from the parameter name instead.
    """
    flags = flags_of("fit")
    for parameter in ADAPTER_OVERRIDES:
        flag = "--" + parameter.replace("_", "-")
        assert flag in flags, f"{flag} is not a flag on `fit` (has: {sorted(flags)})"


def test_each_flag_carries_help_text():
    """A flag with no help is barely more discoverable than the variable it
    replaced, which is the entire reason these were promoted."""
    command = command_of("fit")
    wanted = {"--" + p.replace("_", "-") for p in ADAPTER_OVERRIDES}
    for param in command.params:
        if wanted.intersection(param.opts):
            assert (param.help or "").strip(), f"{param.opts} has no help text"


@pytest.mark.parametrize("parameter,variable", sorted(ADAPTER_OVERRIDES.items()))
def test_the_flag_sets_the_variable_the_adapter_reads(
    parameter, variable, tmp_path, monkeypatch
):
    """The flag's value must reach the adapter, which reads os.environ at read time."""
    monkeypatch.delenv(variable, raising=False)
    seen: dict[str, str | None] = {}
    monkeypatch.setattr("irtcheck.commands.fit.read_any", spy_on(variable, seen))

    flag = "--" + parameter.replace("_", "-")
    runner.invoke(
        app,
        [
            "fit",
            str(write_one_response(tmp_path)),
            "-o",
            str(tmp_path / "o.irt"),
            flag,
            "from-the-flag",
        ],
    )
    assert seen.get("value") == "from-the-flag"


def test_the_flag_wins_over_the_variable(tmp_path, monkeypatch):
    """Both set is not a conflict to resolve at read time — the explicit one wins.

    A user with the variable exported in their shell and the flag on the command
    line means the flag; the reverse would make the flag silently inert.
    """
    monkeypatch.setenv("IRTCHECK_LMEVAL_MODEL_ID", "from-the-environment")
    seen: dict[str, str | None] = {}
    monkeypatch.setattr(
        "irtcheck.commands.fit.read_any", spy_on("IRTCHECK_LMEVAL_MODEL_ID", seen)
    )

    runner.invoke(
        app,
        [
            "fit",
            str(write_one_response(tmp_path)),
            "-o",
            str(tmp_path / "o.irt"),
            "--model-id",
            "from-the-flag",
        ],
    )
    assert seen.get("value") == "from-the-flag"


def test_the_variable_alone_still_works(tmp_path, monkeypatch):
    """Wave 1's interface is not withdrawn — someone's scripts may set it."""
    monkeypatch.setenv("IRTCHECK_LMEVAL_MODEL_ID", "from-the-environment")
    seen: dict[str, str | None] = {}
    monkeypatch.setattr(
        "irtcheck.commands.fit.read_any", spy_on("IRTCHECK_LMEVAL_MODEL_ID", seen)
    )

    runner.invoke(
        app, ["fit", str(write_one_response(tmp_path)), "-o", str(tmp_path / "o.irt")]
    )
    assert seen.get("value") == "from-the-environment"
