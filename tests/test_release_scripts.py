"""The release workflow's decisions, tested without a runner.

`.github/release/decide.py` is the gate between a tag and an upload PyPI will
never let us take back, so every branch of it is pinned here: a matching tag
publishes, a mismatched one refuses before anything is built, a dispatch to
PyPI from a branch refuses, and a TestPyPI rehearsal gets a dev version that
sorts *before* the release it rehearses.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from irtcheck import __version__

RELEASE = Path(__file__).resolve().parents[1] / ".github" / "release"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, RELEASE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decide = load("decide")
set_version = load("set_version")


def env(**overrides: str) -> dict[str, str]:
    base = {"EVENT": "push", "GIT_REF": "v1.2.3", "REF_TYPE": "tag", "RUN_ID": "987", "RUN_ATTEMPT": "2"}
    return base | overrides


def test_a_matching_tag_publishes_to_pypi():
    assert decide.decide(env(), "1.2.3") == ("1.2.3", "pypi")


def test_a_mismatched_tag_refuses_and_names_both_versions():
    with pytest.raises(SystemExit) as refused:
        decide.decide(env(GIT_REF="v1.2.4"), "1.2.3")
    assert "1.2.4" in str(refused.value) and "1.2.3" in str(refused.value)


def test_a_tag_without_the_v_prefix_refuses():
    with pytest.raises(SystemExit, match="does not start with 'v'"):
        decide.decide(env(GIT_REF="1.2.3"), "1.2.3")


def test_dispatching_to_pypi_from_a_branch_refuses():
    with pytest.raises(SystemExit, match="needs a tag"):
        decide.decide(env(EVENT="workflow_dispatch", TARGET="pypi", GIT_REF="master", REF_TYPE="branch"), "1.2.3")


def test_dispatching_to_pypi_from_the_right_tag_publishes():
    assert decide.decide(env(EVENT="workflow_dispatch", TARGET="pypi"), "1.2.3") == ("1.2.3", "pypi")


def test_a_rehearsal_gets_a_dev_version_from_any_branch():
    version, target = decide.decide(
        env(EVENT="workflow_dispatch", TARGET="testpypi", GIT_REF="release/prep", REF_TYPE="branch"), "1.2.3"
    )
    assert (version, target) == ("1.2.3.dev9872", "testpypi")


def test_a_rehearsal_sorts_before_the_release_it_rehearses():
    """A dev version that sorted *after* 1.2.3 could shadow it for anyone
    installing from an index that carries both."""
    pytest.importorskip("packaging")
    from packaging.version import Version

    rehearsal, _ = decide.decide(env(EVENT="workflow_dispatch", TARGET="testpypi"), "1.2.3")
    assert Version(rehearsal) < Version("1.2.3") and Version(rehearsal).is_devrelease


def test_the_packaged_version_is_read_from_the_module():
    assert decide.packaged_version() == __version__


def test_set_version_rewrites_exactly_the_version_line(tmp_path):
    init = tmp_path / "__init__.py"
    init.write_text('"""doc."""\n\n__version__ = "0.1.0"\n\n__all__ = ["__version__"]\n')
    set_version.set_version("0.1.0.dev55", init)
    assert init.read_text() == '"""doc."""\n\n__version__ = "0.1.0.dev55"\n\n__all__ = ["__version__"]\n'


def test_set_version_refuses_when_the_line_is_missing(tmp_path):
    init = tmp_path / "__init__.py"
    init.write_text("from ._version import __version__\n")
    with pytest.raises(SystemExit, match="exactly once"):
        set_version.set_version("0.1.0.dev55", init)


def test_decide_writes_github_outputs(tmp_path):
    output = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(RELEASE / "decide.py")],
        env=os.environ | env(GIT_REF=f"v{__version__}") | {"GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"version={__version__}\ntarget=pypi\n"
