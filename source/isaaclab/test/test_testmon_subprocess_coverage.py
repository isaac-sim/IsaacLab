# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for Testmon subprocess dependency tracking.

Spins up an isolated pytest-in-pytest project whose only link to a helper module
is a child ``subprocess``. Verifies that :mod:`testmon_subprocess_coverage`
records the helper as a dependency, so editing it reselects the test.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _find_tools_dir() -> Path:
    """Locate the repository ``tools/`` directory that holds the Testmon helpers."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "testmon_subprocess_coverage.py"
        if candidate.is_file():
            return candidate.parent
    raise RuntimeError("could not locate tools/testmon_subprocess_coverage.py")


_TOOLS_DIR = _find_tools_dir()

_HELPER_MODULE = "subprocess_only_helper"

# A test whose sole link to the helper is a child ``python -c "..."`` process.
# The helper is never imported in the pytest process, so only subprocess
# coverage can register it as a Testmon dependency.
_TEST_BODY = """\
import os
import subprocess
import sys


def test_calls_helper_in_subprocess():
    # Record that this test actually executed (the harness counts these lines).
    with open(os.environ["SUBPROC_DEP_RUNS"], "a", encoding="utf-8") as handle:
        handle.write("run\\n")

    result = subprocess.run(
        [sys.executable, "-c", "import {module}; assert {module}.contribution() >= 0"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
"""

_CONFTEST_BODY = """\
from testmon_subprocess_coverage import (  # noqa: F401
    pytest_runtest_makereport,
    pytest_runtest_setup,
    pytest_runtest_teardown,
    pytest_sessionfinish,
    pytest_sessionstart,
)
"""


def _dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("testmon", "coverage"))


def _write_helper(project: Path, value: int) -> None:
    (project / f"{_HELPER_MODULE}.py").write_text(
        f"def contribution():\n    return {value}\n",
        encoding="utf-8",
    )


def _clean_env(project: Path, datafile: Path, runs_file: Path) -> dict[str, str]:
    """Environment for a nested pytest run, scrubbed of inherited coverage state."""
    env = os.environ.copy()
    # Drop any coverage/testmon state leaking in from the outer pytest session so
    # the child session manages its own coverage lifecycle.
    for key in ("COVERAGE_PROCESS_START", "COVERAGE_CONTEXT", "COVERAGE_FILE"):
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join([str(_TOOLS_DIR), str(project), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["TESTMON_DATAFILE"] = str(datafile)
    env["SUBPROC_DEP_RUNS"] = str(runs_file)
    return env


def _run_pytest(project: Path, env: dict[str, str], *testmon_args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--testmon",
        *testmon_args,
        "-q",
        str(project),
    ]
    return subprocess.run(cmd, cwd=project, env=env, capture_output=True, text=True, timeout=300)


def _run_count(runs_file: Path) -> int:
    if not runs_file.is_file():
        return 0
    return len([line for line in runs_file.read_text(encoding="utf-8").splitlines() if line.strip()])


def test_testmon_args_are_forwarded_to_per_file_pytest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The package-test runner applies Testmon selection to the pytest process that collects real tests."""
    module_name = "isaaclab_package_test_runner"
    spec = importlib.util.spec_from_file_location(module_name, _TOOLS_DIR / "conftest.py")
    assert spec and spec.loader
    test_runner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, test_runner)
    spec.loader.exec_module(test_runner)

    commands: list[list[str]] = []

    def capture_command(cmd, *args, **kwargs):
        commands.append(cmd)
        return 0, b"", b"", "", 0.0, ""

    monkeypatch.setattr(test_runner, "capture_test_output_with_timeout", capture_command)
    monkeypatch.setattr(test_runner, "_get_diagnostics", lambda *args: "")
    context = test_runner._PassContext(
        test_file="test_selection_probe.py",
        file_name="test_selection_probe.py",
        workspace_root=str(tmp_path),
        isaacsim_ci=False,
        timeout=1,
        startup_deadline=1,
        env={"TESTMON_MODE": "select"},
        pytest_targets=["test_selection_probe.py"],
    )

    test_runner._run_one_pass(context, k_expr=None, suffix="")

    assert commands
    assert "--testmon" in commands[0]
    assert "--testmon-forceselect" in commands[0]


@pytest.mark.skipif(not _dependencies_available(), reason="pytest-testmon and coverage are required")
def test_editing_subprocess_only_dependency_reselects_test(tmp_path: Path) -> None:
    """Testmon reselects a test when a dependency reached only via a subprocess changes."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / f"test_{_HELPER_MODULE}.py").write_text(_TEST_BODY.format(module=_HELPER_MODULE), encoding="utf-8")
    (project / "conftest.py").write_text(_CONFTEST_BODY, encoding="utf-8")
    _write_helper(project, value=1)

    datafile = project / ".testmondata"
    runs_file = project / "runs.txt"
    env = _clean_env(project, datafile, runs_file)

    # 1. Prime Testmon: run everything and record dependencies (incl. subprocess).
    collect = _run_pytest(project, env, "--testmon-noselect")
    assert collect.returncode == 0, f"collect run failed:\n{collect.stdout}\n{collect.stderr}"
    assert datafile.is_file(), "Testmon did not create its data file during collection"
    assert _run_count(runs_file) == 1, f"expected the test to run once during collection:\n{collect.stdout}"

    # 2. Nothing changed -> Testmon must deselect the test (it does not run again).
    unchanged = _run_pytest(project, env)
    assert _run_count(runs_file) == 1, (
        "Testmon reran the test even though nothing changed; deselection is not working.\n"
        f"{unchanged.stdout}\n{unchanged.stderr}"
    )

    # 3. Edit the helper that is only reachable through the subprocess.
    _write_helper(project, value=2)

    # 4. Testmon must notice the subprocess-only dependency changed and reselect.
    changed = _run_pytest(project, env)
    assert _run_count(runs_file) == 2, (
        "Testmon did not reselect the test after its subprocess-only dependency changed; "
        "subprocess coverage is not being attributed to the test.\n"
        f"{changed.stdout}\n{changed.stderr}"
    )
