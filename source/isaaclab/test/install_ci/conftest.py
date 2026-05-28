# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures and configuration for installation CI tests."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``utils`` (sibling of this conftest) importable from tests living in
# subdirectories (``cli/``, ``misc/``).  Required because ``pytest.ini`` uses
# ``--import-mode=importlib``, which does NOT add the conftest's directory to
# ``sys.path``.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import utils as _utils  # noqa: E402
from utils import find_isaaclab_root, run_cmd  # noqa: E402, F401 – re-exported for tests

_CYAN_BRIGHT = "\033[96m"
_RESET = "\033[0m"

_EXECUTION_ENVIRONMENT_KEY = pytest.StashKey[_utils.ExecutionEnvironment]()


# Fixtures


@pytest.fixture(scope="session")
def isaaclab_root() -> Path:
    """Resolved absolute path to the IsaacLab repository root."""
    return find_isaaclab_root()


@pytest.fixture
def tmp_venv(tmp_path: Path):
    """Create a temporary Python virtual-environment and tear it down after the test.

    Yields a dict with:
        ``path``  – Path to the venv directory
        ``python`` – Path to the venv's python executable
        ``pip``    – Path to the venv's pip executable
    """
    venv_dir = tmp_path / "venv"
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    # Upgrade pip inside the venv to avoid old-pip issues
    subprocess.check_call([str(pip_exe), "install", "--upgrade", "pip"], timeout=120)

    yield {"path": venv_dir, "python": python_exe, "pip": pip_exe}

    # Cleanup is handled by tmp_path (pytest removes it automatically)


@pytest.fixture(scope="session")
def wheel_path() -> Path | None:
    """Path to a pre-built isaaclab wheel, or None.

    Set the ``ISAACLAB_WHEEL`` environment variable to the wheel file path
    before running tests.
    """
    value = os.environ.get("ISAACLAB_WHEEL")
    if value:
        p = Path(value).resolve()
        if not p.exists():
            pytest.fail(f"ISAACLAB_WHEEL points to non-existent file: {p}")
        return p
    return None


# Markers


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--noskip",
        "--force",
        "--dontskip",
        action="store_true",
        default=False,
        dest="noskip",
        help="Strip all @pytest.mark.skip markers for this run (does not affect skipif). Alias: --force.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "bug: bug-regression tests (use bug id as argument)")
    config.addinivalue_line("markers", "gpu: tests that require a GPU")
    config.addinivalue_line("markers", "docker: tests that only run inside Docker")
    config.addinivalue_line("markers", "native: tests that only run natively (not in Docker)")
    config.addinivalue_line("markers", "slow: tests that take a long time")
    config.addinivalue_line("markers", "uv: tests that require the uv package manager")
    config.addinivalue_line("markers", "conda: tests that require the conda package manager")
    config.addinivalue_line("markers", "timeout: per-test timeout in seconds")

    try:
        config.stash[_EXECUTION_ENVIRONMENT_KEY] = _utils.detect_execution_environment()
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc

    # Enable real-time output when pytest capture is disabled (-s)
    capture = config.getoption("capture", default="fd")
    _utils.stream_output = capture == "no"


def pytest_report_header(config: pytest.Config) -> str:
    """Show the detected install_ci execution environment in the test header."""
    return f"install_ci execution environment: {config.stash[_EXECUTION_ENVIRONMENT_KEY]}"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print a newline after the PASSED/FAILED/SKIPPED result."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        sys.stdout.write("\n")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Map dynamic bug markers and deselect items with mismatched env markers.

    This allows filtering by bug ID natively in pytest: `-m "<bug-id>"`
    instead of the (unsupported natively) `-m "bug('<bug-id>')"`.

    Tests whose ``docker``/``native`` marker doesn't match the current
    execution environment are *deselected* (removed from collection) so they
    don't appear in ``--collect-only`` output or skew skipped-count metrics.
    """
    execution_environment = config.stash[_EXECUTION_ENVIRONMENT_KEY]
    known_bugs = set()
    for item in items:
        for mark in item.iter_markers(name="bug"):
            for arg in mark.args:
                if isinstance(arg, str):
                    known_bugs.add(arg)

    for bug in known_bugs:
        config.addinivalue_line("markers", f"{bug}: dynamically generated bug marker")

    remaining: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        for mark in item.iter_markers(name="bug"):
            for arg in mark.args:
                if isinstance(arg, str):
                    item.add_marker(arg)

        marker_names = {mark.name for mark in item.iter_markers()}
        try:
            skip_reason = _utils.get_execution_environment_skip_reason(marker_names, execution_environment)
        except ValueError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc

        if skip_reason:
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        items[:] = remaining
        config.hook.pytest_deselected(items=deselected)

    if config.getoption("--noskip"):
        for item in items:
            item.own_markers = [m for m in item.own_markers if m.name != "skip"]
