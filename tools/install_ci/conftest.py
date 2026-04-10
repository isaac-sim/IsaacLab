# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures and configuration for installation CI tests."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import utils as _utils
from utils import find_isaaclab_root, run_cmd  # noqa: F401 – re-exported for tests

_CYAN_BRIGHT = "\033[96m"
_RESET = "\033[0m"

_test_index: dict[str, int] = {}
_test_total: int = 0
_index_built: bool = False


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regression: bug-regression tests")
    config.addinivalue_line("markers", "gpu: tests that require a GPU")
    config.addinivalue_line("markers", "docker_only: tests that only run inside Docker")
    config.addinivalue_line("markers", "needs_network: tests that require network access")
    config.addinivalue_line("markers", "slow: tests that take a long time")
    config.addinivalue_line("markers", "uv: tests that require the uv package manager")

    # Enable real-time output when pytest capture is disabled (-s)
    capture = config.getoption("capture", default="fd")
    _utils.stream_output = capture == "no"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print a newline after the PASSED/FAILED/SKIPPED result."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        sys.stdout.write("\n")
        sys.stdout.flush()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Print the test's docstring as a human-readable explanation before running."""
    global _index_built, _test_total, _test_index
    if not _index_built:
        # Build index from the final selected items (after -k/-m deselection)
        selected = item.session.items
        _test_total = len(selected)
        _test_index.clear()
        for i, it in enumerate(selected, 1):
            _test_index[it.nodeid] = i
        _index_built = True

    idx = _test_index.get(item.nodeid, 0)
    total = _test_total
    prefix = f"[TEST {idx}/{total}]: " if total else ""
    doc = item.function.__doc__
    if doc:
        first_line = doc.strip().split("\n")[0].strip()
        sys.stdout.write(f"\n{_CYAN_BRIGHT}{prefix}{first_line}{_RESET}\n")
        sys.stdout.flush()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    is_windows = platform.system() == "Windows"
    has_uv = shutil.which("uv") is not None
    in_docker = Path("/.dockerenv").exists()

    # Print collected tests
    if items:
        print(f"\nCollected {len(items)} tests:")
        for i, item in enumerate(items, 1):
            print(f"  {i}/{len(items)}: {item.nodeid}")
        print()

    for item in items:
        # Auto-skip docker_only tests when not in Docker
        if "docker_only" in item.keywords and not in_docker:
            item.add_marker(pytest.mark.skip(reason="docker_only: not running inside Docker"))

        # Auto-skip tests requiring uv when uv is not installed
        if item.get_closest_marker("uv") and not has_uv:
            item.add_marker(pytest.mark.skip(reason="uv not available on PATH"))

        # Auto-skip shell CLI tests on Windows
        if "cli_install" in item.nodeid and is_windows:
            item.add_marker(pytest.mark.skip(reason="isaaclab.sh CLI tests are Linux-only"))
