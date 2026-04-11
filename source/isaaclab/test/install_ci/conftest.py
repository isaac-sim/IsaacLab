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
import utils as _utils
from utils import find_isaaclab_root, run_cmd  # noqa: F401 – re-exported for tests

_CYAN_BRIGHT = "\033[96m"
_RESET = "\033[0m"


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
    config.addinivalue_line("markers", "bug: bug-regression tests (use bug id as argument)")
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
