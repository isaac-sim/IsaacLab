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
import time
from pathlib import Path

import pytest

# Helpers

# Set to True when pytest runs with -s / --capture=no so that run_cmd()
# streams subprocess output in real time by default.
_STREAM_OUTPUT: bool = False

_DIM = "\033[2m"
_CYAN_BRIGHT = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"

_test_index: dict[str, int] = {}
_test_total: int = 0


def _find_isaaclab_root() -> Path:
    """Walk up from this file to find the repo root (contains isaaclab.sh)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "isaaclab.sh").exists():
            return parent
    raise FileNotFoundError("Could not locate IsaacLab repository root (no isaaclab.sh found)")


def _has_command(name: str) -> bool:
    """Return True if *name* is available on PATH."""
    return shutil.which(name) is not None


# Fixtures


@pytest.fixture(scope="session")
def isaaclab_root() -> Path:
    """Resolved absolute path to the IsaacLab repository root."""
    return _find_isaaclab_root()


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
    global _STREAM_OUTPUT
    config.addinivalue_line("markers", "regression: bug-regression tests")
    config.addinivalue_line("markers", "gpu: tests that require a GPU")
    config.addinivalue_line("markers", "docker_only: tests that only run inside Docker")
    config.addinivalue_line("markers", "needs_network: tests that require network access")
    config.addinivalue_line("markers", "slow: tests that take a long time")

    # Enable real-time output when pytest capture is disabled (-s)
    capture = config.getoption("capture", default="fd")
    _STREAM_OUTPUT = capture == "no"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print a newline after the PASSED/FAILED/SKIPPED result."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        sys.stdout.write("\n")
        sys.stdout.flush()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Print the test's docstring as a human-readable explanation before running."""
    idx = _test_index.get(item.nodeid, 0)
    total = _test_total
    prefix = f"[TEST {idx}/{total}]: " if total else ""
    doc = item.function.__doc__
    if doc:
        first_line = doc.strip().split("\n")[0].strip()
        sys.stdout.write(f"\n{_CYAN_BRIGHT}{prefix}{first_line}{_RESET}\n")
        sys.stdout.flush()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    global _test_total
    is_windows = platform.system() == "Windows"
    has_uv = _has_command("uv")
    in_docker = Path("/.dockerenv").exists()

    # Build index for numbering
    _test_total = len(items)
    for i, item in enumerate(items, 1):
        _test_index[item.nodeid] = i

    # Print collected tests
    if items:
        print(f"\nCollected {len(items)} tests:")
        for i, item in enumerate(items, 1):
            print(f"  {i}/{_test_total}: {item.nodeid}")
        print()

    for item in items:
        # Auto-skip docker_only tests when not in Docker
        if "docker_only" in item.keywords and not in_docker:
            item.add_marker(pytest.mark.skip(reason="docker_only: not running inside Docker"))

        # Auto-skip tests requiring uv when uv is not installed
        if "uv" in item.nodeid and not has_uv:
            item.add_marker(pytest.mark.skip(reason="uv not available on PATH"))

        # Auto-skip shell CLI tests on Windows
        if "cli_install" in item.nodeid and is_windows:
            item.add_marker(pytest.mark.skip(reason="isaaclab.sh CLI tests are Linux-only"))


# Subprocess helper available to all tests


def run_cmd(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    check: bool = True,
    stream: bool | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, merging *env* into the current environment.

    Args:
        stream: When True, stream stdout/stderr to the console in
            real time instead of capturing them.  Defaults to True when
            pytest is invoked with ``-s`` (``--capture=no``).

    Returns the CompletedProcess; raises CalledProcessError when *check* is
    True and return code != 0.
    """
    if stream is None:
        stream = _STREAM_OUTPUT
    merged_env = {**os.environ, **(env or {})}
    cmd_str = " ".join(str(a) for a in args)
    if stream:
        sys.stdout.write(f"{_MAGENTA}[COMMAND] {cmd_str}{_RESET}\n")
        sys.stdout.flush()
        # Stream output to console in real time.
        t0 = time.monotonic()
        proc = subprocess.Popen(
            [str(a) for a in args],
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []
        try:
            for line in proc.stdout:
                lines.append(line)
                sys.stdout.write(f"{_DIM}{line}{_RESET}")
                sys.stdout.flush()
        except Exception:
            proc.kill()
            raise
        proc.wait(timeout=timeout)
        elapsed = time.monotonic() - t0
        sys.stdout.write(f"{_MAGENTA}[{elapsed:.1f}s]{_RESET}\n")
        sys.stdout.flush()
        result = subprocess.CompletedProcess(
            args=proc.args,
            returncode=proc.returncode,
            stdout="".join(lines),
            stderr="",
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        return result
    return subprocess.run(
        [str(a) for a in args],
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )
