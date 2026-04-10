# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for Isaac Lab installation CI."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_DIM = "\033[2m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"

# Controls whether run_cmd() streams output by default.
# Set to True by conftest.py when pytest runs with -s / --capture=no.
stream_output: bool = False


def find_isaaclab_root() -> Path:
    """Walk up from this file to find the repo root (contains isaaclab.sh)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "isaaclab.sh").exists():
            return parent
    raise FileNotFoundError("Could not locate IsaacLab repository root (no isaaclab.sh found)")


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
        args: Command and arguments to run.
        cwd: Working directory for the subprocess.
        env: Extra environment variables merged into the current environment.
        timeout: Timeout in seconds.
        check: Raise CalledProcessError on non-zero exit.
        stream: When True, stream stdout/stderr to the console in
            real time instead of capturing them.  Defaults to True when
            pytest is invoked with ``-s`` (``--capture=no``).

    Returns:
        The CompletedProcess; raises CalledProcessError when *check* is
        True and return code != 0.
    """
    if stream is None:
        stream = stream_output
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
