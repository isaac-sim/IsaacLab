# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch smoke tests for the three teleoperation entry-point scripts.

Each test launches a teleop script end-to-end (spawning Isaac Sim / Kit as a subprocess) and
asserts it starts up past the regression introduced by the render-flag cleanup in #6656:

* ``apply_isaac_rtx_global_settings`` used to crash with
  ``ModuleNotFoundError: No module named 'omni.replicator'`` when the selected Kit experience
  did not preload the replicator extension, and
* ``record_demos.py`` read the removed ``args_cli.enable_cameras`` attribute, raising
  ``AttributeError``.

These follow the Isaac Lab subprocess-integration test convention (see
``scripts/benchmarks/test``) rather than the isaacsim-CI methodology, so they run in the Isaac
Lab CI ``integration`` lane. The teleop scripts run an unbounded interaction loop, so each is
launched under ``PYTHONUNBUFFERED=1`` (the startup prints are otherwise buffered), watched until
a known startup marker appears, then terminated via its process group.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

# Repository root: this file lives at scripts/environments/teleoperation/test/<file>.py
ROOT = Path(__file__).resolve().parents[4]

# Lightweight Franka tasks: keyboard-driven for the interactive scripts, IsaacTeleop for replay.
_KEYBOARD_TASK = "IsaacContrib-Reach-Franka-IK-Rel"
_ISAAC_TELEOP_TASK = "IsaacContrib-Stack-Cube-Franka-IK-Abs"

# Exact #6656 regression signatures that must never appear in a script's output.
_REGRESSION_SIGNATURES = (
    "No module named 'omni.replicator'",
    "has no attribute 'enable_cameras'",
)

# Marker logged by ManagerBasedEnv creation, which runs *after* the RTX-apply site that regressed.
_ENV_CREATED_MARKER = "Base environment:"

# Generous ceiling for a cold Isaac Sim launch + Franka env creation on CI hardware. The watcher
# returns as soon as the marker is seen, so the typical wall time is well under this.
_STARTUP_TIMEOUT_S = 600.0


def _launch_until_marker(argv: list[str], markers: list[str], log_path: Path) -> str:
    """Launch a teleop script and return its output once a marker appears (or it exits / times out).

    The child is started in its own process group and unbuffered so both ``print`` startup markers
    and ``[INFO]`` logs surface promptly; the whole group is terminated before returning.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(ROOT / argv[0]), *argv[1:]]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    try:
        deadline = time.time() + _STARTUP_TIMEOUT_S
        while time.time() < deadline:
            output = log_path.read_text(errors="replace")
            if any(marker in output for marker in markers):
                break
            if any(sig in output for sig in _REGRESSION_SIGNATURES):
                break
            if proc.poll() is not None:
                break
            time.sleep(2.0)
    finally:
        _terminate_group(proc)
    return log_path.read_text(errors="replace")


def _terminate_group(proc: subprocess.Popen) -> None:
    """Terminate the child's whole process group (Isaac Sim spawns helper processes)."""
    if proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue


def _assert_started_cleanly(output: str, markers: list[str], script: str) -> None:
    """Assert the run cleared the #6656 regression and reached a post-startup marker."""
    tail = output[-4000:]
    for sig in _REGRESSION_SIGNATURES:
        assert sig not in output, f"{script}: #6656 regression signature present ({sig!r})\n---\n{tail}"
    assert any(marker in output for marker in markers), (
        f"{script}: did not reach a startup marker {markers} within {_STARTUP_TIMEOUT_S:.0f}s\n---\n{tail}"
    )


def test_teleop_se3_agent_starts(tmp_path):
    """teleop_se3_agent.py reaches the teleoperation loop without the #6656 regression."""
    argv = [
        "scripts/environments/teleoperation/teleop_se3_agent.py",
        "--task",
        _KEYBOARD_TASK,
        "--teleop_device",
        "keyboard",
    ]
    output = _launch_until_marker(argv, ["teleoperation started"], tmp_path / "se3_agent.log")
    _assert_started_cleanly(output, ["teleoperation started"], "teleop_se3_agent.py")


def test_record_demos_starts(tmp_path):
    """record_demos.py reaches the recording loop (and no longer AttributeErrors on enable_cameras)."""
    argv = [
        "scripts/tools/record_demos.py",
        "--task",
        _KEYBOARD_TASK,
        "--teleop_device",
        "keyboard",
        "--dataset_file",
        str(tmp_path / "dataset.hdf5"),
    ]
    output = _launch_until_marker(argv, ["recording started"], tmp_path / "record_demos.log")
    _assert_started_cleanly(output, ["recording started"], "record_demos.py")


def test_teleop_replay_agent_applies_rtx_without_replicator_crash(tmp_path):
    """teleop_replay_agent.py clears the RTX-apply site (create env) without the replicator crash.

    No MCAP fixture ships with the repo, so a non-existent ``--replay_file`` lets the run exercise
    ``_prepare_env_cfg`` (the regressed RTX-apply call) and env creation, then fail on the missing
    file. Reaching env creation with no replicator ``ModuleNotFoundError`` proves the fix.
    """
    argv = [
        "scripts/environments/teleoperation/teleop_replay_agent.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "--replay_file",
        str(tmp_path / "missing.mcap"),
        "--cloudxr_env",
        "none",
    ]
    output = _launch_until_marker(argv, [_ENV_CREATED_MARKER], tmp_path / "replay_agent.log")
    _assert_started_cleanly(output, [_ENV_CREATED_MARKER], "teleop_replay_agent.py")
