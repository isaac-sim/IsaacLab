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

They also cover the XR startup regression, where a dependency on the unshipped
``omni.kit.xr.bundle.generic`` aborted ``--xr`` wherever the Kit SDK registry was unreachable.

These follow the Isaac Lab subprocess-integration test convention (see
``scripts/benchmarks/test``) rather than the isaacsim-CI methodology, so they run in the Isaac
Lab CI ``integration`` lane. They live under the ``isaaclab_teleop`` package tests so the CI
teleop filter picks them up, but exercise the entry-point scripts under ``scripts/``. The teleop
scripts run an unbounded interaction loop, so each is launched under ``PYTHONUNBUFFERED=1`` (the
startup prints are otherwise buffered), watched until a known startup marker appears, then
terminated via its process group.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

# Repository root: this file lives at source/isaaclab_teleop/test/<file>.py
ROOT = Path(__file__).resolve().parents[3]

# XR experience file whose dependencies decide whether ``--xr`` can start offline.
_XR_EXPERIENCE = ROOT / "apps" / "isaaclab.python.xr.openxr.kit"

# ``.kit`` files repeat section headers, so they cannot be parsed as TOML.
_SECTION_RE = re.compile(r"^\s*\[+(?P<name>[^\[\]]+)\]+\s*$")
_DEPENDENCY_RE = re.compile(r'^\s*"(?P<name>[^"]+)"\s*=')


def _kit_dependencies(experience: Path) -> set[str]:
    """Collect the extension names declared in an experience file's dependency sections.

    Scoped to ``[dependencies]`` tables so quoted keys elsewhere in the file (settings tables
    use them too) are not mistaken for extensions.
    """
    dependencies: set[str] = set()
    in_dependencies = False

    for line in experience.read_text(encoding="utf-8").splitlines():
        section = _SECTION_RE.match(line)
        if section is not None:
            in_dependencies = section.group("name").strip() == "dependencies"
            continue
        if in_dependencies:
            dependency = _DEPENDENCY_RE.match(line)
            if dependency is not None:
                dependencies.add(dependency.group("name"))

    return dependencies


# Lightweight Franka task that configures an IsaacTeleop pipeline (controllers), so omitting
# ``--teleop_device`` drives the scripts through the Isaac Teleop stack rather than the legacy
# native devices.
_ISAAC_TELEOP_TASK = "IsaacContrib-Stack-Cube-Franka-IK-Abs"

# Exact regression signatures that must never appear in a script's output.
_REGRESSION_SIGNATURES = (
    # #6656 render-flag cleanup fallout
    "No module named 'omni.replicator'",
    "has no attribute 'enable_cameras'",
    "has no attribute 'headless'",
    # Registry-only XR dependency: the app exits before any marker is printed
    "Exiting app because of dependency solver failure",
)

# Logged by ``isaaclab_teleop`` when it builds the Isaac Teleop device + retargeting pipeline.
# This runs *after* the RTX-apply site that regressed, and before the OpenXR/CloudXR session is
# entered -- which needs a headset/CloudXR runtime (``NV_CXR_RUNTIME_DIR``) not present in headless
# CI. Reaching it proves the RTX/camera startup and the Isaac Teleop factory both succeed.
_ISAAC_TELEOP_MARKER = "Using IsaacTeleop stack for teleoperation"

# Marker logged by ManagerBasedEnv creation, which runs *after* the RTX-apply site that regressed.
_ENV_CREATED_MARKER = "Base environment:"

# Disable CloudXR: no headset connects in CI, and auto-launching the runtime is unnecessary to
# exercise the Isaac Teleop device factory (the live session is out of scope for a headless smoke).
_NO_CLOUDXR = ["--cloudxr_env", "none", "--no-auto_launch_cloudxr"]

# Generous ceiling for a cold Isaac Sim launch + Franka env creation on CI hardware. On a warm
# shader/asset cache the marker appears in ~25s, but the first launch on a cold runner can take
# ~500s+ while caches populate, so the ceiling is sized well above that to avoid spurious timeouts.
# The watcher returns as soon as the marker is seen, so this costs nothing on warm runs.
_STARTUP_TIMEOUT_S = 900.0


def _launch_until_marker(argv: list[str], markers: list[str], log_path: Path) -> str:
    """Launch a teleop script and return its output once a marker appears (or it exits / times out).

    The child is started in its own process group and unbuffered so both ``print`` startup markers
    and ``[INFO]`` logs surface promptly; the whole group is terminated before returning.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(ROOT / argv[0]), *argv[1:]]
    # Keep the write handle open for the child's whole lifetime (the poll loop reads progress back
    # through a separate read handle); it is closed only when this ``with`` block exits after teardown.
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
    """teleop_se3_agent.py builds the Isaac Teleop device (past the #6656 RTX/camera regression).

    Omitting ``--teleop_device`` selects the Isaac Teleop stack for the IsaacTeleop-configured task.
    The run reaches the Isaac Teleop device factory (``_ISAAC_TELEOP_MARKER``) and then stops when it
    tries to enter the OpenXR/CloudXR session (no headset in CI); reaching the marker is success.
    """
    argv = [
        "scripts/environments/teleoperation/teleop_se3_agent.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        *_NO_CLOUDXR,
    ]
    output = _launch_until_marker(argv, [_ISAAC_TELEOP_MARKER], tmp_path / "se3_agent.log")
    _assert_started_cleanly(output, [_ISAAC_TELEOP_MARKER], "teleop_se3_agent.py")


def test_record_demos_starts(tmp_path):
    """record_demos.py builds the Isaac Teleop device without AttributeError-ing on enable_cameras.

    Same Isaac Teleop path as ``teleop_se3_agent``: reaching ``_ISAAC_TELEOP_MARKER`` proves the RTX
    apply and the ``args_cli.enable_cameras`` read both succeed before the CloudXR session boundary.
    """
    argv = [
        "scripts/tools/record_demos.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "--dataset_file",
        str(tmp_path / "dataset.hdf5"),
        *_NO_CLOUDXR,
    ]
    output = _launch_until_marker(argv, [_ISAAC_TELEOP_MARKER], tmp_path / "record_demos.log")
    _assert_started_cleanly(output, [_ISAAC_TELEOP_MARKER], "record_demos.py")


def test_xr_experience_declares_only_shipped_extensions():
    """The XR experience depends on XR extensions Isaac Sim ships, not on a registry-only bundle.

    ``omni.kit.xr.bundle.generic`` is not shipped and resolves only from the Kit SDK registry, so
    depending on it aborted every ``--xr`` run that could not reach the registry.

    Static on purpose: a machine that reaches the registry caches the bundle on first use and
    loads it locally forever after, so a launch test cannot catch this once that cache is warm.
    """
    dependencies = _kit_dependencies(_XR_EXPERIENCE)

    assert "omni.kit.xr.system.openxr" in dependencies
    assert "omni.kit.xr.ui.window.profile" in dependencies, "omni.kit.xr.profile.ar's Kit 110 successor is missing"
    assert not [name for name in dependencies if name.startswith("omni.kit.xr.bundle.")], (
        "XR bundle meta-extensions are not shipped with Isaac Sim and resolve only from the Kit SDK registry"
    )


def test_record_demos_starts_headless_xr(tmp_path):
    """record_demos.py starts under ``--xr`` with no ``--viz kit`` (the QA-reported failure).

    ``--xr`` without an explicit windowed visualizer resolves to headless and selects
    ``isaaclab.python.xr.openxr.headless.kit``, whose dependency chain aborted the app before any
    Python ran. Either marker is accepted: env creation already clears the dependency solver, which
    is what this gates on. The run stops short of the OpenXR session, which needs a headset.

    The registry is disabled so extensions must resolve from what Isaac Sim ships, keeping the test
    hermetic. See :func:`test_xr_experience_declares_only_shipped_extensions` for the warm-cache
    caveat.
    """
    argv = [
        "scripts/tools/record_demos.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "--dataset_file",
        str(tmp_path / "dataset.hdf5"),
        "--xr",
        *_NO_CLOUDXR,
        "--kit_args",
        "--/app/extensions/registryEnabled=false",
    ]
    markers = [_ISAAC_TELEOP_MARKER, _ENV_CREATED_MARKER]
    output = _launch_until_marker(argv, markers, tmp_path / "record_demos_xr.log")
    _assert_started_cleanly(output, markers, "record_demos.py --xr")


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
