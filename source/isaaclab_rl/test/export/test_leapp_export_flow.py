# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for LEAPP export commands."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from leapp_direct_env import DIRECT_TASK
from leapp_initialized_checkpoints import resolved_path_file, task_checkpoint_dir

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEAPP_ROOT = _REPO_ROOT / "scripts/reinforcement_learning/leapp"
_CHECKPOINT_SCRIPT = Path(__file__).with_name("leapp_initialized_checkpoints.py")
_OUTPUT_TAIL_SIZE = 5000

_PRETRAINED_TASKS = [
    "Isaac-Ant",
    "Isaac-Cartpole",
    "IsaacContrib-Navigation-Flat-AnymalC",
    "Isaac-Velocity-Flat-AnymalD",
    "Isaac-Velocity-Rough-AnymalD",
    "Isaac-Velocity-Rough-G1",
    "IsaacContrib-Velocity-Flat-Spot",
    "Isaac-Reach-Franka",
    "Isaac-Lift-Franka",
    "Isaac-Open-Drawer-Franka",
    "Isaac-Reorient-KukaAllegro",
]

_INITIALIZED_CASES = [
    ("rl_games", "Isaac-Cartpole", "newton_mjwarp"),
    ("skrl", "Isaac-Cartpole", "newton_mjwarp"),
    ("sb3", "Isaac-Cartpole", "newton_mjwarp"),
    ("rsl_rl", "IsaacContrib-Lift-Cube-Franka", None),
    ("rsl_rl", "Isaac-Lift-KukaAllegro", "newton_mjwarp"),
    ("rsl_rl", "Isaac-Humanoid", "newton_mjwarp"),
    ("rsl_rl", "Isaac-Humanoid", "isaacsim_physx"),
    pytest.param(
        "rsl_rl",
        "Isaac-Humanoid",
        "ovphysx",
        marks=pytest.mark.skipif(importlib.util.find_spec("ovphysx") is None, reason="requires the ovphysx extra"),
    ),
]


@pytest.fixture(autouse=True)
def skip_franka(task: str) -> None:
    if "Franka" in task:
        pytest.skip("Known Franka asset cloning issue")


def _run_checked(command: list[str], timeout: int = 600) -> str:
    """Run a command and return its combined output."""
    # OpenUSD versions before 26.5 can corrupt the heap while parsing Franka
    # payloads concurrently. Set the limit before importing USD in the child.
    try:
        result = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PXR_WORK_THREAD_LIMIT": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        message = "\n".join(
            [
                f"Command timed out after {timeout}s: {command}",
                stdout[-_OUTPUT_TAIL_SIZE:],
                stderr[-_OUTPUT_TAIL_SIZE:],
            ]
        )
        pytest.fail(message)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{command} exited with {result.returncode}:\n{output[-_OUTPUT_TAIL_SIZE:]}"
    assert "Traceback (most recent call last):" not in output, output[-_OUTPUT_TAIL_SIZE:]
    return output


def _run_export(backend: str, task: str, checkpoint: str, tmp_path: Path, preset: str | None) -> None:
    script = (
        Path(__file__).with_name("leapp_direct_env.py") if task == DIRECT_TASK else _LEAPP_ROOT / backend / "export.py"
    )
    command = [
        sys.executable,
        str(script),
        "--task",
        task,
        "--checkpoint",
        checkpoint,
        "--export_save_path",
        str(tmp_path / "export"),
        "--disable_graph_visualization",
        "--limit_cpu_threads",
        "1",
    ]
    if preset:
        command.append(f"presets={preset}")
    output = _run_checked(command)
    if checkpoint == "pretrained" and "pre-trained checkpoint is currently unavailable" in output:
        pytest.skip(f"No published checkpoint for {task}")
    artifact_dir = tmp_path / "export" / task
    for name in (f"{task}.onnx", f"{task}.yaml", "log.txt"):
        assert (artifact_dir / name).is_file(), f"Missing {name}:\n{output[-_OUTPUT_TAIL_SIZE:]}"


@pytest.mark.integration
@pytest.mark.parametrize("task", [*_PRETRAINED_TASKS, DIRECT_TASK])
def test_pretrained_export(task: str, tmp_path: Path) -> None:
    """Published RSL-RL checkpoints produce the expected export artifacts."""
    _run_export("rsl_rl", task, "pretrained", tmp_path, preset=None)


@pytest.mark.integration
@pytest.mark.parametrize(("backend", "task", "preset"), _INITIALIZED_CASES)
def test_initialized_export(backend: str, task: str, preset: str | None, tmp_path: Path) -> None:
    """Initialized checkpoints produce artifacts with the requested physics preset."""
    command = [
        sys.executable,
        str(_CHECKPOINT_SCRIPT),
        "--checkpoint_root",
        str(tmp_path),
        "--spec",
        backend,
        task,
    ]
    if preset:
        command.append(preset)
    _run_checked(command, timeout=1200)
    checkpoint = resolved_path_file(task_checkpoint_dir(tmp_path, backend, task)).read_text(encoding="utf-8").strip()
    assert Path(checkpoint).is_file()
    _run_export(backend, task, checkpoint, tmp_path, preset)
