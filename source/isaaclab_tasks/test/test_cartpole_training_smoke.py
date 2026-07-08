# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal end-to-end training smoke for cartpole.

Two cases — state-only and perception (RGB tiled camera) — each spawn the
unified ``isaaclab train`` entrypoint for two PPO iterations on a small env
count. They validate the full pipeline (``./isaaclab.sh`` wrapper, gym
registration, env build, RL wrapper, optimizer step, checkpoint write)
without the cost of a real training run, so the orchestrator can include
them in every CI shape (Linux, ARM/Spark).

The state case uses rsl_rl and the perception case rl_games, so the two
smokes cover two different RL-library dispatch paths.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# Cross-platform: ARM (Linux/aarch64) and Windows CI both opt in.
pytestmark = [pytest.mark.arm_ci, pytest.mark.windows_ci]

_REPO_ROOT = Path(__file__).resolve().parents[3]
# isaaclab.bat on Windows, isaaclab.sh on Linux/macOS — same CLI surface.
_LAUNCHER = str(_REPO_ROOT / ("isaaclab.bat" if os.name == "nt" else "isaaclab.sh"))


def _run_train(
    rl_library: str, task_name: str, num_envs: str = "16", extra_args: list[str] | None = None, timeout: int = 600
) -> None:
    """Spawn a trainer for two iterations and assert it exits cleanly."""
    cmd = [
        _LAUNCHER,
        "train",
        "--rl_library",
        rl_library,
        "--task",
        task_name,
        "--headless",
        "--num_envs",
        num_envs,
        "--max_iterations",
        "2",
        "--seed",
        "42",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"Training command failed for {task_name}: {' '.join(cmd)}\n"
        f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-4000:]}\n"
    )


def test_train_cartpole_state():
    """State-observation cartpole trains for two rsl_rl PPO iterations without errors."""
    _run_train("rsl_rl", "Isaac-Cartpole-Direct")


def test_train_cartpole_perception():
    """RGB-camera cartpole trains for two rl_games PPO iterations without errors."""
    # 32 envs: rl_games asserts batch_size (horizon_length * num_envs = 64 * 32)
    # is divisible by the agent cfg's minibatch_size (2048).
    _run_train(
        "rl_games",
        "Isaac-Cartpole-Camera-Direct",
        num_envs="32",
        extra_args=["--enable_cameras"],
    )
