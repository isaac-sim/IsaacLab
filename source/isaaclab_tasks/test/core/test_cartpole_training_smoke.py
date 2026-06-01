# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal end-to-end training smoke for cartpole.

Two cases — state-only and perception (RGB tiled camera) — each spawn a
``scripts/reinforcement_learning/<framework>/train.py`` for two PPO iterations
on a small env count. They validate the full pipeline (``./isaaclab.sh``
wrapper, gym registration, env build, RL wrapper, optimizer step, checkpoint
write) without the cost of a real training run, so the orchestrator can
include them in every CI shape (Linux, ARM/Spark).

The state case uses rsl_rl (matches Isaac-Cartpole-Direct-v0's registered
config entry); the perception case uses rl_games because the camera-variant
direct envs register ``rl_games_cfg_entry_point`` and ``skrl_cfg_entry_point``
but no ``rsl_rl_cfg_entry_point``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.arm_ci

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _run_train(train_script: str, task_name: str, extra_args: list[str] | None = None, timeout: int = 600) -> None:
    """Spawn a trainer for two iterations and assert it exits cleanly."""
    cmd = [
        "./isaaclab.sh",
        "-p",
        train_script,
        "--task",
        task_name,
        "--headless",
        "--num_envs",
        "16",
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
    _run_train("scripts/reinforcement_learning/rsl_rl/train.py", "Isaac-Cartpole-Direct-v0")


def test_train_cartpole_perception():
    """RGB-camera cartpole trains for two rl_games PPO iterations without errors."""
    # The first camera-enabled run on a cold cache compiles shaders (~600 s)
    # before training starts, so allow well beyond the state-case budget.
    _run_train(
        "scripts/reinforcement_learning/rl_games/train.py",
        "Isaac-Cartpole-Camera-Direct-v0",
        extra_args=["--enable_cameras"],
        timeout=1800,
    )
