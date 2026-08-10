# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared state-training, camera-rendering, and camera-training Cartpole smoke probes.

The caller owns environment preparation. Installation CI executes this file
inside the environment it just installed, while architecture CI executes the
same probes inside its prepared image.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_STATE_TRAIN_CMD = [
    "train",
    "--rl_library",
    "rsl_rl",
    "--task",
    "Isaac-Cartpole-Direct",
    "--num_envs",
    "16",
    "presets=newton_mjwarp",
    "--max_iterations",
    "5",
]

_CAMERA_TRAIN_CMD = [
    "train",
    "--rl_library",
    "rsl_rl",
    "--task",
    "Isaac-Cartpole-Camera-Direct",
    "--num_envs",
    "16",
    "presets=newton_mjwarp,newton_renderer",
    "--max_iterations",
    "2",
]


def _find_isaaclab_root() -> Path:
    """Return the repository root containing the Isaac Lab launcher."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "isaaclab.sh").exists() or (parent / "isaaclab.bat").exists():
            return parent
    raise FileNotFoundError("Could not locate the Isaac Lab repository root")


def _assert_training_passed(result: subprocess.CompletedProcess[str]) -> None:
    """Assert that a short training run completed without a traceback."""
    output = result.stdout + (result.stderr or "")
    assert result.returncode == 0, f"Training failed (rc={result.returncode}):\n{output}"
    assert "Traceback (most recent call last):" not in output, f"Training produced a traceback:\n{output}"
    assert "Training time:" in output, f"Training did not report completion:\n{output}"


def _run_training(command: list[str], timeout: int) -> None:
    """Run one training command in the caller's active environment."""
    isaaclab_root = _find_isaaclab_root()
    launcher = isaaclab_root / ("isaaclab.bat" if os.name == "nt" else "isaaclab.sh")
    result = subprocess.run(
        [str(launcher)] + command,
        cwd=isaaclab_root,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    _assert_training_passed(result)


def test_train_cartpole_state_completes() -> None:
    """Verify that state-observation Cartpole completes short rsl_rl training."""
    _run_training(_STATE_TRAIN_CMD, timeout=600)


def test_render_cartpole_camera_produces_valid_observation_and_reward() -> None:
    """Verify that camera Cartpole renders varying pixels and produces finite rewards."""
    import torch

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    env_cfg = resolve_presets(CartpoleCameraEnvCfg(), selected={"newton_mjwarp", "newton_renderer"})
    env_cfg.scene.num_envs = 2
    env_cfg.frame_stack = 1
    env = None
    try:
        env = CartpoleCameraEnv(cfg=env_cfg)
        obs, _ = env.reset()
        image = obs["policy"]
        expected_shape = (2, 3, env_cfg.tiled_camera.height, env_cfg.tiled_camera.width)
        assert tuple(image.shape) == expected_shape, (
            f"Camera observation shape {tuple(image.shape)} != {expected_shape}"
        )
        assert torch.isfinite(image).all(), "Camera observation contains NaN or infinity"
        assert image.amax() > image.amin(), "Camera observation is constant"

        action = torch.zeros(env.num_envs, 1, device=env.device)
        _, reward, _, _, _ = env.step(action)
        assert torch.isfinite(reward).all(), "Cartpole reward contains NaN or infinity"
    finally:
        if env is not None:
            env.close()


def test_train_cartpole_camera_completes() -> None:
    """Verify that camera-observation Cartpole completes short RSL-RL training."""
    # A cold camera run compiles shaders before training begins.
    _run_training(_CAMERA_TRAIN_CMD, timeout=1800)


if __name__ == "__main__":
    test_train_cartpole_state_completes()
    test_render_cartpole_camera_produces_valid_observation_and_reward()
    test_train_cartpole_camera_completes()
