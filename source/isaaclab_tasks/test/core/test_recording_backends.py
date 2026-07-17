# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for VideoRecorder across all capture backends.

Setup:
    - AppLauncher(headless=True, enable_cameras=True)
    - CartpoleEnv (DirectRLEnv) as the test vehicle; 1 env, 10 physics steps before capture.
Tests:
    - CartpoleEnv(render_mode="rgb_array") + KitVisualizerCfg + backend_source="visualizer"
      -> verify env.render() returns a non-trivial (_H, _W, 3) uint8 frame
    - CartpoleEnv(render_mode="rgb_array") + NewtonVisualizerCfg + Newton physics + backend_source="visualizer"
      -> verify env.render() returns a non-trivial (_H, _W, 3) uint8 frame
    - CartpoleEnv(render_mode="rgb_array") + PhysX physics + backend_source="renderer"
      -> verify env.render() returns a non-trivial (_H, _W, 3) uint8 frame (standalone Kit capture)
    - CartpoleEnv(render_mode="rgb_array") + Newton physics + backend_source="renderer"
      -> verify env.render() returns a non-trivial (_H, _W, 3) uint8 frame (standalone Newton GL capture)
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import VideoRecorderCfg

pytestmark = pytest.mark.isaacsim_ci

_W = 320
_H = 240
_N_STEPS = 10
_MIN_NONZERO_RATIO = 0.01


def _cartpole_cfg(*, backend_source: str = "visualizer"):
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = 1
    cfg.video_recorder = VideoRecorderCfg(backend_source=backend_source, window_width=_W, window_height=_H)
    return cfg


def _capture_frame(env_cfg) -> np.ndarray | None:
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

    sim_utils.create_new_stage()
    env = CartpoleEnv(env_cfg, render_mode="rgb_array")
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, *env.action_space.shape[1:], device=env.device)
        for _ in range(_N_STEPS):
            env.step(actions)
        return env.render()
    finally:
        env.close()


def _assert_valid_frame(frame: np.ndarray | None, label: str) -> None:
    assert frame is not None, f"{label}: env.render() returned None"
    assert isinstance(frame, np.ndarray), f"{label}: expected ndarray, got {type(frame)}"
    assert frame.ndim == 3 and frame.shape[2] == 3, f"{label}: unexpected shape {frame.shape}"
    assert frame.shape[:2] == (_H, _W), f"{label}: expected ({_H}, {_W}, 3), got {frame.shape}"
    assert frame.dtype == np.uint8, f"{label}: expected uint8, got {frame.dtype}"
    nonzero_ratio = np.count_nonzero(frame) / frame.size
    assert nonzero_ratio >= _MIN_NONZERO_RATIO, (
        f"{label}: frame appears to be all-black (nonzero ratio {nonzero_ratio:.4f} < {_MIN_NONZERO_RATIO})"
    )


def test_kit_visualizer_source_records_rgb():
    """Kit visualizer + backend_source='visualizer' captures from the Kit perspective camera."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=_W, window_height=_H)]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "kit-visualizer-source")


def test_newton_visualizer_source_records_rgb():
    """Newton visualizer + Newton physics + backend_source='visualizer' captures from the Newton GL framebuffer."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(window_width=_W, window_height=_H)]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "newton-visualizer-source")


def test_physx_renderer_source_records_rgb():
    """PhysX + backend_source='renderer' captures from a standalone Kit perspective video (no visualizer required)."""
    from isaaclab_physx.physics import PhysxCfg

    env_cfg = _cartpole_cfg(backend_source="renderer")
    env_cfg.sim.physics = PhysxCfg()

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "physx-renderer-source")


def test_newton_renderer_source_records_rgb():
    """Newton + backend_source='renderer' captures from a standalone Newton GL perspective video."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    env_cfg = _cartpole_cfg(backend_source="renderer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "newton-renderer-source")
