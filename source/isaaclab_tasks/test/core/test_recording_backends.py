# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for VideoRecorder across all capture backends.

``_select_video_backend`` has three dispatch paths:

1. ``"kit"`` visualizer present   → Kit Replicator capture (IsaacsimKitPerspectiveVideo)
2. ``"newton"`` visualizer present → Newton GL framebuffer  (NewtonGLVisualizer.render_rgb_array)
3. Neither above (any other visualizer or no visualizer)
   → physics manager video_capture_backend():
     "kit"        → standalone Kit capture
     "newton_gl"  → standalone Newton GL capture
     None         → RuntimeError (not tested here; covered by test_video_recorder.py)

NewtonGLVisualizerCfg is the only Newton type that supports recording (type="newton").
NewtonRTXVisualizerCfg (type="newton_rtx"), RerunVisualizerCfg (type="rerun"),
and ViserVisualizerCfg (type="viser") are all unrecognised by _select_video_backend
and fall through to path 3.

Setup:
    - AppLauncher(headless=True, enable_cameras=True)
    - CartpoleEnv (DirectRLEnv, 1 env) as the test vehicle.
Tests:
    - KitVisualizerCfg + PhysX + backend_source="visualizer"
      -> Kit Replicator capture (path 1, PhysX physics)
    - KitVisualizerCfg + Newton + backend_source="visualizer"
      -> Kit Replicator capture (path 1, Newton physics — cross-backend)
    - NewtonGLVisualizerCfg + Newton + backend_source="visualizer"
      -> Newton GL framebuffer capture (path 2)
    - NewtonRTXVisualizerCfg + Newton + backend_source="visualizer"
      -> falls through to physics manager → Newton GL standalone (path 3)
    - PhysX + backend_source="renderer"
      -> standalone Kit capture (path 3, PhysX physics)
    - Newton + backend_source="renderer"
      -> standalone Newton GL capture (path 3, Newton physics)
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


# ---------------------------------------------------------------------------
# Path 1 — Kit visualizer (type="kit") → Kit Replicator capture
# ---------------------------------------------------------------------------


def test_kit_visualizer_physx_source_records_rgb():
    """Kit visualizer + PhysX + backend_source='visualizer' → Kit Replicator capture."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.physics = PhysxCfg()
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=_W, window_height=_H)]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "kit-visualizer-physx")


def test_kit_visualizer_newton_source_records_rgb():
    """Kit visualizer + Newton physics + backend_source='visualizer' → Kit Replicator capture (cross-backend)."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=_W, window_height=_H)]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "kit-visualizer-newton")


# ---------------------------------------------------------------------------
# Path 2 — Newton GL visualizer (type="newton") → Newton GL framebuffer
# ---------------------------------------------------------------------------


def test_newton_gl_visualizer_source_records_rgb():
    """NewtonGLVisualizerCfg + Newton + backend_source='visualizer' → Newton GL framebuffer capture."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(window_width=_W, window_height=_H)]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "newton-gl-visualizer-source")


# ---------------------------------------------------------------------------
# Path 3 — unrecognised visualizer / no visualizer → physics manager fallback
# ---------------------------------------------------------------------------


def test_newton_rtx_visualizer_falls_through_to_physics_manager():
    """NewtonRTXVisualizerCfg (type='newton_rtx') is unrecognised; falls back to Newton GL standalone."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env_cfg.sim.visualizer_cfgs = [NewtonRTXVisualizerCfg()]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "newton-rtx-visualizer-fallthrough")


def test_physx_renderer_source_records_rgb():
    """PhysX + backend_source='renderer' → standalone Kit capture (no visualizer required)."""
    from isaaclab_physx.physics import PhysxCfg

    env_cfg = _cartpole_cfg(backend_source="renderer")
    env_cfg.sim.physics = PhysxCfg()

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "physx-renderer-source")


def test_newton_renderer_source_records_rgb():
    """Newton + backend_source='renderer' → standalone Newton GL capture (no visualizer required)."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    env_cfg = _cartpole_cfg(backend_source="renderer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "newton-renderer-source")
