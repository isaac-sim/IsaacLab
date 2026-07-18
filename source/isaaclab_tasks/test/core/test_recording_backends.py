# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for VideoRecorder across all capture backends.

``_select_video_backend`` has three dispatch paths:

1. ``"kit"`` visualizer present + PhysX → Kit Replicator capture (IsaacsimKitPerspectiveVideo)
2. ``"newton"`` visualizer present      → Newton GL framebuffer  (NewtonGLVisualizer.render_rgb_array)
3. Neither above, OR Kit visualizer with Newton physics
   → physics manager video_capture_backend():
     "kit"        → standalone Kit capture
     "newton_gl"  → standalone Newton GL capture
     None         → RuntimeError (not tested here; covered by test_video_recorder.py)

When Newton is the physics backend, Kit Replicator recording is skipped
(Newton Fabric writes do not notify RTX's scene delegate, producing black frames).
_select_video_backend falls back to Newton GL standalone with a logged warning.

Path 3 fallthrough (Rerun, Viser, NewtonRTX visualizer types) is covered at the
unit level by test_video_recorder.py::test_resolve_backend_unsupported_visualizer_falls_back.

Setup:
    - AppLauncher(headless=True, enable_cameras=True)
    - CartpoleEnv (DirectRLEnv, 1 env) as the test vehicle.
Tests:
    - KitVisualizerCfg + PhysX + backend_source="visualizer"
      -> Kit Replicator capture (path 1) [passes]
    - KitVisualizerCfg + Newton + backend_source="visualizer"
      -> Newton GL standalone (path 3 fallback: Kit skipped for Newton physics) [passes]
    - NewtonGLVisualizerCfg + Newton + backend_source="visualizer"
      -> Newton GL framebuffer capture (path 2) [passes]
    - PhysX + backend_source="renderer"
      -> standalone Kit capture (path 3, PhysX physics) [passes]
    - Newton + backend_source="renderer"
      -> standalone Newton GL capture (path 3, Newton physics) [passes]
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
# Cartpole against a black background is sparse; even a valid capture only fills
# a small fraction of the frame. 0.5 % is enough to reject an all-zero buffer.
_MIN_NONZERO_RATIO = 0.005
_WARMUP_RENDER_BUDGET = 40


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
        # Kit Replicator render products are zero-initialised and need several RTX
        # frames before the buffer is populated (same warmup issue as OVRTX tests).
        # Poll until the frame is non-black or the budget is exhausted.
        for _ in range(_WARMUP_RENDER_BUDGET):
            frame = env.render()
            if frame is not None and np.count_nonzero(frame) > 0:
                return frame
            env.sim.render()
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
    """Kit visualizer + Newton + backend_source='visualizer' → Newton GL standalone (Kit skipped for Newton).

    Kit Replicator recording does not work with Newton physics because Newton Fabric
    writes do not notify RTX's scene delegate. _select_video_backend detects Newton
    and falls back to standalone Newton GL capture with a logged warning.
    """
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
    """NewtonVisualizerCfg + Newton + backend_source='visualizer' → Newton GL framebuffer capture."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    env_cfg = _cartpole_cfg(backend_source="visualizer")
    env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(window_width=_W, window_height=_H)]

    frame = _capture_frame(env_cfg)
    _assert_valid_frame(frame, "newton-gl-visualizer-source")


# ---------------------------------------------------------------------------
# Path 3 — no recording-capable visualizer → physics manager fallback
# ---------------------------------------------------------------------------
# Dispatch logic for Rerun/Viser/NewtonRTX visualizer types falling through to
# the physics manager is covered at unit level by:
#   test_video_recorder.py::test_resolve_backend_unsupported_visualizer_falls_back
# NewtonRTXVisualizerCfg is not factory-registered ("newton_rtx" is unsupported)
# and cannot be instantiated as a visualizer, so there is no integration test for it.


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
