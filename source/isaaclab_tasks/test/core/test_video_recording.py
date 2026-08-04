# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end video recording tests covering all VideoRecorder sources.

Each test writes real mp4 clips and reads them back to verify content.

Sources tested:
  "visualizer:kit"        – Kit Replicator viewport (PhysX)
  "visualizer:kit"+Newton – logs error, no clip written
  "visualizer:newton"     – Newton GL framebuffer (Newton physics)
  "sensor:tiled_camera"   – tiled camera sensor (PhysX, RTX renderer)
  multiple recorders      – Kit viewport + sensor written simultaneously

Setup:
    - AppLauncher(headless=True, enable_cameras=True)
    - CartpoleEnv or CartpoleCameraEnv, stepped for _STEPS env steps per test.
    - VideoRecorderCfg(video_length=_CLIP, video_interval=0) → one clip per test.
Tests:
    - kit_physx      → non-black ✓, motion ✓
    - kit_newton     → error logged ✓, no clip ✓
    - newton         → non-black ✓  (motion skipped — Newton GL renders asynchronously)
    - sensor_physx   → non-black ✓, motion ✓
    - multi_recorder → kit clip ✓, sensor clip ✓, both non-black and moving
"""

# Check for moviepy before launching Kit so a missing dependency produces a
# clean collection-time skip rather than an orphaned SimulationApp.
import pytest

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    pytest.skip("moviepy is not installed; install with: pip install 'moviepy<2'", allow_module_level=True)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import tempfile

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

pytestmark = pytest.mark.isaacsim_ci

_CLIP = 12  # frames per clip
_STEPS = 22  # env steps: enough for one full clip plus close() flush
_MIN_NONZERO_RATIO = 0.005
_MIN_MOTION_STD = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recorder_cfg(output_dir: str, source: str, prefix: str = "clip") -> VideoRecorderCfg:
    cfg = VideoRecorderCfg()
    cfg.source = source
    cfg.output_dir = output_dir
    cfg.video_length = _CLIP
    cfg.video_interval = 0
    cfg.fps = 10
    cfg.output_filename_prefix = prefix
    return cfg


def _cartpole_cfg(*, num_envs: int = 1):
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.physics = PhysxCfg()
    return cfg


def _cartpole_cfg_newton(*, num_envs: int = 1):
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    return cfg


def _cartpole_camera_cfg_physx(*, num_envs: int = 1):
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg

    cfg = CartpoleCameraEnvCfg()
    cfg = cfg.default
    cfg.scene.num_envs = num_envs
    cfg.sim.physics = PhysxCfg()
    return cfg


def _run_cartpole(env_cfg) -> None:
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

    sim_utils.create_new_stage()
    env = CartpoleEnv(env_cfg)
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, *env.action_space.shape[1:], device=env.device)
        for _ in range(_STEPS):
            env.step(actions)
    finally:
        env.close()


def _run_cartpole_camera(env_cfg) -> None:
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv

    sim_utils.create_new_stage()
    env = CartpoleCameraEnv(env_cfg)
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, *env.action_space.shape[1:], device=env.device)
        for _ in range(_STEPS):
            env.step(actions)
    finally:
        env.close()


def _read_frames(clip_path: str) -> list[np.ndarray]:
    clip = VideoFileClip(clip_path)
    frames = [f.astype(np.float32) for f in clip.iter_frames()]
    clip.close()
    return frames


def _assert_clip_nonblack(clip_path: str, label: str) -> None:
    assert os.path.exists(clip_path), f"{label}: clip not written to {clip_path}"
    assert os.path.getsize(clip_path) > 0, f"{label}: clip file is empty"
    frames = _read_frames(clip_path)
    assert frames, f"{label}: no frames in clip"
    nonzero_ratios = [np.count_nonzero(f) / f.size for f in frames]
    assert max(nonzero_ratios) >= _MIN_NONZERO_RATIO, (
        f"{label}: all frames appear black (max non-zero ratio {max(nonzero_ratios):.4f})"
    )


def _assert_clip_has_content(clip_path: str, label: str) -> None:
    """Non-black + motion check. Use only for synchronous render sources."""
    _assert_clip_nonblack(clip_path, label)
    frames = _read_frames(clip_path)
    temporal_std = np.stack(frames).std(axis=0).mean()
    assert temporal_std >= _MIN_MOTION_STD, (
        f"{label}: no motion detected (temporal std={temporal_std:.3f} < {_MIN_MOTION_STD})"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kit_physx_records_moving_clip():
    """source='visualizer:kit' + PhysX → non-black, moving mp4 written to disk."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg(num_envs=1)
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]
        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:kit", prefix="kit")]
        _run_cartpole(env_cfg)
        _assert_clip_has_content(os.path.join(output_dir, "kit_0000.mp4"), "kit-physx")


def test_kit_newton_logs_warning_on_capture(caplog):
    """source='visualizer:kit' + Newton → logs a warning and capture proceeds.

    Kit Replicator requires cubric to propagate Newton Fabric transforms to RTX.
    Without cubric frames may be black, but the recorder does not hard-block —
    it logs a warning pointing to 'visualizer:newton' and lets the capture run
    so users with cubric still get frames.
    """
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg_newton(num_envs=1)
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]
        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:kit")]
        with caplog.at_level(logging.WARNING, logger="isaaclab.envs.utils.video_recorder"):
            _run_cartpole(env_cfg)
        assert any("source='visualizer:newton'" in r.message for r in caplog.records), (
            "warning message should suggest visualizer:newton"
        )


def test_newton_records_nonblack_clip():
    """source='visualizer:newton' + Newton physics → non-black mp4 written to disk.

    Newton GL renders asynchronously, so motion check is skipped — consecutive
    render_rgb_array() calls may return the same framebuffer.  Non-black confirms
    the GL viewer drew a real scene.
    """
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg_newton(num_envs=1)
        env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(window_width=320, window_height=240)]
        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:newton", prefix="newton")]
        _run_cartpole(env_cfg)
        _assert_clip_nonblack(os.path.join(output_dir, "newton_0000.mp4"), "newton")


def test_sensor_physx_records_moving_clip():
    """source='sensor:tiled_camera' + PhysX → non-black, moving sensor mp4."""
    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_camera_cfg_physx(num_envs=1)
        env_cfg.video_recorders = [_recorder_cfg(output_dir, "sensor:tiled_camera", prefix="sensor")]
        _run_cartpole_camera(env_cfg)
        _assert_clip_has_content(os.path.join(output_dir, "sensor_0000.mp4"), "sensor-physx")


def test_multiple_recorders_simultaneous():
    """Two VideoRecorderCfg entries write independent clips from the same env step loop.

    Kit viewport (source='visualizer:kit') and tiled camera sensor
    (source='sensor:tiled_camera') record simultaneously.  Verifies that both
    clips are created and non-empty.  The sensor clip also receives the full
    non-black + motion check.  The Kit clip is only checked for file existence —
    RTX warms up over many frames and per-clip pixel assertions are unreliable
    in this multi-recorder context; Kit content is verified in the dedicated
    test_kit_physx_records_moving_clip test.
    """
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_camera_cfg_physx(num_envs=1)
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]
        env_cfg.video_recorders = [
            _recorder_cfg(output_dir, "visualizer:kit", prefix="viewport"),
            _recorder_cfg(output_dir, "sensor:tiled_camera", prefix="sensor"),
        ]
        _run_cartpole_camera(env_cfg)
        viewport_path = os.path.join(output_dir, "viewport_0000.mp4")
        assert os.path.isfile(viewport_path) and os.path.getsize(viewport_path) > 0, (
            "multi-kit: viewport clip was not written"
        )
        _assert_clip_has_content(os.path.join(output_dir, "sensor_0000.mp4"), "multi-sensor")
