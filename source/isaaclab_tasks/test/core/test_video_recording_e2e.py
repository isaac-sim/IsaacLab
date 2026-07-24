# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end video recording tests for all VideoRecorder sources.

Each test writes real mp4 clips to /tmp/isaaclab_recording_tests/ and reads
them back to verify content.  Two checks are applied:

  Non-black check (all tests):
    at least one frame has >= 0.5% non-zero pixels — rules out solid-black output.

  Motion check (Kit and sensor tests only, NOT Newton GL):
    mean temporal std across pixels >= 0.1 — confirms the scene is actually moving.
    Threshold is intentionally loose (Kit nontiled achieves ~36, Kit tiled ~0.22,
    sensors ~1–35); the point is distinguishing live rendering from a frozen frame.

Why Newton GL skips the motion check
-------------------------------------
Newton GL renders in a background OpenGL thread.  ``render_rgb_array()`` returns
the latest framebuffer snapshot via ``get_frame()``, but that snapshot is only
updated when the GL thread completes a render pass.  Because the GL thread runs
asynchronously from the env step loop, consecutive ``get_frame()`` calls within a
short burst often return the same image, giving a near-zero temporal std even when
the physics simulation has advanced.  The non-black check still confirms that the
GL viewer drew a real scene.

Sources tested
--------------
  "visualizer:kit"           – Kit Replicator viewport capture (PhysX)
  "visualizer:kit:tiled"     – Kit tiled camera grid (PhysX, tiled_cam_view=True)
  "visualizer:newton"        – Newton GL framebuffer, interactive view (Newton)
  "visualizer:newton" tiled  – Newton GL framebuffer, tiled-panel view (Newton)
  "sensor:tiled_camera" RTX  – Camera sensor, RTX renderer (PhysX)
  "sensor:tiled_camera" Warp – Camera sensor, Newton Warp renderer (Newton physics)
  multiple VideoRecorderCfg  – Kit viewport + sensor written simultaneously

Setup:
    - AppLauncher(headless=True, enable_cameras=True)
    - CartpoleEnv (or CartpoleCameraEnv), num_envs=1 or 4 for tiled tests.
    - _CLIP frames per clip; _STEPS env steps ensure the clip closes naturally.
Tests:
    - kit_nontiled_physx      → non-black ✓, motion ✓
    - kit_tiled_physx         → non-black ✓, motion ✓ (tiled grid)
    - newton_nontiled         → non-black ✓  (motion skipped — async GL)
    - newton_tiled_viewer     → non-black ✓  (motion skipped — async GL)
    - sensor_rtx_physx        → non-black ✓, motion ✓ (RTX renderer)
    - sensor_warp_newton      → non-black ✓, motion ✓ (Newton Warp renderer)
    - multiple_recorders      → both kit + sensor clips non-black ✓ and moving ✓
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import shutil

import numpy as np
import pytest
import torch
from moviepy.editor import VideoFileClip

import isaaclab.sim as sim_utils
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

pytestmark = pytest.mark.isaacsim_ci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLIP = 12  # frames per clip
_STEPS = 22  # env steps per test: enough for one full clip plus close() flush
_MIN_NONZERO_RATIO = 0.005  # at least 0.5% non-black pixels in one frame
_MIN_MOTION_STD = 0.1  # mean temporal std across pixels for motion-capable sources

OUTPUT_DIR = "/tmp/isaaclab_recording_tests"


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
    """CartpoleCameraEnv with PhysX physics and RGB tiled camera (RTX renderer)."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg

    cfg = CartpoleCameraEnvCfg()
    cfg = cfg.default  # rgb preset, RTX renderer
    cfg.scene.num_envs = num_envs
    cfg.sim.physics = PhysxCfg()
    return cfg


def _cartpole_camera_cfg_newton(*, num_envs: int = 1):
    """CartpoleCameraEnv with Newton physics and RGB tiled camera (Newton Warp renderer).

    CartpoleCameraEnvCfg uses MultiBackendRendererCfg; under Newton physics the
    camera sensor automatically switches to the Newton Warp renderer.
    """
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg

    cfg = CartpoleCameraEnvCfg()
    cfg = cfg.default
    cfg.scene.num_envs = num_envs
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
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
    """Read all frames from an mp4 as float32 arrays in [0, 255]."""
    clip = VideoFileClip(clip_path)
    frames = [f.astype(np.float32) for f in clip.iter_frames()]
    clip.close()
    return frames


def _assert_clip_nonblack(clip_path: str, label: str) -> None:
    """Assert the clip exists, is non-empty, and has at least one non-black frame."""
    assert os.path.exists(clip_path), f"{label}: clip not written to {clip_path}"
    assert os.path.getsize(clip_path) > 0, f"{label}: clip file is empty"

    frames = _read_frames(clip_path)
    assert frames, f"{label}: no frames in clip"

    nonzero_ratios = [np.count_nonzero(f) / f.size for f in frames]
    max_nonzero = max(nonzero_ratios)
    assert max_nonzero >= _MIN_NONZERO_RATIO, (
        f"{label}: all frames appear black (max non-zero ratio {max_nonzero:.4f} < {_MIN_NONZERO_RATIO})"
    )
    print(f"  [non-black ✓] {label}: {len(frames)} frames, max_nonzero={max_nonzero:.3f}")


def _assert_clip_shows_motion(clip_path: str, label: str) -> None:
    """Assert the clip's frames change over time (temporal std >= _MIN_MOTION_STD).

    Use only for sources whose rendering is synchronous with the env step
    (Kit Replicator, tiled camera sensors).  Skip for Newton GL, which renders
    asynchronously and may return the same framebuffer across consecutive steps.
    """
    frames = _read_frames(clip_path)
    stacked = np.stack(frames)  # (N, H, W, 3)
    temporal_std = stacked.std(axis=0).mean()
    assert temporal_std >= _MIN_MOTION_STD, (
        f"{label}: no motion detected (temporal std={temporal_std:.3f} < {_MIN_MOTION_STD}). "
        "Frames appear frozen or nearly identical."
    )
    print(f"  [motion ✓]    {label}: temporal_std={temporal_std:.3f}")


def _assert_clip_has_content(clip_path: str, label: str) -> None:
    """Full check: non-black + motion.  For synchronous render sources only."""
    _assert_clip_nonblack(clip_path, label)
    _assert_clip_shows_motion(clip_path, label)


# ---------------------------------------------------------------------------
# Test 1: Kit visualizer — non-tiled (PhysX)
# ---------------------------------------------------------------------------


def test_kit_nontiled_physx_records_moving_clip():
    """source='visualizer:kit' + PhysX → non-black, moving mp4 clip written to disk."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    output_dir = os.path.join(OUTPUT_DIR, "kit_nontiled")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_cfg(num_envs=1)
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]
    env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:kit", prefix="kit")]

    _run_cartpole(env_cfg)

    _assert_clip_has_content(os.path.join(output_dir, "kit_0000.mp4"), "kit-nontiled-physx")


# ---------------------------------------------------------------------------
# Test 2: Kit visualizer — tiled (PhysX, 4 envs)
# ---------------------------------------------------------------------------


def test_kit_tiled_physx_records_moving_clip():
    """source='visualizer:kit:tiled' + PhysX → tiled grid mp4 with non-black, moving content.

    Uses 4 envs so the tiled view has 4 tiles.  source='visualizer:kit:tiled' calls
    KitVisualizer.render_tiled_rgb_array(), which composes camera RGB tiles into one image.
    """
    from isaaclab_visualizers.kit import KitVisualizerCfg

    output_dir = os.path.join(OUTPUT_DIR, "kit_tiled")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_cfg(num_envs=4)
    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            window_width=320,
            window_height=240,
            tiled_cam_view=True,
            tiled_cam_num=4,
            tiled_cam_target_prim_path="/World/envs/*/Robot",
        )
    ]
    env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:kit:tiled", prefix="kit_tiled")]

    _run_cartpole(env_cfg)

    _assert_clip_has_content(os.path.join(output_dir, "kit_tiled_0000.mp4"), "kit-tiled-physx")


# ---------------------------------------------------------------------------
# Test 3: Newton visualizer — non-tiled (interactive viewport)
# ---------------------------------------------------------------------------


def test_newton_nontiled_records_nonblack_clip():
    """source='visualizer:newton' + Newton physics → non-black mp4 clip written to disk.

    Newton GL renders asynchronously in a background OpenGL thread.
    ``render_rgb_array()`` → ``get_frame()`` returns the latest framebuffer snapshot,
    which may not update on every env step, so a strict temporal-std motion check
    would be unreliable.  We verify the scene is drawn (non-black) instead.
    """
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    output_dir = os.path.join(OUTPUT_DIR, "newton_nontiled")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_cfg_newton(num_envs=1)
    env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(window_width=320, window_height=240)]
    env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:newton", prefix="newton")]

    _run_cartpole(env_cfg)

    _assert_clip_nonblack(os.path.join(output_dir, "newton_0000.mp4"), "newton-nontiled")


# ---------------------------------------------------------------------------
# Test 4: Newton visualizer — tiled-panel viewer mode (4 envs)
#
# When tiled_cam_view=True, the Newton GL window switches to showing a grid of
# per-env camera tiles.  source='visualizer:newton' captures the full GL window
# via render_rgb_array(), so the recorded clip shows the tiled panel.
#
# Note: source='visualizer:newton:tiled' would call render_tiled_rgb_array() which
# is not yet implemented on NewtonVisualizer (raises RuntimeError).  That source
# string is covered by the unit test for the error path.
# ---------------------------------------------------------------------------


def test_newton_tiled_viewer_mode_records_nonblack_clip():
    """source='visualizer:newton' with tiled_cam_view=True → tiled-panel window captured.

    Non-black check only — motion check skipped due to Newton GL async rendering.
    """
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    output_dir = os.path.join(OUTPUT_DIR, "newton_tiled_viewer")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_cfg_newton(num_envs=4)
    env_cfg.sim.visualizer_cfgs = [
        NewtonVisualizerCfg(
            window_width=320,
            window_height=240,
            tiled_cam_view=True,
            tiled_cam_num=4,
            tiled_cam_target_prim_path="/World/envs/*/Robot",
        )
    ]
    env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:newton", prefix="newton_tiled")]

    _run_cartpole(env_cfg)

    _assert_clip_nonblack(os.path.join(output_dir, "newton_tiled_0000.mp4"), "newton-tiled-viewer")


# ---------------------------------------------------------------------------
# Test 5: Sensor source — RGB, RTX renderer (PhysX)
# ---------------------------------------------------------------------------


def test_sensor_rtx_physx_records_moving_clip():
    """source='sensor:tiled_camera' + PhysX (RTX renderer) → non-black, moving sensor mp4.

    CartpoleCameraEnvCfg.default uses the RTX renderer for PhysX physics.
    The video recorder reads env.scene.sensors['tiled_camera'].data.output['rgb'].
    """
    output_dir = os.path.join(OUTPUT_DIR, "sensor_rtx_physx")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_camera_cfg_physx(num_envs=1)
    env_cfg.video_recorders = [_recorder_cfg(output_dir, "sensor:tiled_camera", prefix="sensor_rtx")]

    _run_cartpole_camera(env_cfg)

    _assert_clip_has_content(os.path.join(output_dir, "sensor_rtx_0000.mp4"), "sensor-rtx-physx")


# ---------------------------------------------------------------------------
# Test 6: Sensor source — RGB, Newton Warp renderer (Newton physics)
# ---------------------------------------------------------------------------


def test_sensor_warp_newton_records_moving_clip():
    """source='sensor:tiled_camera' + Newton physics (Warp renderer) → non-black, moving mp4.

    CartpoleCameraEnvCfg uses MultiBackendRendererCfg; under Newton physics the
    camera sensor automatically switches to the Newton Warp renderer.
    """
    output_dir = os.path.join(OUTPUT_DIR, "sensor_warp_newton")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_camera_cfg_newton(num_envs=1)
    env_cfg.video_recorders = [_recorder_cfg(output_dir, "sensor:tiled_camera", prefix="sensor_warp")]

    _run_cartpole_camera(env_cfg)

    _assert_clip_has_content(os.path.join(output_dir, "sensor_warp_0000.mp4"), "sensor-warp-newton")


# ---------------------------------------------------------------------------
# Test 7: Multiple simultaneous VideoRecorderCfg entries (Kit + sensor, PhysX)
# ---------------------------------------------------------------------------


def test_multiple_recorders_simultaneous_kit_and_sensor():
    """Two VideoRecorderCfg entries → both clips non-black and moving, written independently.

    Kit viewport (source='visualizer:kit') and tiled camera sensor
    (source='sensor:tiled_camera') record simultaneously from the same env.
    Each clip is verified independently — a failure in one source is clearly identified.
    """
    from isaaclab_visualizers.kit import KitVisualizerCfg

    output_dir = os.path.join(OUTPUT_DIR, "multi_recorder")
    shutil.rmtree(output_dir, ignore_errors=True)

    env_cfg = _cartpole_camera_cfg_physx(num_envs=1)
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]
    env_cfg.video_recorders = [
        _recorder_cfg(output_dir, "visualizer:kit", prefix="viewport"),
        _recorder_cfg(output_dir, "sensor:tiled_camera", prefix="sensor"),
    ]

    _run_cartpole_camera(env_cfg)

    _assert_clip_has_content(os.path.join(output_dir, "viewport_0000.mp4"), "multi-kit-viewport")
    _assert_clip_has_content(os.path.join(output_dir, "sensor_0000.mp4"), "multi-sensor")

    print(f"\nAll clips in {OUTPUT_DIR}/ — open with any mp4 player for visual inspection.")
