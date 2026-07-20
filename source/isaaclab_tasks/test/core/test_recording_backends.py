# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the internal video recording system.

Recording is driven by VideoRecorderCfg entries on env_cfg.video_recorders.
The env calls VideoRecorder.step() internally after each physics step, collects
frames, and flushes clips via moviepy when env.close() is called.

Source strings tested:
  "visualizer:kit"    – Kit Replicator capture from KitVisualizer.render_rgb_array()
  "visualizer:newton" – Newton GL framebuffer from NewtonVisualizer.render_rgb_array()
  "sensor:<name>"     – env.scene.sensors[name].data.output["rgb"]

Setup:
    - AppLauncher(headless=True, enable_cameras=True)
    - CartpoleEnv (DirectRLEnv, 1 env) stepped for _STEPS env steps per test.
    - VideoRecorderCfg(video_length=_CLIP, video_interval=0) → one clip per test.
Tests:
    - source="visualizer:kit" + PhysX → Kit viewport frame captured (non-black)
    - source="visualizer:kit" + Newton → logs error, no clip written
    - source="visualizer:newton" + Newton → Newton GL framebuffer (non-black)
    - source="visualizer" (auto) + PhysX + Kit visualizer → auto-selects Kit (non-black)
    - source="sensor:tiled_camera" + PhysX + camera env → sensor RGB captured (non-black)
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

pytestmark = pytest.mark.isaacsim_ci

_CLIP = 5  # short clip so tests finish quickly
_STEPS = 8  # slightly more than video_length to ensure one full clip is written
_MIN_NONZERO_RATIO = 0.005


def _assert_frames_nonempty(frames: list[np.ndarray], label: str) -> None:
    """Assert frames list is non-empty and at least one frame is non-black."""
    assert frames, f"{label}: no frames captured"
    ratios = [np.count_nonzero(f) / f.size for f in frames]
    assert any(r >= _MIN_NONZERO_RATIO for r in ratios), (
        f"{label}: all captured frames appear black (max nonzero ratio {max(ratios):.4f})"
    )


def _recorder_cfg(output_dir: str, source: str) -> VideoRecorderCfg:
    cfg = VideoRecorderCfg()
    cfg.source = source
    cfg.output_dir = output_dir
    cfg.video_length = _CLIP
    cfg.video_interval = 0
    cfg.fps = 10
    return cfg


def _run_env(env_cfg, *, steps: int = _STEPS) -> None:
    """Step the env and close it (which flushes clips)."""
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

    sim_utils.create_new_stage()
    env = CartpoleEnv(env_cfg)
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, *env.action_space.shape[1:], device=env.device)
        for _ in range(steps):
            env.step(actions)
    finally:
        env.close()


def _cartpole_cfg():
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = 1
    return cfg


# ---------------------------------------------------------------------------
# Kit visualizer sources
# ---------------------------------------------------------------------------


def test_kit_visualizer_physx_writes_clip():
    """source='visualizer:kit' + PhysX → clip written via Kit Replicator."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg()
        env_cfg.sim.physics = PhysxCfg()
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]

        # Capture the frames list passed to moviepy to verify non-black content
        # without requiring a working ffmpeg in CI.
        captured_frames: list[list[np.ndarray]] = []

        def _mock_clip(frames, fps):
            captured_frames.append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:kit")]
        with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=_mock_clip):
            _run_env(env_cfg)

        assert captured_frames, "kit-visualizer-physx: no frames captured by moviepy"
        all_frames = captured_frames[0]
        _assert_frames_nonempty(all_frames, "kit-visualizer-physx")


def test_kit_visualizer_newton_logs_error_and_produces_no_clip(caplog):
    """source='visualizer:kit' + Newton → logs an error and produces no clip.

    Kit Replicator is not supported with Newton physics. The recorder logs a
    clear error pointing to 'visualizer:newton' and captures no frames rather
    than silently writing a black video.
    """
    import logging

    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg()
        env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]

        clip_calls = []

        def _mock_clip(frames, fps):
            clip_calls.append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:kit")]
        with (
            patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=_mock_clip),
            caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"),
        ):
            _run_env(env_cfg)

        assert not clip_calls, "kit+newton should produce no clip"
        assert any("source='visualizer:newton'" in r.message for r in caplog.records), (
            "error message should suggest visualizer:newton"
        )


# ---------------------------------------------------------------------------
# Newton visualizer source
# ---------------------------------------------------------------------------


def test_newton_visualizer_writes_clip():
    """source='visualizer:newton' + Newton → Newton GL framebuffer captured."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg()
        env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
        env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(window_width=320, window_height=240)]

        captured_frames: list[list[np.ndarray]] = []

        def _mock_clip(frames, fps):
            captured_frames.append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer:newton")]
        with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=_mock_clip):
            _run_env(env_cfg)

        assert captured_frames, "newton-visualizer: no frames captured"
        _assert_frames_nonempty(captured_frames[0], "newton-visualizer")


# ---------------------------------------------------------------------------
# Auto source
# ---------------------------------------------------------------------------


def test_auto_source_picks_active_kit_visualizer():
    """source='visualizer' (default) auto-selects the first active visualizer."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = _cartpole_cfg()
        env_cfg.sim.physics = PhysxCfg()
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]

        captured_frames: list[list[np.ndarray]] = []

        def _mock_clip(frames, fps):
            captured_frames.append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        env_cfg.video_recorders = [_recorder_cfg(output_dir, "visualizer")]
        with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=_mock_clip):
            _run_env(env_cfg)

        assert captured_frames, "auto-source: no frames captured"


# ---------------------------------------------------------------------------
# Multiple simultaneous streams
# ---------------------------------------------------------------------------


def test_multiple_recorders_write_independent_clips():
    """Two VideoRecorderCfg entries write two independent clip sequences."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
        env_cfg = _cartpole_cfg()
        env_cfg.sim.physics = PhysxCfg()
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(window_width=320, window_height=240)]

        stream_frames: dict[str, list[list[np.ndarray]]] = {"a": [], "b": []}

        def _mock_clip_a(frames, fps):
            stream_frames["a"].append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        def _mock_clip_b(frames, fps):
            stream_frames["b"].append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        env_cfg.video_recorders = [
            _recorder_cfg(dir_a, "visualizer:kit"),
            _recorder_cfg(dir_b, "visualizer:kit"),
        ]

        # Patch at the module level; both recorders share the same mock.
        clip_calls: list[list[np.ndarray]] = []

        def _mock_clip(frames, fps):
            clip_calls.append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=_mock_clip):
            _run_env(env_cfg)

        # Two recorders → two clip calls (one per recorder).
        assert len(clip_calls) == 2, f"expected 2 clip calls, got {len(clip_calls)}"


# ---------------------------------------------------------------------------
# Sensor source
# ---------------------------------------------------------------------------


def test_sensor_source_captures_tiled_camera():
    """source='sensor:tiled_camera' reads RGB from the scene sensor."""
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg

    with tempfile.TemporaryDirectory() as output_dir:
        env_cfg = CartpoleCameraEnvCfg()
        env_cfg = getattr(env_cfg, "default", env_cfg)
        env_cfg.scene.num_envs = 1
        env_cfg.sim.physics = PhysxCfg()

        captured_frames: list[list[np.ndarray]] = []

        def _mock_clip(frames, fps):
            captured_frames.append(list(frames))
            mock = MagicMock()
            mock.write_videofile = MagicMock()
            return mock

        env_cfg.video_recorders = [_recorder_cfg(output_dir, "sensor:tiled_camera")]
        with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=_mock_clip):
            sim_utils.create_new_stage()
            env = CartpoleCameraEnv(env_cfg)
            try:
                env.reset()
                actions = torch.zeros(env.num_envs, *env.action_space.shape[1:], device=env.device)
                for _ in range(_STEPS):
                    env.step(actions)
            finally:
                env.close()

        assert captured_frames, "sensor-tiled-camera: no frames captured"
