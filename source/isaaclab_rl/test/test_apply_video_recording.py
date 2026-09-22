# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for video-recording entrypoint helpers."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest
from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_rl.entrypoints.common import apply_video_recording, video_playback_steps, wrap_record_video


def _args(**kwargs: object) -> SimpleNamespace:
    defaults = dict(video=True, video_length=None, video_interval=None)
    return SimpleNamespace(**{**defaults, **kwargs})


def test_apply_video_recording_noop_when_video_false():
    """Video recording remains disabled when the CLI flag is false or absent."""
    env_cfg = ManagerBasedRLEnvCfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args(video=False))
    assert env_cfg.video_recorders == []

    apply_video_recording(env_cfg, "/tmp/logs", SimpleNamespace())
    assert env_cfg.video_recorders == []


def test_apply_video_recording_injects_correct_recorder():
    """Video recording creates a default recorder and headless Kit visualizer."""
    env_cfg = ManagerBasedRLEnvCfg()
    apply_video_recording(env_cfg, "/my/log", _args(video_length=42, video_interval=500), subdir="play")
    assert len(env_cfg.video_recorders) == 1
    recorder = env_cfg.video_recorders[0]
    assert recorder.source == "visualizer:kit"
    assert recorder.video_length == 42
    assert recorder.video_interval == 500
    assert recorder.output_dir == os.path.join("/my/log", "videos", "play")
    assert recorder.output_filename_prefix == "clip"
    assert len(env_cfg.sim.visualizer_cfgs) == 1
    assert isinstance(env_cfg.sim.visualizer_cfgs[0], KitVisualizerCfg)
    assert env_cfg.sim.visualizer_cfgs[0].headless


@pytest.mark.parametrize(
    ("existing_prefix", "checkpoint_name", "expected_prefix"),
    [
        ("clip", "model_1200.pt", "clip_model_1200"),
        ("eval", "model_42.pt", "eval_model_42"),
        ("clip_model_1200", "model_120.pt", "clip_model_1200_model_120"),
        ("clip", "custom_1200.pt", "clip"),
        ("clip", "final.pt", "clip"),
    ],
)
def test_apply_video_recording_labels_play_video_with_checkpoint_stem(
    existing_prefix: str, checkpoint_name: str, expected_prefix: str
):
    """Play videos append numeric model checkpoint stems as distinct tokens."""
    existing = VideoRecorderCfg()
    existing.output_filename_prefix = existing_prefix

    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/my/log", _args(), subdir="play", checkpoint_path=f"/my/log/{checkpoint_name}")

    assert env_cfg.video_recorders[0].output_filename_prefix == expected_prefix


def test_apply_video_recording_leaves_train_video_prefix_unchanged():
    """Checkpoint labels are only applied to play videos, not training videos."""
    env_cfg = ManagerBasedRLEnvCfg()
    apply_video_recording(env_cfg, "/my/log", _args(), checkpoint_path="/my/log/model_1200.pt")

    assert env_cfg.video_recorders[0].output_filename_prefix == "clip"


def test_apply_video_recording_uses_cfg_defaults_when_cli_not_passed():
    """Unset CLI options preserve the recorder length and historical interval."""
    defaults = VideoRecorderCfg()
    env_cfg = ManagerBasedRLEnvCfg()
    apply_video_recording(env_cfg, "/my/log", _args())
    recorder = env_cfg.video_recorders[0]
    assert recorder.video_length == defaults.video_length
    assert recorder.video_interval == 2000


def test_apply_video_recording_patches_existing_recorders():
    """CLI length and interval overrides preserve other recorder settings."""
    existing = VideoRecorderCfg()
    existing.source = "sensor:tiled_camera"
    existing.output_dir = "/my/custom/path"
    existing.fps = 60

    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10, video_interval=500))

    assert len(env_cfg.video_recorders) == 1
    recorder = env_cfg.video_recorders[0]
    assert recorder.source == "sensor:tiled_camera"
    assert recorder.output_dir == "/my/custom/path"
    assert recorder.fps == 60
    assert recorder.video_length == 10
    assert recorder.video_interval == 500


def test_apply_video_recording_rejects_viz_none_with_video():
    """An explicitly disabled visualizer is incompatible with video recording."""
    env_cfg = ManagerBasedRLEnvCfg()

    with pytest.raises(ValueError, match="--video is not compatible with --viz none"):
        apply_video_recording(env_cfg, "/my/log", _args(visualizer=None, visualizer_explicit=True))


@pytest.mark.parametrize("no_capture_viz", ["rerun", "viser"])
def test_apply_video_recording_rejects_no_capture_visualizers(no_capture_viz: str):
    """Video recording rejects visualizers without frame capture."""
    env_cfg = ManagerBasedRLEnvCfg()

    with pytest.raises(ValueError, match="--video is not supported"):
        apply_video_recording(env_cfg, "/my/log", _args(visualizer=[no_capture_viz]))


@pytest.mark.parametrize(
    ("visualizers", "expected_source"),
    [(["rerun", "kit"], "visualizer:kit"), (["newton_rtx"], "visualizer:newton_rtx")],
)
def test_apply_video_recording_uses_requested_capture_visualizer(visualizers: list[str], expected_source: str) -> None:
    """Video recording selects the first capture-capable visualizer."""
    env_cfg = ManagerBasedRLEnvCfg()

    apply_video_recording(env_cfg, "/my/log", _args(visualizer=visualizers))

    assert len(env_cfg.video_recorders) == 1
    assert env_cfg.video_recorders[0].source == expected_source


def test_wrap_record_video_is_noop_stub(caplog: pytest.LogCaptureFixture) -> None:
    """The compatibility stub warns only when video recording is requested."""
    env = object()

    result = wrap_record_video(env, "/tmp/logs", _args(video=False))
    assert result is env
    assert not caplog.records

    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.entrypoints.common"):
        result = wrap_record_video(env, "/tmp/logs", _args(video=True))
    assert result is env
    assert any("wrap_record_video" in r.message for r in caplog.records)


def test_video_playback_steps_waits_for_every_recorder():
    """Playback runs until the recorder whose first clip ends last has finished it."""
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [
        VideoRecorderCfg(video_length=100),
        VideoRecorderCfg(video_length=50, step_offset=80),
        VideoRecorderCfg(video_length=120, step_offset=5),
    ]
    expected = max(cfg.step_offset + cfg.video_length for cfg in env_cfg.video_recorders)

    assert video_playback_steps(_args(), env_cfg) == expected


def test_video_playback_steps_keeps_step_offset_with_cli_length():
    """A ``--video_length`` override still waits for a recorder's configured step offset."""
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", step_offset=30)]

    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10))

    assert video_playback_steps(_args(video_length=10), env_cfg) == 30 + 10


def test_video_playback_steps_is_unbounded_without_video():
    """Playback is unbounded when video is not requested or no recorder is configured."""
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [VideoRecorderCfg()]
    assert video_playback_steps(_args(video=False), env_cfg) is None
    assert video_playback_steps(_args(), ManagerBasedRLEnvCfg()) is None
