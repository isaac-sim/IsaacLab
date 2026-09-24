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


@pytest.mark.parametrize(
    ("video_length", "video_interval", "expected_length", "expected_interval"),
    [(42, 500, 42, 500), (None, None, VideoRecorderCfg().video_length, 2000)],
    ids=["cli_overrides", "cfg_defaults"],
)
def test_apply_video_recording_creates_default_recorder(
    video_length: int | None, video_interval: int | None, expected_length: int, expected_interval: int
):
    """Without declared recorders, one headless Kit recorder is created, taking CLI values over defaults."""
    env_cfg = ManagerBasedRLEnvCfg()
    apply_video_recording(
        env_cfg, "/my/log", _args(video_length=video_length, video_interval=video_interval), subdir="play"
    )
    assert len(env_cfg.video_recorders) == 1
    recorder = env_cfg.video_recorders[0]
    assert recorder.source == "visualizer:kit"
    assert (recorder.video_length, recorder.video_interval) == (expected_length, expected_interval)
    assert recorder.output_dir == os.path.join("/my/log", "videos", "play")
    assert recorder.output_filename_prefix == "clip"
    assert len(env_cfg.sim.visualizer_cfgs) == 1
    assert isinstance(env_cfg.sim.visualizer_cfgs[0], KitVisualizerCfg)
    assert env_cfg.sim.visualizer_cfgs[0].headless


@pytest.mark.parametrize(
    ("subdir", "existing_prefix", "checkpoint_name", "expected_prefix"),
    [
        ("play", "clip", "model_1200.pt", "clip_model_1200"),
        ("play", "eval", "model_42.pt", "eval_model_42"),
        ("play", "clip_model_1200", "model_120.pt", "clip_model_1200_model_120"),
        ("play", "clip", "custom_1200.pt", "clip"),
        ("play", "clip", "final.pt", "clip"),
        ("train", "clip", "model_1200.pt", "clip"),
    ],
)
def test_apply_video_recording_labels_play_video_with_checkpoint_stem(
    subdir: str, existing_prefix: str, checkpoint_name: str, expected_prefix: str
):
    """Play videos append numeric model checkpoint stems as distinct tokens; training videos keep their prefix."""
    existing = VideoRecorderCfg()
    existing.output_filename_prefix = existing_prefix

    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/my/log", _args(), subdir=subdir, checkpoint_path=f"/my/log/{checkpoint_name}")

    assert env_cfg.video_recorders[0].output_filename_prefix == expected_prefix


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


@pytest.mark.parametrize(
    ("visualizer_args", "message"),
    [
        (dict(visualizer=None, visualizer_explicit=True), "--video is not compatible with --viz none"),
        (dict(visualizer=["rerun"]), "--video is not supported"),
        (dict(visualizer=["viser"]), "--video is not supported"),
    ],
)
def test_apply_video_recording_rejects_visualizers_without_capture(visualizer_args: dict, message: str):
    """Video recording rejects a disabled visualizer and visualizers without frame capture."""
    with pytest.raises(ValueError, match=message):
        apply_video_recording(ManagerBasedRLEnvCfg(), "/my/log", _args(**visualizer_args))


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


@pytest.mark.parametrize(
    ("recorders", "cli_args", "expected_steps"),
    [
        # (video_length, step_offset) per recorder; the second recorder's first clip ends last, at 80 + 50
        ([(100, 0), (50, 80), (120, 5)], dict(), 130),
        # --video_length replaces the length but the configured offset still delays the clip
        ([(200, 30)], dict(video_length=10), 40),
        ([(200, 0)], dict(video=False), None),
    ],
    ids=["waits_for_last_recorder", "cli_length_keeps_offset", "unbounded_without_video"],
)
def test_video_playback_steps(recorders: list[tuple[int, int]], cli_args: dict, expected_steps: int | None):
    """Playback runs until every recorder has finished its first clip, and is unbounded without --video."""
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:kit", video_length=length, step_offset=offset)
        for length, offset in recorders
    ]
    args = _args(**cli_args)
    apply_video_recording(env_cfg, "/tmp/logs", args)

    assert video_playback_steps(args, env_cfg) == expected_steps
