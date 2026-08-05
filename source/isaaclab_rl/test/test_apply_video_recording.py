# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for apply_video_recording and wrap_record_video.

Pure Python — no simulation context or Kit app required.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from isaaclab_rl.entrypoints.common import apply_video_recording, wrap_record_video


def _args(**kwargs) -> SimpleNamespace:
    # video_length and video_interval default to None (not passed at CLI)
    defaults = dict(video=True, video_length=None, video_interval=None)
    return SimpleNamespace(**{**defaults, **kwargs})


def _env_cfg():
    cfg = MagicMock()
    cfg.video_recorders = []
    return cfg


def test_apply_video_recording_noop_when_video_false():
    """video=False (or missing) leaves env_cfg.video_recorders untouched."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args(video=False))
    assert env_cfg.video_recorders == []

    apply_video_recording(env_cfg, "/tmp/logs", SimpleNamespace())
    assert env_cfg.video_recorders == []


def test_apply_video_recording_injects_correct_recorder():
    """video=True with no pre-configured recorders creates a default with log_dir output_dir."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args(video_length=42, video_interval=500), subdir="play")
    assert len(env_cfg.video_recorders) == 1
    defaults = VideoRecorderCfg()
    rec = env_cfg.video_recorders[0]
    assert rec.source == defaults.source  # default source preserved
    assert rec.video_length == 42  # CLI override applied
    assert rec.video_interval == 500  # CLI override applied
    assert rec.output_dir == os.path.join("/my/log", "videos", "play")


def test_apply_video_recording_uses_cfg_defaults_when_cli_not_passed():
    """video=True without --video_length/--video_interval uses historical CLI cadence."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    defaults = VideoRecorderCfg()
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args())  # no video_length/interval
    rec = env_cfg.video_recorders[0]
    assert rec.video_length == defaults.video_length  # default kept
    # CLI fallback uses video_interval=2000 to match historical --video cadence
    # (record immediately then every 2000 steps), not VideoRecorderCfg()'s default of 0.
    assert rec.video_interval == 2000


def test_apply_video_recording_patches_existing_recorders():
    """Existing recorders are kept; only video_length and video_interval are overwritten."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    existing = VideoRecorderCfg()
    existing.source = "sensor:tiled_camera"
    existing.output_dir = "/my/custom/path"
    existing.fps = 60

    env_cfg = _env_cfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10, video_interval=500))

    # existing recorder is kept — not replaced
    assert len(env_cfg.video_recorders) == 1
    rec = env_cfg.video_recorders[0]
    assert rec.source == "sensor:tiled_camera"  # preserved
    assert rec.output_dir == "/my/custom/path"  # preserved
    assert rec.fps == 60  # preserved
    assert rec.video_length == 10  # CLI override applied
    assert rec.video_interval == 500  # CLI override applied


def test_wrap_record_video_is_noop_stub(caplog):
    """wrap_record_video returns the env unchanged and warns only when video=True."""
    env = MagicMock()

    result = wrap_record_video(env, "/tmp/logs", _args(video=False))
    assert result is env
    assert not caplog.records

    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.entrypoints.common"):
        result = wrap_record_video(env, "/tmp/logs", _args(video=True))
    assert result is env
    assert any("wrap_record_video" in r.message for r in caplog.records)
