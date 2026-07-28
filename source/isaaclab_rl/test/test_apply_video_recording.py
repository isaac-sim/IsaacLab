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
    defaults = dict(video=True, video_length=200, video_interval=0)
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
    """video=True injects one recorder with the right source, length, interval, and output_dir."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args(video_length=42, video_interval=500), subdir="play")
    assert len(env_cfg.video_recorders) == 1
    rec = env_cfg.video_recorders[0]
    assert rec.source == "visualizer"
    assert rec.video_length == 42
    assert rec.video_interval == 500
    assert rec.output_dir == os.path.join("/my/log", "videos", "play")


def test_apply_video_recording_replaces_existing_recorders():
    """Calling apply_video_recording replaces any pre-existing video_recorders list."""
    env_cfg = _env_cfg()
    env_cfg.video_recorders = ["old_entry"]
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10))
    assert len(env_cfg.video_recorders) == 1
    assert env_cfg.video_recorders[0].video_length == 10


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
