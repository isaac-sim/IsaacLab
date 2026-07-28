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
    """Build a minimal args_cli namespace for apply_video_recording."""
    defaults = dict(video=True, video_length=200, video_interval=0)
    return SimpleNamespace(**{**defaults, **kwargs})


def _env_cfg():
    cfg = MagicMock()
    cfg.video_recorders = []
    return cfg


# ---------------------------------------------------------------------------
# apply_video_recording — no-op path
# ---------------------------------------------------------------------------


def test_apply_video_recording_noop_when_video_false():
    """args_cli.video=False must leave env_cfg.video_recorders untouched."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args(video=False))
    assert env_cfg.video_recorders == []


def test_apply_video_recording_noop_when_video_missing():
    """Missing video attribute (no --video flag registered) behaves like False."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", SimpleNamespace())
    assert env_cfg.video_recorders == []


# ---------------------------------------------------------------------------
# apply_video_recording — injection path
# ---------------------------------------------------------------------------


def test_apply_video_recording_injects_one_recorder():
    """video=True injects exactly one VideoRecorderCfg into env_cfg.video_recorders."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args())
    assert len(env_cfg.video_recorders) == 1


def test_apply_video_recording_source_is_visualizer():
    """The injected recorder uses source='visualizer' (auto-select active visualizer)."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args())
    assert env_cfg.video_recorders[0].source == "visualizer"


def test_apply_video_recording_maps_video_length():
    """video_length CLI flag is copied into the injected cfg."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=42))
    assert env_cfg.video_recorders[0].video_length == 42


def test_apply_video_recording_maps_video_interval():
    """video_interval CLI flag is copied into the injected cfg."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_interval=500))
    assert env_cfg.video_recorders[0].video_interval == 500


def test_apply_video_recording_missing_video_interval_defaults_to_zero():
    """If args_cli has no video_interval attribute, the cfg defaults to 0."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", SimpleNamespace(video=True, video_length=100))
    assert env_cfg.video_recorders[0].video_interval == 0


def test_apply_video_recording_output_dir_train_subdir():
    """Default subdir='train' → output_dir ends with videos/train."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args())
    expected = os.path.join("/my/log", "videos", "train")
    assert env_cfg.video_recorders[0].output_dir == expected


def test_apply_video_recording_output_dir_play_subdir():
    """subdir='play' → output_dir ends with videos/play."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args(), subdir="play")
    expected = os.path.join("/my/log", "videos", "play")
    assert env_cfg.video_recorders[0].output_dir == expected


def test_apply_video_recording_replaces_existing_recorders():
    """Calling apply_video_recording replaces any existing video_recorders list."""
    env_cfg = _env_cfg()
    env_cfg.video_recorders = ["old_entry"]
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10))
    assert len(env_cfg.video_recorders) == 1
    assert env_cfg.video_recorders[0].video_length == 10


# ---------------------------------------------------------------------------
# wrap_record_video — backwards-compat stub
# ---------------------------------------------------------------------------


def test_wrap_record_video_returns_env_unchanged():
    """wrap_record_video always returns the original env object."""
    env = MagicMock()
    result = wrap_record_video(env, "/tmp/logs", _args(video=False))
    assert result is env


def test_wrap_record_video_returns_env_unchanged_when_video_true():
    """wrap_record_video returns env even when video=True (it's a no-op stub)."""
    env = MagicMock()
    result = wrap_record_video(env, "/tmp/logs", _args(video=True))
    assert result is env


def test_wrap_record_video_logs_warning_when_video_true(caplog):
    """wrap_record_video emits a logger.warning when video=True."""
    env = MagicMock()
    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.entrypoints.common"):
        wrap_record_video(env, "/tmp/logs", _args(video=True))
    assert any("wrap_record_video" in r.message for r in caplog.records)


def test_wrap_record_video_no_warning_when_video_false(caplog):
    """wrap_record_video must not log when video=False."""
    env = MagicMock()
    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.entrypoints.common"):
        wrap_record_video(env, "/tmp/logs", _args(video=False))
    assert not caplog.records
