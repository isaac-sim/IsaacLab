# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for VideoRecorder, VideoRecorderCfg, and the ViewerCfg deprecation shim.

All tests are pure-Python mocks — no simulation context or Kit app required.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

_FRAME = np.ones((8, 12, 3), dtype=np.uint8) * 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_moviepy():
    """Stub out ImageSequenceClip so tests run without moviepy installed.

    Tests that specifically validate the ImportError path re-patch to None
    inside their own context managers, which takes precedence over this stub.
    """
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", MagicMock()):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> VideoRecorderCfg:
    defaults = dict(source="visualizer", output_dir="/tmp/test_videos", fps=30, video_length=4, video_interval=0)
    cfg = VideoRecorderCfg()
    for k, v in {**defaults, **overrides}.items():
        setattr(cfg, k, v)
    return cfg


class _FakeViz:
    def __init__(self, viz_type: str, frame: np.ndarray | None = None):
        self.cfg = SimpleNamespace(visualizer_type=viz_type)
        self._frame = frame if frame is not None else _FRAME.copy()
        self.render_calls = 0

    def render_rgb_array(self) -> np.ndarray:
        self.render_calls += 1
        return self._frame


def _make_env(visualizers=(), sensors: dict | None = None):
    env = MagicMock()
    env.sim.visualizers = list(visualizers)
    env.scene.sensors = sensors or {}
    return env


# ---------------------------------------------------------------------------
# _parse_source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("visualizer", ("visualizer", "", "")),
        ("visualizer:kit", ("visualizer", "kit", "")),
        ("visualizer:newton:tiled", ("visualizer", "newton", "tiled")),
        ("sensor:tiled_camera", ("sensor", "tiled_camera", "")),
        ("  visualizer:kit  ", ("visualizer", "kit", "")),
    ],
)
def test_parse_source(source, expected):
    assert _parse_source(source) == expected


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_init_raises_value_error_for_unknown_source_kind():
    with pytest.raises(ValueError, match="Unrecognized source kind"):
        VideoRecorder(_cfg(source="badkind:foo"), _make_env())


def test_init_raises_import_error_when_moviepy_missing():
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", None):
        with pytest.raises(ImportError, match="moviepy"):
            VideoRecorder(_cfg(), _make_env())


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


def test_trigger_one_shot_fires_once():
    """video_interval=0 → single clip starts at step 1 and never re-triggers."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=2, video_interval=0), _make_env(visualizers=[viz])
    )
    closed_count = [0]

    def counting_close():
        recorder._frames = []
        recorder._recording = False
        closed_count[0] += 1

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(8):
            recorder.step()
    assert closed_count[0] == 1


def test_trigger_recurring_fires_periodically():
    """video_interval=3 + video_length=1 → exactly 3 clips over 9 steps, first at step 1."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=1, video_interval=3), _make_env(visualizers=[viz])
    )
    close_count = [0]
    trigger_steps = []

    def counting_close():
        close_count[0] += 1
        trigger_steps.append(recorder._step_count)
        recorder._frames = []
        recorder._recording = False

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(9):
            recorder.step()
    assert close_count[0] == 3
    # First clip should trigger at step 1 (not step 3).
    assert trigger_steps[0] == 1, f"Expected first clip at step 1, got step {trigger_steps[0]}"


# ---------------------------------------------------------------------------
# Frame collection, step_offset, frame_stride
# ---------------------------------------------------------------------------


def test_step_accumulates_frames_and_closes_clip():
    """Recorder collects exactly video_length frames then calls _close_clip."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer:kit", video_length=3), _make_env(visualizers=[viz]))
    with patch.object(recorder, "_close_clip") as mock_close:
        for _ in range(3):
            recorder.step()
        assert mock_close.call_count == 1
    assert viz.render_calls == 3


def test_step_offset_delays_first_trigger():
    """step_offset=5 means the first clip starts at step 6, not step 1."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", step_offset=5, video_length=100, video_interval=0),
        _make_env(visualizers=[viz]),
    )
    for _ in range(5):
        recorder.step()
    assert not recorder._recording
    recorder.step()  # step 6 — triggers
    assert recorder._recording


def test_frame_stride_subsamples_frames():
    """frame_stride=2 captures one frame every 2 steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=4, frame_stride=2), _make_env(visualizers=[viz])
    )
    with patch.object(recorder, "_close_clip"):
        for _ in range(4):
            recorder.step()
    assert viz.render_calls == 2


# ---------------------------------------------------------------------------
# Visualizer frame routing
# ---------------------------------------------------------------------------


def test_visualizer_source_auto_picks_first_with_render_rgb_array():
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer"), _make_env(visualizers=[viz]))
    recorder.step()
    assert viz.render_calls == 1


def test_visualizer_source_auto_no_visualizer_logs_and_returns_none(caplog):
    """source='visualizer' with no visualizers logs an error once and returns None instead of raising."""
    import logging

    recorder = VideoRecorder(_cfg(source="visualizer"), _make_env(visualizers=[]))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("no recording-capable visualizer" in r.message for r in caplog.records)


def test_kit_visualizer_newton_physics_logs_warning(caplog):
    """source='visualizer:kit' with Newton physics logs a warning and attempts capture.

    With cubric the capture succeeds; without it frames may be black.  Either way
    the recorder warns and does not hard-fail.
    """
    import logging

    kit_viz = _FakeViz("kit")
    env = _make_env(visualizers=[kit_viz])
    env.sim.physics_manager.video_capture_backend.return_value = "newton_gl"

    recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)
    with caplog.at_level(logging.WARNING, logger="isaaclab.envs.utils.video_recorder"):
        recorder._get_frame()

    assert any("source='visualizer:newton'" in r.message for r in caplog.records)
    # The recorder attempts capture rather than short-circuiting.
    assert kit_viz.render_calls == 1


# ---------------------------------------------------------------------------
# Sensor frame routing
# ---------------------------------------------------------------------------


def test_sensor_source_reads_rgb():
    import torch

    rgb = torch.ones((1, 8, 12, 3), dtype=torch.uint8) * 200
    sensor = MagicMock()
    sensor.data.output = {"rgb": rgb}
    recorder = VideoRecorder(_cfg(source="sensor:tiled_camera"), _make_env(sensors={"tiled_camera": sensor}))
    frame = recorder._get_frame()
    assert frame is not None
    assert frame.shape == (8, 12, 3)


def test_sensor_source_missing_logs_and_returns_none(caplog):
    """Missing sensor logs an error (listing available sensors) and returns None instead of raising."""
    import logging

    sensors = {"tiled_camera": MagicMock()}
    recorder = VideoRecorder(_cfg(source="sensor:missing"), _make_env(sensors=sensors))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("tiled_camera" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------


def test_close_clip_writes_mp4_via_moviepy():
    frames = [_FRAME.copy(), _FRAME.copy()]
    recorder = VideoRecorder(_cfg(output_dir="/tmp/test_clips", fps=10), _make_env())
    recorder._frames = frames
    recorder._recording = True

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip) as mock_cls:
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            recorder._close_clip()

    mock_cls.assert_called_once_with(frames, fps=10)
    mock_clip.write_videofile.assert_called_once()
    assert not recorder._recording
    assert recorder._frames == []


def test_close_with_empty_frame_buffer_does_not_write():
    recorder = VideoRecorder(_cfg(), _make_env())
    recorder._recording = True
    recorder._frames = []
    mock_cls = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", mock_cls):
        recorder.close()
    mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# ViewerCfg deprecation shim
# ---------------------------------------------------------------------------


def test_viewer_cfg_warns_on_non_default_field():
    with pytest.warns(DeprecationWarning, match="ViewerCfg is deprecated"):
        ViewerCfg(eye=(1.0, 2.0, 3.0))


def test_viewer_cfg_default_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ViewerCfg()  # must not raise


# ---------------------------------------------------------------------------
# _apply_deprecated_viewer_cfg bridge
# ---------------------------------------------------------------------------


def _make_env_cfg(eye=(7.5, 7.5, 7.5)):
    viewer = ViewerCfg()
    viewer.eye = eye
    sim = SimpleNamespace(default_visualizer_cfg=None)
    return SimpleNamespace(viewer=viewer, sim=sim)


def test_apply_deprecated_viewer_sets_visualizer_cfg():
    from isaaclab.envs.common import _apply_deprecated_viewer_cfg

    env_cfg = _make_env_cfg(eye=(1.0, 2.0, 3.0))
    _apply_deprecated_viewer_cfg(env_cfg)
    assert env_cfg.sim.default_visualizer_cfg is not None
    assert env_cfg.sim.default_visualizer_cfg.eye == (1.0, 2.0, 3.0)


def test_apply_deprecated_viewer_noop_when_defaults():
    from isaaclab.envs.common import _apply_deprecated_viewer_cfg

    env_cfg = _make_env_cfg()
    _apply_deprecated_viewer_cfg(env_cfg)
    assert env_cfg.sim.default_visualizer_cfg is None


# ---------------------------------------------------------------------------
# Minor 11: asset_root / asset_body origin_type migration
# ---------------------------------------------------------------------------


def test_apply_deprecated_viewer_asset_root_migration():
    """origin_type='asset_root' → origin_type='asset' + origin_track_path=asset_name."""
    from isaaclab.envs.common import _apply_deprecated_viewer_cfg

    env_cfg = _make_env_cfg(eye=(1.0, 2.0, 3.0))
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    _apply_deprecated_viewer_cfg(env_cfg)
    cfg = env_cfg.sim.default_visualizer_cfg
    assert cfg is not None
    assert getattr(cfg, "origin_type", None) == "asset"
    assert getattr(cfg, "origin_track_path", None) == "robot"


def test_apply_deprecated_viewer_asset_body_migration():
    """origin_type='asset_body' → origin_type='asset' + origin_track_path='asset/body'."""
    from isaaclab.envs.common import _apply_deprecated_viewer_cfg

    env_cfg = _make_env_cfg(eye=(1.0, 2.0, 3.0))
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "panda_hand"
    _apply_deprecated_viewer_cfg(env_cfg)
    cfg = env_cfg.sim.default_visualizer_cfg
    assert cfg is not None
    assert getattr(cfg, "origin_type", None) == "asset"
    assert getattr(cfg, "origin_track_path", None) == "robot/panda_hand"


# ---------------------------------------------------------------------------
# Minor 12: conflict branch — default_visualizer_cfg already set
# ---------------------------------------------------------------------------


def test_apply_deprecated_viewer_skips_when_default_visualizer_cfg_already_set():
    """If sim.default_visualizer_cfg is already set, the shim logs and returns without overwriting."""
    from unittest.mock import MagicMock

    from isaaclab.envs.common import _apply_deprecated_viewer_cfg

    existing_cfg = MagicMock()
    env_cfg = _make_env_cfg(eye=(1.0, 2.0, 3.0))
    env_cfg.sim.default_visualizer_cfg = existing_cfg
    _apply_deprecated_viewer_cfg(env_cfg)
    # Must not overwrite the existing cfg.
    assert env_cfg.sim.default_visualizer_cfg is existing_cfg


# ---------------------------------------------------------------------------
# Minor 13: keep_last_n_clips pruning
# ---------------------------------------------------------------------------


def test_keep_last_n_clips_prunes_old_clips():
    """keep_last_n_clips=2 removes the oldest clip once a third is written."""

    recorder = VideoRecorder(
        _cfg(output_dir="/tmp/test_prune", keep_last_n_clips=2),
        _make_env(),
    )
    removed = []

    def fake_remove(path):
        removed.append(path)

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip):
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            with patch("isaaclab.envs.utils.video_recorder.os.remove", side_effect=fake_remove):
                for i in range(3):
                    recorder._frames = [_FRAME.copy()]
                    recorder._recording = True
                    recorder._close_clip()

    # After 3 clips with keep_last_n_clips=2, clip index 0 should be removed.
    assert any("_0000.mp4" in p for p in removed), f"Expected clip 0 to be removed, got: {removed}"


# ---------------------------------------------------------------------------
# Minor 14: partial-clip close() flush
# ---------------------------------------------------------------------------


def test_close_flushes_partial_clip():
    """close() with non-empty _frames and _recording=True flushes the clip."""
    recorder = VideoRecorder(_cfg(output_dir="/tmp/test_partial", fps=10), _make_env())
    recorder._frames = [_FRAME.copy()]
    recorder._recording = True

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip):
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            recorder.close()

    mock_clip.write_videofile.assert_called_once()
    assert not recorder._recording
    assert recorder._frames == []


# ---------------------------------------------------------------------------
# Minor 15: one-shot trigger uses mock _close_clip (no disk I/O)
# ---------------------------------------------------------------------------


def test_trigger_one_shot_fires_once_no_disk_io():
    """video_interval=0 → single clip starts at step 1; _close_clip called exactly once (mocked)."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=2, video_interval=0), _make_env(visualizers=[viz])
    )
    close_count = [0]

    def counting_close():
        recorder._frames = []
        recorder._recording = False
        close_count[0] += 1

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(8):
            recorder.step()

    assert close_count[0] == 1
