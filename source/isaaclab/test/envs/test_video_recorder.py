# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for VideoRecorder and VideoRecorderCfg.

All tests are pure-Python mocks — no simulation context or Kit app required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

_FRAME = np.ones((8, 12, 3), dtype=np.uint8) * 128


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
    """Minimal visualizer stub with render_rgb_array support."""

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


def test_parse_source_bare_visualizer():
    assert _parse_source("visualizer") == ("visualizer", "", "")


def test_parse_source_typed_visualizer():
    assert _parse_source("visualizer:kit") == ("visualizer", "kit", "")


def test_parse_source_visualizer_with_sub():
    assert _parse_source("visualizer:newton:tiled") == ("visualizer", "newton", "tiled")


def test_parse_source_sensor():
    assert _parse_source("sensor:tiled_camera") == ("sensor", "tiled_camera", "")


def test_parse_source_strips_whitespace():
    assert _parse_source("  visualizer:kit  ") == ("visualizer", "kit", "")


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_init_raises_value_error_for_unknown_source_kind():
    """Unrecognized source kind raises ValueError at construction time."""
    with pytest.raises(ValueError, match="Unrecognized source kind"):
        VideoRecorder(_cfg(source="badkind:foo"), _make_env())


def test_init_raises_import_error_when_moviepy_missing():
    """Missing moviepy raises ImportError at construction time."""
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", None):
        with pytest.raises(ImportError, match="moviepy"):
            VideoRecorder(_cfg(), _make_env())


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


def test_trigger_step_zero_fires_at_first_step():
    """video_interval=0 → single clip starts at step 1."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=100, video_interval=0), _make_env(visualizers=[viz])
    )
    assert not recorder._recording
    recorder.step()
    assert recorder._recording


def test_trigger_step_zero_does_not_retrigger():
    """video_interval=0 → single clip, no re-trigger after it closes."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=2, video_interval=0), _make_env(visualizers=[viz])
    )
    closed_count = [0]

    original_close = recorder._close_clip

    def counting_close():
        original_close()
        closed_count[0] += 1

    recorder._close_clip = counting_close
    for _ in range(8):
        recorder.step()
    assert closed_count[0] == 1  # only one clip, not re-triggered


def test_trigger_step_positive_fires_periodically():
    """video_interval=3 + video_length=1 → exactly 3 clips written over 9 steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=1, video_interval=3), _make_env(visualizers=[viz])
    )
    close_count = [0]

    def counting_close():
        close_count[0] += 1
        recorder._frames = []
        recorder._recording = False

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(9):
            recorder.step()

    assert close_count[0] == 3


# ---------------------------------------------------------------------------
# Frame collection
# ---------------------------------------------------------------------------


def test_step_accumulates_frames_until_video_length():
    """Recorder accumulates frames and closes clip after video_length steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer:kit", video_length=3), _make_env(visualizers=[viz]))

    with patch.object(recorder, "_close_clip") as mock_close:
        for _ in range(3):
            recorder.step()
        assert mock_close.call_count == 1

    assert viz.render_calls == 3


def test_step_does_not_collect_when_no_trigger():
    """No frames collected before the first trigger."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer:kit", video_interval=10), _make_env(visualizers=[viz]))
    for _ in range(5):
        recorder.step()
    assert viz.render_calls == 0


# ---------------------------------------------------------------------------
# step_offset
# ---------------------------------------------------------------------------


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
    recorder.step()  # step 6 — offset cleared, effective step 1 → triggers
    assert recorder._recording


def test_step_offset_applied_to_recurring_clips():
    """step_offset shifts the whole cadence — first trigger at offset+interval."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", step_offset=10, video_length=1, video_interval=5),
        _make_env(visualizers=[viz]),
    )
    close_count = [0]

    def counting_close():
        close_count[0] += 1
        recorder._frames = []
        recorder._recording = False

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(25):  # steps 1-25; effective steps 1-15 after offset 10
            recorder.step()

    # effective steps 5, 10, 15 → 3 triggers
    assert close_count[0] == 3


# ---------------------------------------------------------------------------
# frame_stride
# ---------------------------------------------------------------------------


def test_frame_stride_subsamples_frames():
    """frame_stride=2 captures one frame every 2 steps — half the frames for the same clip duration."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=4, frame_stride=2),
        _make_env(visualizers=[viz]),
    )
    with patch.object(recorder, "_close_clip"):
        for _ in range(4):
            recorder.step()
    assert viz.render_calls == 2  # 4 steps / stride 2


def test_frame_stride_one_is_default_behaviour():
    """frame_stride=1 captures every step (default)."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=4, frame_stride=1),
        _make_env(visualizers=[viz]),
    )
    with patch.object(recorder, "_close_clip"):
        for _ in range(4):
            recorder.step()
    assert viz.render_calls == 4


# ---------------------------------------------------------------------------
# Multiple simultaneous recorders
# ---------------------------------------------------------------------------


def test_multiple_recorders_independent_sources():
    """Two recorders with different sources each call their own visualizer."""
    kit_viz = _FakeViz("kit")
    newton_viz = _FakeViz("newton")
    rec_kit = VideoRecorder(_cfg(source="visualizer:kit", video_length=3), _make_env(visualizers=[kit_viz, newton_viz]))
    rec_newton = VideoRecorder(
        _cfg(source="visualizer:newton", video_length=3), _make_env(visualizers=[kit_viz, newton_viz])
    )
    with patch.object(rec_kit, "_close_clip"), patch.object(rec_newton, "_close_clip"):
        for _ in range(3):
            rec_kit.step()
            rec_newton.step()
    assert kit_viz.render_calls == 3
    assert newton_viz.render_calls == 3


# ---------------------------------------------------------------------------
# Visualizer frame routing
# ---------------------------------------------------------------------------


def test_visualizer_source_auto_picks_first_with_render_rgb_array():
    """source='visualizer' picks the first active visualizer that has render_rgb_array."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer"), _make_env(visualizers=[viz]))
    recorder.step()
    assert viz.render_calls == 1


def test_visualizer_source_typed_filters_by_type():
    """source='visualizer:newton' only calls the Newton visualizer."""
    kit_viz = _FakeViz("kit")
    newton_viz = _FakeViz("newton")
    recorder = VideoRecorder(_cfg(source="visualizer:newton"), _make_env(visualizers=[kit_viz, newton_viz]))
    recorder.step()
    assert kit_viz.render_calls == 0
    assert newton_viz.render_calls == 1


def test_visualizer_source_auto_no_visualizer_raises():
    """source='visualizer' with no active visualizers raises RuntimeError on first frame."""
    recorder = VideoRecorder(_cfg(source="visualizer"), _make_env(visualizers=[]))
    with pytest.raises(RuntimeError, match="no recording-capable visualizer"):
        recorder._get_frame()


def test_visualizer_source_typed_missing_raises():
    """source='visualizer:kit' with only Newton active raises RuntimeError listing active types."""
    recorder = VideoRecorder(_cfg(source="visualizer:kit"), _make_env(visualizers=[_FakeViz("newton")]))
    with pytest.raises(RuntimeError, match="newton"):
        recorder._get_frame()


def test_kit_visualizer_newton_physics_logs_error_and_returns_none(caplog):
    """source='visualizer:kit' with Newton physics logs an error and returns None — no fallback."""
    import logging

    kit_viz = _FakeViz("kit")
    env = _make_env(visualizers=[kit_viz])
    env.sim.physics_manager.video_capture_backend.return_value = "newton_gl"

    recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()

    assert frame is None
    assert kit_viz.render_calls == 0
    assert any("source='visualizer:newton'" in r.message for r in caplog.records)


def test_visualizer_tiled_calls_render_tiled_rgb_array():
    """source='visualizer:newton:tiled' (or kit:tiled) calls render_tiled_rgb_array() if present."""
    for viz_type in ("newton", "kit"):
        viz = _FakeViz(viz_type)
        tiled_frame = np.ones((16, 24, 3), dtype=np.uint8) * 64
        viz.render_tiled_rgb_array = MagicMock(return_value=tiled_frame)
        recorder = VideoRecorder(_cfg(source=f"visualizer:{viz_type}:tiled"), _make_env(visualizers=[viz]))
        frame = recorder._get_frame()
        viz.render_tiled_rgb_array.assert_called_once()
        assert frame is tiled_frame


def test_visualizer_tiled_raises_when_not_supported():
    """source='visualizer:kit:tiled' raises RuntimeError when visualizer has no tiled capture."""
    viz = _FakeViz("kit")  # no render_tiled_rgb_array
    recorder = VideoRecorder(_cfg(source="visualizer:kit:tiled"), _make_env(visualizers=[viz]))
    with pytest.raises(RuntimeError, match="tiled"):
        recorder._get_frame()


# ---------------------------------------------------------------------------
# Sensor frame routing
# ---------------------------------------------------------------------------


def test_sensor_source_reads_rgb():
    """source='sensor:tiled_camera' reads the rgb channel from the sensor."""
    import torch

    rgb = torch.ones((1, 8, 12, 3), dtype=torch.uint8) * 200
    sensor = MagicMock()
    sensor.data.output = {"rgb": rgb}
    recorder = VideoRecorder(_cfg(source="sensor:tiled_camera"), _make_env(sensors={"tiled_camera": sensor}))
    frame = recorder._get_frame()
    assert frame is not None
    assert frame.shape == (8, 12, 3)


def test_sensor_source_missing_raises_with_available_list():
    """source='sensor:missing' raises RuntimeError listing available sensors."""
    sensors = {"tiled_camera": MagicMock(), "wrist_cam": MagicMock()}
    recorder = VideoRecorder(_cfg(source="sensor:missing"), _make_env(sensors=sensors))
    with pytest.raises(RuntimeError, match="missing") as exc_info:
        recorder._get_frame()
    assert "tiled_camera" in str(exc_info.value)


def test_sensor_source_missing_suggests_camera_when_none_exist():
    """source='sensor:cam' with no sensors at all includes a hint to add a CameraCfg."""
    recorder = VideoRecorder(_cfg(source="sensor:cam"), _make_env(sensors={}))
    with pytest.raises(RuntimeError, match="CameraCfg"):
        recorder._get_frame()


def test_sensor_source_no_rgb_output_raises():
    """Sensor found but missing 'rgb' in output raises RuntimeError."""
    sensor = MagicMock()
    sensor.data.output = {"depth": MagicMock()}  # no rgb key
    recorder = VideoRecorder(_cfg(source="sensor:cam"), _make_env(sensors={"cam": sensor}))
    with pytest.raises(RuntimeError, match="no 'rgb' output"):
        recorder._get_frame()


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------


def test_output_filename_prefix_used_in_clip_name():
    """Clip filenames use output_filename_prefix instead of the hardcoded 'clip' default."""
    recorder = VideoRecorder(_cfg(output_dir="/tmp/vids", output_filename_prefix="viewport"), _make_env())
    assert recorder._clip_path(0) == "/tmp/vids/viewport_0000.mp4"
    assert recorder._clip_path(3) == "/tmp/vids/viewport_0003.mp4"


def test_keep_last_n_clips_deletes_old_clips():
    """keep_last_n_clips=2 removes clips older than the last 2 after each write."""
    recorder = VideoRecorder(_cfg(output_dir="/tmp/vids", keep_last_n_clips=2), _make_env())
    recorder._clip_index = 3  # simulate having written 3 clips already

    removed = []
    with patch("isaaclab.envs.utils.video_recorder.os.remove", side_effect=lambda p: removed.append(p)):
        recorder._maybe_delete_old_clips()

    # clip_index=3, keep_last=2 → cutoff=1 → delete index 0 only
    assert len(removed) == 1
    assert removed[0].endswith("clip_0000.mp4")


def test_keep_last_n_clips_none_deletes_nothing():
    """keep_last_n_clips=None (default) never deletes any clips."""
    recorder = VideoRecorder(_cfg(keep_last_n_clips=None), _make_env())
    recorder._clip_index = 10
    with patch("isaaclab.envs.utils.video_recorder.os.remove") as mock_rm:
        recorder._maybe_delete_old_clips()
    mock_rm.assert_not_called()


def test_close_clip_writes_mp4_via_moviepy():
    """_close_clip() passes collected frames to moviepy and writes a file."""
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


def test_close_clip_increments_clip_index():
    """Each clip gets a unique incremented index."""
    recorder = VideoRecorder(_cfg(), _make_env())
    recorder._frames = [_FRAME]
    recorder._recording = True

    with (
        patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip"),
        patch("isaaclab.envs.utils.video_recorder.os.makedirs"),
    ):
        recorder._close_clip()
        recorder._frames = [_FRAME]
        recorder._recording = True
        recorder._close_clip()

    assert recorder._clip_index == 2


def test_close_flushes_partial_clip():
    """VideoRecorder.close() writes any buffered frames before teardown."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(video_length=100), _make_env(visualizers=[viz]))
    recorder._frames = [_FRAME, _FRAME]
    recorder._recording = True

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip):
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            recorder.close()

    mock_clip.write_videofile.assert_called_once()


# ---------------------------------------------------------------------------
# Additional _parse_source edge cases
# ---------------------------------------------------------------------------


def test_parse_source_extra_segments_truncated_to_three():
    """Extra colon-delimited segments beyond the third are silently ignored."""
    from isaaclab.envs.utils.video_recorder import _parse_source

    kind, type_or_name, sub = _parse_source("visualizer:kit:tiled:extra")
    assert kind == "visualizer"
    assert type_or_name == "kit"
    assert sub == "tiled"


# ---------------------------------------------------------------------------
# Trigger edge cases
# ---------------------------------------------------------------------------


def test_trigger_interval_one_fires_every_step():
    """video_interval=1 with video_length=1 fires a new clip on every step."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(video_length=1, video_interval=1), _make_env(visualizers=[viz]))

    close_calls = 0

    def _fake_close():
        nonlocal close_calls
        close_calls += 1
        recorder._recording = False
        recorder._frames = []

    recorder._close_clip = _fake_close

    with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
        for _ in range(5):
            recorder.step()

    assert close_calls == 5


# ---------------------------------------------------------------------------
# close() with empty frame buffer
# ---------------------------------------------------------------------------


def test_close_with_empty_frame_buffer_does_not_write():
    """close() must not call moviepy when _frames is empty (e.g. all None frames)."""
    recorder = VideoRecorder(_cfg(), _make_env())
    recorder._recording = True
    recorder._frames = []

    mock_cls = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", mock_cls):
        recorder.close()

    mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _maybe_delete_old_clips error tolerance
# ---------------------------------------------------------------------------


def test_maybe_delete_old_clips_tolerates_missing_file():
    """FileNotFoundError from os.remove is swallowed silently."""
    recorder = VideoRecorder(_cfg(keep_last_n_clips=1), _make_env())
    recorder._clip_index = 3

    with patch("isaaclab.envs.utils.video_recorder.os.remove", side_effect=FileNotFoundError):
        recorder._maybe_delete_old_clips()  # must not raise
