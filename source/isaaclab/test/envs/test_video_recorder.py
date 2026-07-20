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

from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

_FRAME = np.ones((8, 12, 3), dtype=np.uint8) * 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> VideoRecorderCfg:
    defaults = dict(source="visualizer", output_dir="/tmp/test_videos", fps=30, clip_length=4, clip_trigger_step=0)
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
    assert _parse_source("visualizer:newton/tiled") == ("visualizer", "newton", "tiled")


def test_parse_source_sensor():
    assert _parse_source("sensor:tiled_camera") == ("sensor", "tiled_camera", "")


def test_parse_source_sensor_with_data_type():
    assert _parse_source("sensor:wrist_cam/depth") == ("sensor", "wrist_cam", "depth")


def test_parse_source_strips_whitespace():
    assert _parse_source("  visualizer:kit  ") == ("visualizer", "kit", "")


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


def test_trigger_step_zero_fires_at_first_step():
    """clip_trigger_step=0 → single clip starts at step 1."""
    recorder = VideoRecorder(_cfg(clip_length=100, clip_trigger_step=0), _make_env())
    assert not recorder._recording
    recorder.step()
    assert recorder._recording


def test_trigger_step_zero_does_not_retrigger():
    """clip_trigger_step=0 → single clip, no re-trigger after it closes."""
    viz = _FakeViz("kit")  # provide frames so clip actually closes
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", clip_length=2, clip_trigger_step=0), _make_env(visualizers=[viz])
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
    """clip_trigger_step=3 + clip_length=1 → exactly 3 clips written over 9 steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", clip_length=1, clip_trigger_step=3), _make_env(visualizers=[viz])
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


def test_step_accumulates_frames_until_clip_length():
    """Recorder accumulates frames and closes clip after clip_length steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer:kit", clip_length=3), _make_env(visualizers=[viz]))

    with patch.object(recorder, "_close_clip") as mock_close:
        for _ in range(3):
            recorder.step()
        # Clip should close after 3 frames.
        assert mock_close.call_count == 1

    assert viz.render_calls == 3


def test_step_does_not_collect_when_no_trigger():
    """No frames collected before the first trigger."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(_cfg(source="visualizer:kit", clip_trigger_step=10), _make_env(visualizers=[viz]))
    for _ in range(5):
        recorder.step()
    assert viz.render_calls == 0


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


def test_visualizer_source_missing_logs_error_and_returns_none(caplog):
    """source='visualizer:kit' with no matching active visualizer logs an error with helpful context."""
    import logging

    recorder = VideoRecorder(_cfg(source="visualizer:kit"), _make_env(visualizers=[_FakeViz("newton")]))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("source='visualizer:kit'" in r.message for r in caplog.records)
    assert any("newton" in r.message for r in caplog.records)  # active types listed


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
    assert kit_viz.render_calls == 0  # Kit was never called
    assert any("source='visualizer:newton'" in r.message for r in caplog.records)


def test_visualizer_tiled_calls_render_tiled_rgb_array():
    """source='visualizer:newton/tiled' calls render_tiled_rgb_array() if present."""
    viz = _FakeViz("newton")
    tiled_frame = np.ones((16, 24, 3), dtype=np.uint8) * 64
    viz.render_tiled_rgb_array = MagicMock(return_value=tiled_frame)
    recorder = VideoRecorder(_cfg(source="visualizer:newton/tiled"), _make_env(visualizers=[viz]))
    frame = recorder._get_frame()
    viz.render_tiled_rgb_array.assert_called_once()
    assert frame is tiled_frame


def test_visualizer_tiled_logs_warning_when_not_supported(caplog):
    """source='visualizer:kit/tiled' warns when the visualizer has no tiled capture."""
    import logging

    viz = _FakeViz("kit")  # no render_tiled_rgb_array
    recorder = VideoRecorder(_cfg(source="visualizer:kit/tiled"), _make_env(visualizers=[viz]))
    with caplog.at_level(logging.WARNING, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("tiled" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Sensor frame routing
# ---------------------------------------------------------------------------


def test_sensor_source_reads_rgb_by_default():
    """source='sensor:tiled_camera' reads the rgb channel from the sensor."""
    import torch

    rgb = torch.ones((1, 8, 12, 3), dtype=torch.uint8) * 200
    sensor = MagicMock()
    sensor.data.output = {"rgb": rgb}  # plain Tensor — no .torch wrapper needed
    recorder = VideoRecorder(_cfg(source="sensor:tiled_camera"), _make_env(sensors={"tiled_camera": sensor}))
    frame = recorder._get_frame()
    assert frame is not None
    assert frame.shape == (8, 12, 3)


def test_sensor_source_reads_specified_data_type():
    """source='sensor:wrist_cam/depth' reads the depth channel."""
    import torch

    depth = torch.ones((1, 4, 6, 1), dtype=torch.float32) * 2.5
    sensor = MagicMock()
    sensor.data.output = {"depth": depth}
    recorder = VideoRecorder(_cfg(source="sensor:wrist_cam/depth"), _make_env(sensors={"wrist_cam": sensor}))
    frame = recorder._get_frame()
    assert frame is not None


def test_sensor_source_missing_logs_error_with_available_list(caplog):
    """source='sensor:missing' logs an error listing available sensors."""
    import logging

    sensors = {"tiled_camera": MagicMock(), "wrist_cam": MagicMock()}
    recorder = VideoRecorder(_cfg(source="sensor:missing"), _make_env(sensors=sensors))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("missing" in r.message for r in caplog.records)
    assert any("tiled_camera" in r.message for r in caplog.records)  # available sensors listed


def test_sensor_source_missing_suggests_adding_camera_when_none_exist(caplog):
    """source='sensor:cam' with no sensors at all logs a helpful hint."""
    import logging

    recorder = VideoRecorder(_cfg(source="sensor:cam"), _make_env(sensors={}))
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        frame = recorder._get_frame()
    assert frame is None
    assert any("CameraCfg" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------


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
    recorder = VideoRecorder(_cfg(clip_length=100), _make_env(visualizers=[viz]))
    # Manually inject some frames mid-clip.
    recorder._frames = [_FRAME, _FRAME]
    recorder._recording = True

    mock_clip = MagicMock()
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", return_value=mock_clip):
        with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
            recorder.close()

    mock_clip.write_videofile.assert_called_once()
