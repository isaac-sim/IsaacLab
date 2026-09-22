# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`VideoRecorder`.

The environment is a stub exposing ``sim`` and ``scene`` and ``moviepy`` is replaced by a recording double,
so no simulation context or Kit app is required.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

pytestmark = pytest.mark.unit

_FRAME = np.full((8, 12, 3), 128, dtype=np.uint8)
_LOGGER = "isaaclab.envs.utils.video_recorder"


@pytest.fixture(autouse=True)
def image_sequence_clip():
    """Stub out ImageSequenceClip so tests run without moviepy installed."""
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip") as clip_cls:
        yield clip_cls


def _cfg(**overrides) -> VideoRecorderCfg:
    return VideoRecorderCfg(**{"output_dir": "/tmp/test_videos", "fps": 30, "video_length": 4, **overrides})


class _FakeViz:
    def __init__(self, viz_type: str):
        self.cfg = SimpleNamespace(visualizer_type=viz_type)
        self.render_calls = 0

    def render_rgb_array(self) -> np.ndarray:
        self.render_calls += 1
        return _FRAME.copy()


class _FakeSim:
    def __init__(self, visualizers=(), is_rendering=True, physics_backend=None):
        self.visualizers = list(visualizers)
        self.is_rendering = is_rendering
        self.forward_calls = 0
        self.physics_manager = SimpleNamespace(video_capture_backend=lambda: physics_backend)

    def forward(self) -> None:
        self.forward_calls += 1


def _make_env(visualizers=(), sensors: dict | None = None, **sim_kwargs):
    return SimpleNamespace(
        sim=_FakeSim(visualizers, **sim_kwargs), scene=SimpleNamespace(sensors=sensors or {}), step_dt=0.1
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("visualizer", ("visualizer", "", "")),
        ("visualizer:kit", ("visualizer", "kit", "")),
        ("visualizer:newton:streaming_view", ("visualizer", "newton", "streaming_view")),
        ("sensor:tiled_camera", ("sensor", "tiled_camera", "")),
        ("  visualizer:kit  ", ("visualizer", "kit", "")),
    ],
)
def test_parse_source(source, expected):
    assert _parse_source(source) == expected


def test_init_validation():
    """Unknown source kinds and a missing moviepy are rejected on construction."""
    with pytest.raises(ValueError, match="Unrecognized source kind"):
        VideoRecorder(_cfg(source="badkind:foo"), _make_env())
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", None):
        with pytest.raises(ImportError, match="moviepy"):
            VideoRecorder(_cfg(), _make_env())


@pytest.mark.parametrize(
    ("existing", "expected_index"),
    [([], 0), (["clip_0000.mp4", "clip_0007.mp4", "clip_final.mp4", "other_0008.mp4"], 8)],
    ids=["empty_dir", "existing_clips"],
)
def test_init_clip_index_continues_after_existing_files(tmp_path, existing, expected_index):
    """The clip index continues after the highest existing clip of the same prefix."""
    output_dir = tmp_path / "videos"
    output_dir.mkdir()
    for name in existing:
        (output_dir / name).touch()

    recorder = VideoRecorder(_cfg(output_dir=str(output_dir)), _make_env())

    assert recorder._clip_index == expected_index


@pytest.mark.parametrize(
    ("video_interval", "video_length", "expected_trigger_steps"),
    [(0, 2, [1]), (3, 1, [1, 4, 7])],
    ids=["one_shot", "recurring"],
)
def test_trigger_schedule(video_interval, video_length, expected_trigger_steps):
    """A zero interval records a single clip at step 1; a positive interval records recurring clips."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", video_length=video_length, video_interval=video_interval),
        _make_env(visualizers=[viz]),
    )
    trigger_steps = []

    def counting_close():
        trigger_steps.append(recorder._step_count - video_length + 1)
        recorder._frames = []
        recorder._recording = False

    with patch.object(recorder, "_close_clip", side_effect=counting_close):
        for _ in range(9):
            recorder.step()
    assert trigger_steps == expected_trigger_steps


def test_step_offset_and_frame_stride():
    """Recording starts after ``step_offset`` steps and captures one frame every ``frame_stride`` steps."""
    viz = _FakeViz("kit")
    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", step_offset=5, video_length=4, frame_stride=2), _make_env(visualizers=[viz])
    )
    for _ in range(5):
        recorder.step()
    assert not recorder._recording and viz.render_calls == 0

    with patch.object(recorder, "_close_clip") as mock_close:
        recorder.step()
        assert recorder._recording
        for _ in range(3):
            recorder.step()
        assert mock_close.call_count == 1
    assert viz.render_calls == 2


def test_visualizer_source_selection():
    """Auto mode picks the first capture-capable visualizer and ``newton`` aliases ``newton_gl``."""
    kit_viz, newton_viz = _FakeViz("kit"), _FakeViz("newton_gl")
    env = _make_env(visualizers=[kit_viz, newton_viz])

    assert VideoRecorder(_cfg(source="visualizer"), env)._get_frame() is not None
    assert (kit_viz.render_calls, newton_viz.render_calls) == (1, 0)
    assert VideoRecorder(_cfg(source="visualizer:newton"), env)._get_frame() is not None
    assert (kit_viz.render_calls, newton_viz.render_calls) == (1, 1)


def test_visualizer_source_refreshes_physics_before_on_demand_capture():
    """Without continuous rendering, physics transforms are synchronized before a frame is read."""
    env = _make_env(visualizers=[_FakeViz("kit")], is_rendering=False)
    recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)

    assert recorder._get_frame() is not None
    assert env.sim.forward_calls == 1


@pytest.mark.parametrize(
    ("source", "sensors", "expected_message"),
    [
        ("visualizer", {}, "no recording-capable visualizer"),
        ("sensor:missing", {"tiled_camera": object()}, "tiled_camera"),
    ],
    ids=["no_visualizer", "missing_sensor"],
)
def test_missing_source_logs_once_and_returns_none(caplog, source, sensors, expected_message):
    """A missing visualizer or sensor logs an error once and suppresses further capture attempts."""
    recorder = VideoRecorder(_cfg(source=source), _make_env(sensors=sensors))
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        assert recorder._get_frame() is None
        assert recorder._get_frame() is None
    error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert expected_message in error_records[0].message


def test_kit_visualizer_newton_physics_logs_warning_once(caplog):
    """Kit capture with Newton physics warns once per recorder and keeps capturing frames."""
    kit_viz = _FakeViz("kit")
    env = _make_env(visualizers=[kit_viz], physics_backend="newton_gl")

    recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for _ in range(5):
            recorder._get_frame()
        VideoRecorder(_cfg(source="visualizer:kit"), env)._get_frame()

    cubric_warnings = [r for r in caplog.records if "source='visualizer:newton'" in r.message]
    assert len(cubric_warnings) == 2
    assert kit_viz.render_calls == 6


def test_sensor_source_reads_rgb():
    """Sensor sources read the first environment's RGB output as an ``(H, W, 3)`` frame."""
    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.full((1, 8, 12, 3), 200, dtype=torch.uint8)}))
    recorder = VideoRecorder(_cfg(source="sensor:tiled_camera"), _make_env(sensors={"tiled_camera": sensor}))
    frame = recorder._get_frame()
    assert frame is not None
    assert frame.shape == (8, 12, 3)


def test_close_clip_writes_mp4_and_clears_frames(image_sequence_clip):
    """Buffered frames are written through moviepy at the configured fps, then the buffer is cleared."""
    frames = [_FRAME.copy(), _FRAME.copy()]
    recorder = VideoRecorder(_cfg(fps=10), _make_env())
    recorder._frames = frames
    recorder._recording = True

    with patch("isaaclab.envs.utils.video_recorder.os.makedirs"):
        recorder.close()

    image_sequence_clip.assert_called_once_with(frames, fps=10)
    image_sequence_clip.return_value.write_videofile.assert_called_once()
    assert not recorder._recording
    assert recorder._frames == []


def test_close_with_empty_frame_buffer_does_not_write(image_sequence_clip):
    recorder = VideoRecorder(_cfg(), _make_env())
    recorder._recording = True
    recorder.close()
    image_sequence_clip.assert_not_called()


def test_keep_last_n_clips_prunes_only_existing_old_clips():
    """Only existing clips below the retention cutoff are deleted; sparse indices are not probed."""
    recorder = VideoRecorder(_cfg(output_dir="/tmp/test_sparse_prune", keep_last_n_clips=2), _make_env())
    recorder._clip_index = 10_000
    removed = []

    with patch("isaaclab.envs.utils.video_recorder.os.path.isdir", return_value=True):
        with patch(
            "isaaclab.envs.utils.video_recorder.os.listdir",
            return_value=["clip_0001.mp4", "clip_9997.mp4", "clip_9998.mp4", "other_0000.mp4"],
        ):
            with patch("isaaclab.envs.utils.video_recorder.os.remove", side_effect=removed.append):
                recorder._maybe_delete_old_clips()

    assert removed == ["/tmp/test_sparse_prune/clip_0001.mp4", "/tmp/test_sparse_prune/clip_9997.mp4"]
