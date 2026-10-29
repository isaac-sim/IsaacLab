# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for VideoRecorder and VideoRecorderCfg.

All tests are pure-Python mocks — no simulation context or Kit app required.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg, parse_video_source
from isaaclab.utils import validate
from isaaclab.visualizers import VisualizerBackendCfg, VisualizerCfg

_FRAME = np.ones((8, 12, 3), dtype=np.uint8) * 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_moviepy():
    """Stub out ImageSequenceClip so tests run without moviepy installed.

    Missing-dependency coverage blocks the module import in its own context.
    """
    with patch.dict(
        "sys.modules", {"moviepy.video.io.ImageSequenceClip": SimpleNamespace(ImageSequenceClip=MagicMock())}
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> VideoRecorderCfg:
    defaults = dict(source="viz", output_dir="/tmp/test_videos", fps=30, video_length=4, video_interval=0)
    return VideoRecorderCfg(**(defaults | overrides))


class _FakeViz:
    def __init__(self, viz_type: str, frame: np.ndarray | None = None):
        self.cfg = VisualizerCfg(backend=VisualizerBackendCfg(visualizer_type=viz_type))
        self._frame = frame if frame is not None else _FRAME.copy()
        self.render_calls = 0

    def render_rgb_array(self) -> np.ndarray:
        self.render_calls += 1
        return self._frame

    def supports_markers(self) -> bool:
        return True


def _make_env(visualizers=(), sensors: dict | None = None):
    env = SimpleNamespace(sim=MagicMock(), scene=SimpleNamespace(), step_dt=0.02)
    env.sim.visualizers = list(visualizers)
    env.scene.sensors = sensors or {}
    from isaaclab.sim import SimulationContext

    env.sim._backend_registry = []
    env.sim._scene_data_provider.get_camera_sensors.return_value = env.scene.sensors
    env.sim.get_or_create_backend.side_effect = lambda cfg, **kwargs: SimulationContext.get_or_create_backend(
        env.sim, cfg, **kwargs
    )
    env.sim.get_physics_step_count.return_value = 0
    return env


# ---------------------------------------------------------------------------
# parse_video_source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("viz", ("viz", "", "")),
        ("kit", ("viz", "kit", "")),
        ("newton_gl", ("viz", "newton_gl", "")),
        ("newton_rtx", ("viz", "newton_rtx", "")),
        ("viser", ("viz", "viser", "")),
        ("rerun", ("viz", "rerun", "")),
        ("viz:newton_gl:streaming_view", ("viz", "newton_gl", "streaming_view")),
        ("sensor:tiled_camera", ("sensor", "tiled_camera", "")),
        ("sensor:tiled_camera:depth", ("sensor", "tiled_camera", "depth")),
        # ``visualizer`` is the long form of ``viz``
        ("visualizer:kit", ("viz", "kit", "")),
        # the deprecated ``newton`` type maps to ``newton_gl``, with a DeprecationWarning
        ("visualizer:newton:streaming_view", ("viz", "newton_gl", "streaming_view")),
        ("newton", ("viz", "newton_gl", "")),
        ("a=b", "Invalid video source"),
        ("viz:foo", "Invalid video source"),
        ("viz:kit:bar", "Invalid video source"),
        ("sensor", "Invalid video source"),
        ("sensor:", "Invalid video source"),
        ("sensor:cam:foo", "Invalid video source"),
        ("newton_gl:streaming_view", "Invalid video source"),
    ],
)
def test_parse_video_source(source, expected):
    if isinstance(expected, str):  # an invalid source and the error it raises
        with pytest.raises(ValueError, match=expected):
            parse_video_source(source)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert parse_video_source(source) == expected
    assert [w.category for w in caught] == [DeprecationWarning] * ("newton" in source.split(":"))


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_init_raises_import_error_when_moviepy_missing():
    with patch.dict("sys.modules", {"moviepy.video.io.ImageSequenceClip": None}):
        with pytest.raises(ImportError, match="moviepy"):
            VideoRecorder(_cfg(), _make_env())


@pytest.mark.parametrize(
    "overrides,error",
    [
        (dict(video_length=1, frame_stride=1, video_interval=0, step_offset=0), None),
        (dict(video_length=0), "video_length=0"),
        (dict(frame_stride=0), "frame_stride=0"),
        (dict(video_interval=-1), "video_interval=-1"),
        (dict(step_offset=-1), "step_offset=-1"),
    ],
    ids=["valid_boundary", "video_length", "frame_stride", "video_interval", "step_offset"],
)
def test_cfg_validate_clip_schedule(overrides, error):
    cfg = _cfg(**overrides)
    if error is None:
        validate(cfg)
    else:
        with pytest.raises(ValueError, match=error):
            validate(cfg)


# ---------------------------------------------------------------------------
# Clip schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schedule,num_steps,expected_clips",
    [
        (dict(video_length=2), 8, [[1, 2]]),
        (dict(video_length=1, video_interval=3), 9, [[1], [4], [7]]),
        (dict(video_length=2, step_offset=5), 8, [[6, 7]]),
        (dict(video_length=4, frame_stride=2), 4, [[2, 4]]),
        (dict(video_length=6, video_interval=3, frame_stride=2), 8, [[2], [5], [8]]),
        (dict(video_length=4), 2, [[1, 2]]),
        (dict(video_length=2, step_offset=5), 3, []),
        (dict(video_length=4, frame_stride=2, fps=None), 4, [[2, 4]]),
    ],
    ids=[
        "one_shot",
        "recurring",
        "step_offset",
        "frame_stride",
        "overlapping_clips",
        "close_flushes_partial",
        "close_skips_empty",
        "automatic_fps_without_gym_metadata",
    ],
)
def test_step_schedule_writes_expected_clips(tmp_path, schedule, num_steps, expected_clips):
    """Each written clip holds the frames of exactly the env steps its schedule captures, then close() flushes."""
    viz = _FakeViz("kit")
    step = 0
    viz.render_rgb_array = lambda: np.full_like(_FRAME, step)
    written = []

    def write_clip(frames, fps):
        written.append(([int(frame[0, 0, 0]) for frame in frames], fps))
        return MagicMock()

    recorder = VideoRecorder(_cfg(source="viz:kit", output_dir=str(tmp_path), **schedule), _make_env(visualizers=[viz]))
    with patch.object(recorder, "_clip_class", side_effect=write_clip):
        for step in range(1, num_steps + 1):
            recorder.step()
        recorder.close()

    assert [steps for steps, _ in written] == expected_clips
    expected_fps = recorder.cfg.fps or round(1 / (0.02 * recorder.cfg.frame_stride))
    assert all(fps == expected_fps for _, fps in written)

    if recorder.cfg.fps is None:
        # Gym environments may override the playback rate in their metadata.
        env = gym.Env()
        env.sim, env.step_dt, env.metadata = _make_env([viz]).sim, 0.02, {"render_fps": 40}
        recorder = VideoRecorder(recorder.cfg, env)
        with patch.object(recorder, "_clip_class", side_effect=write_clip):
            for _ in range(num_steps):
                recorder.step()
        assert written[-1][1] == 20


# ---------------------------------------------------------------------------
# Visualizer frame routing
# ---------------------------------------------------------------------------


def _rgb_sensor():
    import warp as wp

    from isaaclab.utils.warp import ProxyArray

    sensor = MagicMock()
    sensor.cfg.data_types = ["rgb"]
    sensor.data.output = {"rgb": ProxyArray(wp.full((1, *_FRAME.shape), 200, dtype=wp.uint8, device="cpu"))}
    return sensor


@pytest.mark.parametrize(
    "source,make_env",
    [
        ("viz", lambda: _make_env(visualizers=[_FakeViz("kit")])),
        ("viz:newton_gl", lambda: _make_env(visualizers=[_FakeViz("newton_gl")])),
        ("sensor:tiled_camera", lambda: _make_env(sensors={"tiled_camera": _rgb_sensor()})),
    ],
    ids=["auto_visualizer", "newton_gl", "sensor_rgb"],
)
def test_source_resolves_frame(source, make_env, caplog):
    env = make_env()
    visualizers = env.sim.visualizers
    recorder = VideoRecorder(_cfg(source=source), env)
    # Capture is bound during construction; later frames do not discover producers again.
    env.sim.visualizers = []
    if source.startswith("sensor:"):
        env.sim._scene_data_provider.get_camera_sensors.return_value = {}
    if source == "viz:newton_gl":
        visualizers[0]._frame = None
        assert recorder._capture() is None
        visualizers[0]._frame = _FRAME.copy()
    frame = recorder._capture()
    assert frame is not None
    assert frame.shape == _FRAME.shape
    assert not caplog.records


@pytest.mark.parametrize(
    "source,make_env,error,message",
    [
        ("badkind:foo", _make_env, ValueError, "Invalid video source"),
        ("viz", lambda: _make_env(visualizers=[]), RuntimeError, "no recording-capable visualizer"),
        (
            "viz:newton_gl:streaming_view",
            lambda: _make_env(visualizers=[_FakeViz("newton_gl")]),
            RuntimeError,
            "Enable streaming_view",
        ),
        ("sensor:missing", lambda: _make_env(sensors={"tiled_camera": MagicMock()}), ValueError, "missing"),
        ("sensor:tiled_camera:depth", lambda: _make_env(sensors={"tiled_camera": _rgb_sensor()}), KeyError, "depth"),
    ],
)
def test_invalid_source_raises(source, make_env, error, message):
    """Invalid sources fail when the recorder is constructed, before training starts."""
    with pytest.raises(error, match=message):
        VideoRecorder(_cfg(source=source), make_env())


# ---------------------------------------------------------------------------
# keep_last_n_clips pruning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "existing,expected",
    [
        ([], ["clip_0000.mp4"]),
        (
            ["clip_0001.mp4", "clip_9997.mp4", "clip_9998.mp4", "clip_final.mp4", "other_10000.mp4"],
            ["clip_9998.mp4", "clip_9999.mp4", "clip_final.mp4", "other_10000.mp4"],
        ),
    ],
    ids=["empty_dir", "resume_and_prune"],
)
def test_clip_files_resume_and_prune_only_matching_indices(tmp_path, existing, expected):
    """Continue the highest clip index and retain only this recorder's last N numbered clips."""
    for name in existing:
        (tmp_path / name).touch()
    recorder = VideoRecorder(
        _cfg(output_dir=str(tmp_path), keep_last_n_clips=2, video_length=1), _make_env([_FakeViz("newton_gl")])
    )
    with patch.object(recorder, "_clip_class") as encoder:
        encoder.return_value.write_videofile.side_effect = lambda path, **kwargs: (tmp_path / path).touch()
        recorder.step()

    assert sorted(path.name for path in tmp_path.iterdir()) == expected
