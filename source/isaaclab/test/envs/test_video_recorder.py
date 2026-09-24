# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for VideoRecorder, VideoRecorderCfg, and the ViewerCfg deprecation shim.

All tests are pure-Python mocks — no simulation context or Kit app required.
"""

from __future__ import annotations

import logging
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
        ("visualizer:newton:streaming_view", ("visualizer", "newton", "streaming_view")),
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
        cfg.validate()
    else:
        with pytest.raises(ValueError, match=error):
            cfg.validate()


@pytest.mark.parametrize(
    "existing,expected_index",
    [([], 0), (["clip_0000.mp4", "clip_0007.mp4", "clip_final.mp4", "other_0008.mp4"], 8)],
    ids=["empty_dir", "after_highest_matching_clip"],
)
def test_init_continues_clip_index_after_existing_clips(tmp_path, existing, expected_index):
    for name in existing:
        (tmp_path / name).touch()

    recorder = VideoRecorder(_cfg(output_dir=str(tmp_path)), _make_env())

    assert recorder._clip_index == expected_index


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
        (dict(video_length=4), 2, [[1, 2]]),
        (dict(video_length=2, step_offset=5), 3, []),
    ],
    ids=["one_shot", "recurring", "step_offset", "frame_stride", "close_flushes_partial", "close_skips_empty"],
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

    recorder = VideoRecorder(
        _cfg(source="visualizer:kit", output_dir=str(tmp_path), **schedule), _make_env(visualizers=[viz])
    )
    with patch("isaaclab.envs.utils.video_recorder.ImageSequenceClip", side_effect=write_clip):
        for step in range(1, num_steps + 1):
            recorder.step()
        recorder.close()

    assert [steps for steps, _ in written] == expected_clips
    assert all(fps == recorder.cfg.fps for _, fps in written)


# ---------------------------------------------------------------------------
# Visualizer frame routing
# ---------------------------------------------------------------------------


def test_visualizer_source_refreshes_physics_before_on_demand_capture():
    """On-demand capture reads a frame after physics transforms are synchronized."""
    synchronized = False

    class _FreshFrameViz(_FakeViz):
        def render_rgb_array(self) -> np.ndarray:
            return np.full_like(self._frame, 255 if synchronized else 0)

    viz = _FreshFrameViz("kit")
    env = _make_env(visualizers=[viz])
    env.sim.is_rendering = False

    def synchronize_physics() -> None:
        nonlocal synchronized
        synchronized = True

    env.sim.forward.side_effect = synchronize_physics
    recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)

    frame = recorder._get_frame()

    assert frame is not None
    assert np.all(frame == 255)


def test_kit_visualizer_newton_physics_logs_warning(caplog):
    """source='visualizer:kit' with Newton physics logs a warning and attempts capture.

    With cubric the capture succeeds; without it frames may be black.  Either way
    the recorder warns and does not hard-fail.

    The warned-about condition is fixed configuration state, so the message is emitted
    once per recorder rather than once per captured frame.
    """
    kit_viz = _FakeViz("kit")
    env = _make_env(visualizers=[kit_viz])
    env.sim.physics_manager.video_capture_backend.return_value = "newton_gl"

    recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)
    with caplog.at_level(logging.WARNING, logger="isaaclab.envs.utils.video_recorder"):
        for _ in range(5):
            recorder._get_frame()
        second_recorder = VideoRecorder(_cfg(source="visualizer:kit"), env)
        second_recorder._get_frame()

    cubric_warnings = [r for r in caplog.records if "source='visualizer:newton'" in r.message]
    assert len(cubric_warnings) == 2
    # Capture is still attempted on every frame rather than short-circuiting.
    assert kit_viz.render_calls == 6


def _rgb_sensor():
    import torch

    sensor = MagicMock()
    sensor.data.output = {"rgb": torch.full((1, *_FRAME.shape), 200, dtype=torch.uint8)}
    return sensor


@pytest.mark.parametrize(
    "source,make_env",
    [
        ("visualizer", lambda: _make_env(visualizers=[_FakeViz("kit")])),
        ("visualizer:newton", lambda: _make_env(visualizers=[_FakeViz("newton_gl")])),
        ("sensor:tiled_camera", lambda: _make_env(sensors={"tiled_camera": _rgb_sensor()})),
    ],
    ids=["auto_visualizer", "newton_alias", "sensor_rgb"],
)
def test_source_resolves_frame(source, make_env):
    frame = VideoRecorder(_cfg(source=source), make_env())._get_frame()
    assert frame is not None
    assert frame.shape == _FRAME.shape


@pytest.mark.parametrize(
    "source,make_env,message",
    [
        ("visualizer", lambda: _make_env(visualizers=[]), "no recording-capable visualizer"),
        ("sensor:missing", lambda: _make_env(sensors={"tiled_camera": MagicMock()}), "tiled_camera"),
    ],
)
def test_unavailable_source_logs_error_and_returns_none(caplog, source, make_env, message):
    """A missing source logs one error naming what is available and yields no frame instead of raising."""
    recorder = VideoRecorder(_cfg(source=source), make_env())
    with caplog.at_level(logging.ERROR, logger="isaaclab.envs.utils.video_recorder"):
        assert recorder._get_frame() is None
    assert any(message in r.message for r in caplog.records)


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
# keep_last_n_clips pruning
# ---------------------------------------------------------------------------


def test_keep_last_n_clips_prunes_only_older_clips(tmp_path):
    """Pruning deletes this recorder's clips older than the last N and leaves other prefixes alone."""
    for name in ["clip_0001.mp4", "clip_9997.mp4", "clip_9998.mp4", "clip_9999.mp4", "other_0000.mp4"]:
        (tmp_path / name).touch()
    recorder = VideoRecorder(_cfg(output_dir=str(tmp_path), keep_last_n_clips=2), _make_env())

    recorder._maybe_delete_old_clips()

    assert sorted(path.name for path in tmp_path.iterdir()) == ["clip_9998.mp4", "clip_9999.mp4", "other_0000.mp4"]
