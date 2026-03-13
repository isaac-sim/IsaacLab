# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for VideoRecorder."""
import importlib.util, pathlib, sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("_vr", pathlib.Path(__file__).parent / "video_recorder.py")
_module = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_module); VideoRecorder = _module.VideoRecorder

_BLANK_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
_DEFAULT_CFG = dict(
    render_mode="rgb_array", video_mode="perspective", fallback_camera_cfg=None,
    video_num_tiles=-1, camera_eye=(7.5, 7.5, 7.5), camera_lookat=(0.0, 0.0, 0.0),
    gl_viewer_width=1280, gl_viewer_height=720,
)


def _create_recorder(**kw):
    """Return a VideoRecorder with __init__ bypassed and all deps mocked out."""
    recorder = object.__new__(VideoRecorder)
    recorder.cfg = SimpleNamespace(**{**_DEFAULT_CFG, **kw})
    recorder._scene = MagicMock(); recorder._scene.sensors = {}
    recorder._fallback_tiled_camera = None
    recorder._gl_viewer = None
    recorder._gl_viewer_init_attempted = False
    return recorder


def test_init_perspective_mode_does_not_spawn_fallback():
    """In perspective mode, __init__ never spawns a TiledCamera fallback."""
    scene = MagicMock(); scene.sensors = {}; scene.num_envs = 1
    cfg = SimpleNamespace(**{**_DEFAULT_CFG, "fallback_camera_cfg": MagicMock()})
    with patch.dict(sys.modules, {"pyglet": MagicMock()}):
        with patch.object(VideoRecorder, "_spawn_fallback_cameras") as mock_spawn:
            VideoRecorder(cfg, scene)
    mock_spawn.assert_not_called()


def test_init_tiled_mode_spawns_fallback_when_configured():
    """In tiled mode with a fallback_camera_cfg, __init__ calls _spawn_fallback_cameras."""
    scene = MagicMock(); scene.sensors = {}; scene.num_envs = 1
    cfg = SimpleNamespace(**{**_DEFAULT_CFG, "video_mode": "tiled", "fallback_camera_cfg": MagicMock()})
    with patch.dict(sys.modules, {"pyglet": MagicMock()}):
        with patch.object(VideoRecorder, "_spawn_fallback_cameras", return_value=MagicMock()) as mock_spawn:
            VideoRecorder(cfg, scene)
    mock_spawn.assert_called_once()


def test_render_rgb_array_perspective_uses_gl_viewer_when_available():
    """Perspective mode returns a GL viewer frame when _gl_viewer is set."""
    recorder = _create_recorder()
    recorder._gl_viewer = MagicMock(); recorder._gl_viewer_init_attempted = True
    with patch.object(recorder, "_render_newton_gl_rgb_array", return_value=_BLANK_720p) as mock_gl:
        result = recorder.render_rgb_array()
    mock_gl.assert_called_once()
    assert result.shape == (720, 1280, 3)


def test_render_rgb_array_perspective_falls_through_to_kit_when_no_gl_viewer():
    """Kit capture path is used when no GL viewer is available (Kit backend)."""
    recorder = _create_recorder(); recorder._gl_viewer_init_attempted = True
    with patch.object(recorder, "_render_kit_perspective_rgb_array", return_value=_BLANK_720p) as mock_kit:
        recorder.render_rgb_array()
    mock_kit.assert_called_once()


def test_render_rgb_array_tiled_raises_when_no_camera():
    """Tiled mode with no TiledCamera raises RuntimeError with a descriptive message."""
    recorder = _create_recorder(video_mode="tiled")
    with patch.object(recorder, "_find_video_camera", return_value=None):
        with pytest.raises(RuntimeError, match="tiled mode"):
            recorder.render_rgb_array()


def test_gl_exception_returns_blank_ndarray_not_none():
    """GL renderer crash must return a blank ndarray, never None, so RecordVideo never sees None."""
    recorder = _create_recorder(); recorder._gl_viewer = MagicMock(); recorder._gl_viewer_init_attempted = True
    with patch.dict(sys.modules, {"isaaclab.sim": MagicMock(SimulationContext=MagicMock(instance=MagicMock(side_effect=RuntimeError)))}):
        frame = recorder._render_newton_gl_rgb_array()
    assert isinstance(frame, np.ndarray) and frame.shape == (720, 1280, 3)


def test_find_video_camera_does_not_cache_none():
    """A None result is not cached, allowing retry on the next call."""
    recorder = _create_recorder(video_mode="tiled")
    FakeTiledCamera = type("TiledCamera", (), {})
    with patch.dict(sys.modules, {"isaaclab": MagicMock(), "isaaclab.sensors": MagicMock(), "isaaclab.sensors.camera": MagicMock(TiledCamera=FakeTiledCamera)}):
        result = recorder._find_video_camera()
    assert result is None and not hasattr(recorder, "_video_camera")


def test_find_video_camera_caches_result_when_found():
    """A found camera is cached so the scene is not re-scanned on subsequent calls."""
    recorder = _create_recorder(video_mode="tiled")
    FakeTiledCamera = type("TiledCamera", (), {})
    camera = MagicMock(); camera.__class__ = FakeTiledCamera
    camera.is_initialized = True; camera.data.output = {"rgb": MagicMock(shape=(4, 64, 64, 3))}
    recorder._scene.sensors = {"cam": camera}
    with patch.dict(sys.modules, {"isaaclab": MagicMock(), "isaaclab.sensors": MagicMock(), "isaaclab.sensors.camera": MagicMock(TiledCamera=FakeTiledCamera)}):
        result = recorder._find_video_camera()
    assert result is camera and hasattr(recorder, "_video_camera")


def test_gl_viewer_init_attempted_only_once():
    """_try_init_gl_viewer is called at most once regardless of render call count."""
    recorder = _create_recorder(); recorder._gl_viewer_init_attempted = False
    def _set_flag(): recorder._gl_viewer_init_attempted = True
    with patch.object(recorder, "_try_init_gl_viewer", side_effect=_set_flag) as mock_init, \
         patch.object(recorder, "_render_kit_perspective_rgb_array", return_value=_BLANK_720p):
        for _ in range(3): recorder.render_rgb_array()
    mock_init.assert_called_once()
