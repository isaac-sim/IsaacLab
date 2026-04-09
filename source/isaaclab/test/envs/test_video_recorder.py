# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for VideoRecorder."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isaaclab.envs.utils import video_recorder as _video_recorder_module
from isaaclab.envs.utils.video_recorder import VideoRecorder

_BLANK_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
_DEFAULT_CFG = dict(
    env_render_mode="rgb_array",
    camera_position=(7.5, 7.5, 7.5),
    camera_target=(0.0, 0.0, 0.0),
    window_width=1280,
    window_height=720,
)


def _create_recorder(**kw):
    """Return a VideoRecorder with __init__ bypassed and all deps mocked out."""
    backend = kw.pop("_backend", None)
    recorder = object.__new__(VideoRecorder)
    recorder.cfg = SimpleNamespace(**{**_DEFAULT_CFG, **kw})
    recorder._scene = MagicMock()
    recorder._scene.sensors = {}
    recorder._scene._sensor_renderer_types = MagicMock(return_value=[])
    recorder._backend = backend
    cap = MagicMock()
    cap.render_rgb_array = MagicMock(return_value=_BLANK_720p)
    recorder._capture = cap if backend else None
    return recorder


def test_init_perspective_mode_creates_kit_capture():
    """With kit backend, __init__ builds Isaac Sim Kit perspective capture."""
    scene = MagicMock()
    scene.sensors = {}
    scene.num_envs = 1
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    fake_capture = MagicMock()
    kit_mod = MagicMock()
    kit_mod.create_isaacsim_kit_perspective_video = MagicMock(return_value=fake_capture)
    with patch.object(_video_recorder_module, "_resolve_video_backend", return_value="kit"):
        with patch.dict(
            sys.modules,
            {
                "isaaclab_physx.video_recording": MagicMock(),
                "isaaclab_physx.video_recording.isaacsim_kit_perspective_video": kit_mod,
                "isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg": MagicMock(),
            },
        ):
            vr = VideoRecorder(cfg, scene)
    kit_mod.create_isaacsim_kit_perspective_video.assert_called_once()
    assert vr._capture is fake_capture


def test_init_newton_backend_creates_newton_capture():
    """With newton_gl backend, __init__ builds Newton GL perspective capture."""
    scene = MagicMock()
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    fake_capture = MagicMock()
    newton_mod = MagicMock()
    newton_mod.create_newton_gl_perspective_video = MagicMock(return_value=fake_capture)
    with patch.object(_video_recorder_module, "_resolve_video_backend", return_value="newton_gl"):
        with patch.dict(
            sys.modules,
            {
                "pyglet": MagicMock(),
                "isaaclab_newton.video_recording": MagicMock(),
                "isaaclab_newton.video_recording.newton_gl_perspective_video": newton_mod,
                "isaaclab_newton.video_recording.newton_gl_perspective_video_cfg": MagicMock(),
            },
        ):
            vr = VideoRecorder(cfg, scene)
    newton_mod.create_newton_gl_perspective_video.assert_called_once()
    assert vr._capture is fake_capture


def test_render_rgb_array_delegates_to_capture():
    """render_rgb_array returns capture.render_rgb_array()."""
    recorder = _create_recorder(_backend="kit")
    result = recorder.render_rgb_array()
    recorder._capture.render_rgb_array.assert_called_once()
    assert result.shape == (720, 1280, 3)


def test_render_rgb_array_none_when_no_backend():
    """Without rgb_array env_render_mode, _capture is None and render returns None."""
    recorder = _create_recorder(env_render_mode=None)
    recorder._backend = None
    recorder._capture = None
    assert recorder.render_rgb_array() is None


def test_capture_exception_propagates():
    """Failures in backend capture propagate."""
    recorder = _create_recorder(_backend="newton_gl")
    recorder._capture.render_rgb_array.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError, match="fail"):
        recorder.render_rgb_array()


def test_render_rgb_array_calls_capture_each_step():
    """Each render_rgb_array call hits the backend capture."""
    recorder = _create_recorder(_backend="kit")
    for _ in range(3):
        recorder.render_rgb_array()
    assert recorder._capture.render_rgb_array.call_count == 3
