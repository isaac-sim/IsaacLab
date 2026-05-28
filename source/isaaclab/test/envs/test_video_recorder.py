# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for VideoRecorder."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isaaclab.envs.utils import video_recorder as _video_recorder_module
from isaaclab.envs.utils.video_recorder import VideoRecorder, _resolve_video_backend, _sync_camera_from_visualizer

pytestmark = pytest.mark.isaacsim_ci
_BLANK_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
_DEFAULT_CFG = dict(
    env_render_mode="rgb_array",
    eye=(7.5, 7.5, 7.5),
    lookat=(0.0, 0.0, 0.0),
    backend_source="visualizer",
    window_width=1280,
    window_height=720,
)


def _create_recorder(**kw):
    """Return a VideoRecorder with ``__init__`` bypassed and all deps mocked out."""
    backend = kw.pop("_backend", None)
    matched_visualizer = kw.pop("_matched_visualizer", None)
    live_visualizer = kw.pop("_live_visualizer", None)
    recorder = object.__new__(VideoRecorder)
    recorder.cfg = SimpleNamespace(**{**_DEFAULT_CFG, **kw})
    recorder._scene = MagicMock()
    recorder._scene.sensors = {}
    recorder._scene._sensor_renderer_types = MagicMock(return_value=[])
    recorder._scene.sim.visualizers = []
    recorder._backend = backend
    recorder._matched_visualizer = matched_visualizer
    recorder._live_visualizer = live_visualizer
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
    with patch.object(_video_recorder_module, "_resolve_video_backend", return_value=("kit", None)):
        with patch.object(_video_recorder_module, "_sync_camera_from_visualizer"):
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
    assert vr._matched_visualizer is None


def test_init_newton_backend_creates_newton_capture():
    """With newton_gl backend, __init__ builds Newton GL perspective capture."""
    scene = MagicMock()
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    fake_capture = MagicMock()
    newton_mod = MagicMock()
    newton_mod.create_newton_gl_perspective_video = MagicMock(return_value=fake_capture)
    with patch.object(_video_recorder_module, "_resolve_video_backend", return_value=("newton_gl", "newton")):
        with patch.object(_video_recorder_module, "_sync_camera_from_visualizer"):
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
    assert vr._matched_visualizer == "newton"


def test_init_kit_from_visualizer_syncs_camera():
    """When backend comes from a visualizer, _sync_camera_from_visualizer is called."""
    scene = MagicMock()
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    with patch.object(_video_recorder_module, "_resolve_video_backend", return_value=("kit", "kit")):
        with patch.object(_video_recorder_module, "_sync_camera_from_visualizer") as mock_sync:
            with patch.dict(
                sys.modules,
                {
                    "isaaclab_physx.video_recording": MagicMock(),
                    "isaaclab_physx.video_recording.isaacsim_kit_perspective_video": MagicMock(),
                    "isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg": MagicMock(),
                },
            ):
                VideoRecorder(cfg, scene)
    mock_sync.assert_called_once_with(scene, "kit", cfg)


def test_init_no_visualizer_skips_camera_sync():
    """When backend comes from physics/renderer stack, camera sync is skipped."""
    scene = MagicMock()
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    with patch.object(_video_recorder_module, "_resolve_video_backend", return_value=("kit", None)):
        with patch.object(_video_recorder_module, "_sync_camera_from_visualizer") as mock_sync:
            with patch.dict(
                sys.modules,
                {
                    "isaaclab_physx.video_recording": MagicMock(),
                    "isaaclab_physx.video_recording.isaacsim_kit_perspective_video": MagicMock(),
                    "isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg": MagicMock(),
                },
            ):
                VideoRecorder(cfg, scene)
    mock_sync.assert_not_called()


def _make_scene(visualizer_types, physics_name="PhysxPhysicsManager", renderer_types=None):
    scene = MagicMock()
    scene.sim.resolve_visualizer_types.return_value = visualizer_types
    scene.sim.physics_manager.__name__ = physics_name
    scene._sensor_renderer_types.return_value = renderer_types or []
    return scene


def test_resolve_backend_prefers_kit_visualizer():
    """When 'kit' visualizer is active, backend is 'kit' with matched type 'kit'."""
    scene = _make_scene(["kit"])
    backend, matched = _resolve_video_backend(scene)
    assert backend == "kit"
    assert matched == "kit"


def test_resolve_backend_prefers_newton_visualizer():
    """When 'newton' visualizer is active, backend is 'newton_gl' with matched type 'newton'."""
    scene = _make_scene(["newton"], physics_name="NewtonPhysicsManager")
    backend, matched = _resolve_video_backend(scene)
    assert backend == "newton_gl"
    assert matched == "newton"


def test_resolve_backend_renderer_source_ignores_visualizer():
    """When backend_source is 'renderer', active visualizers do not drive backend selection."""
    scene = _make_scene(["newton"], physics_name="PhysxPhysicsManager")
    backend, matched = _resolve_video_backend(scene, backend_source="renderer")
    assert backend == "kit"
    assert matched is None


def test_resolve_backend_kit_wins_over_newton_visualizer():
    """When both kit and newton visualizers are active, kit takes priority."""
    scene = _make_scene(["newton", "kit"])
    backend, matched = _resolve_video_backend(scene)
    assert backend == "kit"
    assert matched == "kit"


def test_resolve_backend_unsupported_visualizer_falls_through():
    """viser/rerun visualizers fall through to physics stack detection."""
    scene = _make_scene(["viser"], physics_name="PhysxPhysicsManager")
    backend, matched = _resolve_video_backend(scene)
    assert backend == "kit"
    assert matched is None


def test_resolve_backend_fallback_physx_returns_none_matched():
    """Physics/renderer fallback returns None as matched visualizer."""
    scene = _make_scene([], physics_name="PhysxPhysicsManager")
    backend, matched = _resolve_video_backend(scene)
    assert backend == "kit"
    assert matched is None


def test_resolve_backend_fallback_newton_physics_returns_none_matched():
    """Newton physics fallback returns None as matched visualizer."""
    scene = _make_scene([], physics_name="NewtonPhysicsManager")
    backend, matched = _resolve_video_backend(scene)
    assert backend == "newton_gl"
    assert matched is None


def test_resolve_backend_raises_when_no_supported_backend():
    """RuntimeError when no supported backend can be detected."""
    scene = _make_scene([], physics_name="UnknownManager")
    with pytest.raises(RuntimeError, match="No supported backend detected"):
        _resolve_video_backend(scene)


def test_resolve_backend_raises_for_invalid_backend_source():
    """Only 'visualizer' and 'renderer' are valid backend source modes."""
    scene = _make_scene([])
    with pytest.raises(ValueError, match="backend_source"):
        _resolve_video_backend(scene, backend_source="invalid")


def _make_visualizer_cfg(visualizer_type, eye=None, lookat=None):
    return SimpleNamespace(visualizer_type=visualizer_type, eye=eye, lookat=lookat)


def test_sync_camera_overwrites_cfg_from_visualizer():
    """Visualizer cfg eye/lookat are written into VideoRecorderCfg."""
    scene = MagicMock()
    scene.sim._resolve_visualizer_cfgs.return_value = [
        _make_visualizer_cfg("newton", eye=(1.0, 2.0, 3.0), lookat=(4.0, 5.0, 6.0)),
    ]
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    _sync_camera_from_visualizer(scene, "newton", cfg)
    assert cfg.eye == (1.0, 2.0, 3.0)
    assert cfg.lookat == (4.0, 5.0, 6.0)


def test_sync_camera_skips_wrong_visualizer_type():
    """Only the matching visualizer type updates the cfg."""
    scene = MagicMock()
    scene.sim._resolve_visualizer_cfgs.return_value = [
        _make_visualizer_cfg("kit", eye=(9.0, 9.0, 9.0), lookat=(1.0, 1.0, 1.0)),
    ]
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    original_eye = cfg.eye
    _sync_camera_from_visualizer(scene, "newton", cfg)
    assert cfg.eye == original_eye  # unchanged


def test_sync_camera_handles_missing_camera_fields():
    """If visualizer cfg has no camera fields, existing cfg values are kept."""
    scene = MagicMock()
    vcfg = _make_visualizer_cfg("newton", eye=None, lookat=None)
    scene.sim._resolve_visualizer_cfgs.return_value = [vcfg]
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    original_eye = cfg.eye
    _sync_camera_from_visualizer(scene, "newton", cfg)
    assert cfg.eye == original_eye


def test_sync_camera_handles_resolve_exception():
    """If _resolve_visualizer_cfgs raises, no exception propagates and cfg is unchanged."""
    scene = MagicMock()
    scene.sim._resolve_visualizer_cfgs.side_effect = RuntimeError("boom")
    cfg = SimpleNamespace(**_DEFAULT_CFG)
    original_eye = cfg.eye
    _sync_camera_from_visualizer(scene, "newton", cfg)
    assert cfg.eye == original_eye


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


def test_render_rgb_array_calls_sync_newton_camera_when_newton_visualizer():
    """render_rgb_array triggers _sync_newton_camera when matched_visualizer is 'newton'."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    with patch.object(recorder, "_sync_newton_camera") as mock_sync:
        recorder.render_rgb_array()
    mock_sync.assert_called_once()


def test_render_rgb_array_skips_sync_for_kit_visualizer():
    """render_rgb_array does NOT call _sync_newton_camera for kit backend."""
    recorder = _create_recorder(_backend="kit", _matched_visualizer="kit")
    with patch.object(recorder, "_sync_newton_camera") as mock_sync:
        recorder.render_rgb_array()
    mock_sync.assert_not_called()


def test_render_rgb_array_skips_sync_when_no_visualizer():
    """render_rgb_array does NOT call _sync_newton_camera when using physics/renderer stack."""
    recorder = _create_recorder(_backend="kit", _matched_visualizer=None)
    with patch.object(recorder, "_sync_newton_camera") as mock_sync:
        recorder.render_rgb_array()
    mock_sync.assert_not_called()


def _make_newton_visualizer(pos=(1.0, 2.0, 3.0), yaw_deg=45.0, pitch_deg=30.0):
    """Return a mock that quacks like a NewtonVisualizer with a live camera."""
    viz = MagicMock()
    viz.cfg.visualizer_type = "newton"
    cam = MagicMock()
    cam.pos = pos
    cam.yaw = yaw_deg
    cam.pitch = pitch_deg
    viz._viewer = MagicMock()
    viz._viewer.camera = cam
    return viz


def test_sync_newton_camera_lazy_lookup_finds_visualizer():
    """_sync_newton_camera resolves the Newton visualizer on the first call."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    newton_viz = _make_newton_visualizer()
    recorder._scene.sim.visualizers = [newton_viz]

    recorder._sync_newton_camera()

    assert recorder._live_visualizer is newton_viz
    recorder._capture.update_camera.assert_called_once()


def test_sync_newton_camera_uses_cached_visualizer():
    """_sync_newton_camera uses the cached _live_visualizer and skips the list walk."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    newton_viz = _make_newton_visualizer()
    # pre-cache the visualizer
    recorder._live_visualizer = newton_viz

    other_viz = _make_newton_visualizer(pos=(99.0, 99.0, 99.0))
    # replace sim.visualizers with a second Newton visualizer
    # if the cache is bypassed the recorder would use this one instead.
    recorder._scene.sim.visualizers = [other_viz]
    recorder._sync_newton_camera()
    position = recorder._capture.update_camera.call_args[0][0]
    assert position != (99.0, 99.0, 99.0)
    recorder._capture.update_camera.assert_called_once()


def test_sync_newton_camera_correct_position_forwarded():
    """_sync_newton_camera reads cam.pos and passes it as position to update_camera."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    newton_viz = _make_newton_visualizer(pos=(10.0, 20.0, 30.0), yaw_deg=0.0, pitch_deg=0.0)
    recorder._live_visualizer = newton_viz

    recorder._sync_newton_camera()

    args = recorder._capture.update_camera.call_args
    position = args[0][0]
    assert position == (10.0, 20.0, 30.0)


def test_sync_newton_camera_target_derived_from_pitch_yaw():
    """Target is reconstructed from pitch/yaw and is unit-distance from position."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    pos = (0.0, 0.0, 0.0)
    yaw_deg, pitch_deg = 0.0, 0.0  # looking along +X at horizon
    newton_viz = _make_newton_visualizer(pos=pos, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    recorder._live_visualizer = newton_viz

    recorder._sync_newton_camera()

    args = recorder._capture.update_camera.call_args[0]
    position, target = args
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    dz = target[2] - position[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    assert abs(dist - 1.0) < 1e-6
    assert abs(dx - 1.0) < 1e-6
    assert abs(dy) < 1e-6
    assert abs(dz) < 1e-6


def test_sync_newton_camera_no_visualizer_does_not_raise():
    """_sync_newton_camera silently skips when no Newton visualizer is registered."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    recorder._scene.sim.visualizers = []
    recorder._sync_newton_camera()  # must not raise
    recorder._capture.update_camera.assert_not_called()


def test_sync_newton_camera_skips_non_newton_visualizers():
    """_sync_newton_camera ignores visualizers whose type is not 'newton'."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    kit_viz = MagicMock()
    kit_viz.cfg.visualizer_type = "kit"
    recorder._scene.sim.visualizers = [kit_viz]
    recorder._sync_newton_camera()
    recorder._capture.update_camera.assert_not_called()


def test_sync_newton_camera_skips_when_viewer_is_none():
    """_sync_newton_camera skips camera update when _viewer is None (headless fallback)."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    viz = MagicMock()
    viz.cfg.visualizer_type = "newton"
    viz._viewer = None
    recorder._live_visualizer = viz
    recorder._sync_newton_camera()
    recorder._capture.update_camera.assert_not_called()


def test_sync_newton_camera_called_per_frame():
    """_sync_newton_camera (and thus update_camera) is called on every render step."""
    recorder = _create_recorder(_backend="newton_gl", _matched_visualizer="newton")
    newton_viz = _make_newton_visualizer()
    recorder._live_visualizer = newton_viz

    for _ in range(4):
        recorder.render_rgb_array()

    assert recorder._capture.update_camera.call_count == 4
