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

from isaaclab.envs.utils.video_recorder import VideoRecorder, _select_video_backend

pytestmark = pytest.mark.isaacsim_ci
_FRAME = np.zeros((8, 12, 3), dtype=np.uint8)
_DEFAULT_CFG = {
    "env_render_mode": "rgb_array",
    "eye": (7.5, 7.5, 7.5),
    "lookat": (0.0, 0.0, 0.0),
    "backend_source": "visualizer",
    "window_width": 1280,
    "window_height": 720,
}


class _FakeVisualizer:
    """Initialized visualizer with a recorder-facing framebuffer."""

    def __init__(
        self,
        visualizer_type: str,
        *,
        eye: tuple[float, float, float] = (1.0, 2.0, 3.0),
        lookat: tuple[float, float, float] = (4.0, 5.0, 6.0),
    ):
        self.cfg = SimpleNamespace(visualizer_type=visualizer_type, eye=eye, lookat=lookat)
        self.render_calls = 0

    def render_rgb_array(self) -> np.ndarray:
        self.render_calls += 1
        return _FRAME


def _make_cfg(**overrides):
    return SimpleNamespace(**(_DEFAULT_CFG | overrides))


def _make_visualizer_cfg(visualizer_type, eye=(1.0, 2.0, 3.0), lookat=(4.0, 5.0, 6.0)):
    return SimpleNamespace(visualizer_type=visualizer_type, eye=eye, lookat=lookat)


def _make_scene(visualizer_cfgs=(), physics_backend="PhysxPhysicsManager", renderer_types=()):
    scene = MagicMock()
    scene.sim._resolve_visualizer_cfgs.return_value = list(visualizer_cfgs)
    scene.sim.physics_manager.__name__ = physics_backend
    scene.sim.visualizers = []
    scene._sensor_renderer_types.return_value = list(renderer_types)
    return scene


def test_resolve_backend_prefers_kit_visualizer():
    """Kit has priority when both video-capable visualizers are configured."""
    newton_cfg = _make_visualizer_cfg("newton")
    kit_cfg = _make_visualizer_cfg("kit")
    scene = _make_scene([newton_cfg, kit_cfg])

    backend, visualizer_cfg = _select_video_backend(scene, "visualizer")

    assert backend == "kit"
    assert visualizer_cfg is kit_cfg


def test_resolve_backend_selects_newton_visualizer():
    """A configured Newton visualizer selects direct Newton framebuffer capture."""
    newton_cfg = _make_visualizer_cfg("newton")

    backend, visualizer_cfg = _select_video_backend(_make_scene([newton_cfg], "NewtonPhysicsManager"), "visualizer")

    assert backend == "newton_gl"
    assert visualizer_cfg is newton_cfg


def test_resolve_backend_renderer_source_ignores_visualizers():
    """Renderer source bypasses active visualizers."""
    newton_cfg = _make_visualizer_cfg("newton")

    backend, visualizer_cfg = _select_video_backend(_make_scene([newton_cfg], "PhysxPhysicsManager"), "renderer")

    assert backend == "kit"
    assert visualizer_cfg is None


def test_resolve_backend_unsupported_visualizer_falls_back():
    """A visualizer without RGB capture falls back to the renderer stack."""
    viser_cfg = _make_visualizer_cfg("viser")

    backend, visualizer_cfg = _select_video_backend(_make_scene([viser_cfg], "PhysxPhysicsManager"), "visualizer")

    assert backend == "kit"
    assert visualizer_cfg is None


@pytest.mark.parametrize(
    ("physics_backend", "renderer_types", "expected"),
    [
        ("PhysxPhysicsManager", (), "kit"),
        ("unknown", ("isaac_rtx",), "kit"),
        ("NewtonPhysicsManager", (), "newton_gl"),
        ("unknown", ("newton_warp",), "newton_gl"),
    ],
)
def test_resolve_backend_uses_renderer_stack(physics_backend, renderer_types, expected):
    """Physics and sensor renderers select the fallback capture backend."""
    backend, visualizer = _select_video_backend(
        _make_scene(physics_backend=physics_backend, renderer_types=renderer_types),
        "visualizer",
    )

    assert backend == expected
    assert visualizer is None


def test_resolve_backend_raises_without_supported_source():
    """An unsupported stack cannot provide RGB video."""
    with pytest.raises(RuntimeError, match="No supported backend detected"):
        _select_video_backend(_make_scene(physics_backend="unknown"), "visualizer")


def test_resolve_backend_rejects_invalid_source():
    """Only visualizer and renderer are valid backend sources."""
    with pytest.raises(ValueError, match="backend_source"):
        _select_video_backend(_make_scene(), "invalid")


def test_newton_visualizer_capture_is_bound_on_first_render():
    """Newton video reads the initialized visualizer framebuffer without creating another viewer."""
    newton_cfg = _make_visualizer_cfg("newton")
    newton = _FakeVisualizer("newton")
    scene = _make_scene([newton_cfg], "NewtonPhysicsManager")
    recorder = VideoRecorder(_make_cfg(), scene)

    assert recorder._capture is None
    scene.sim.visualizers = [newton]
    assert recorder.render_rgb_array() is _FRAME
    assert recorder._capture is newton

    scene.sim.visualizers = []
    assert recorder.render_rgb_array() is _FRAME
    assert newton.render_calls == 2


def test_init_creates_kit_capture_with_visualizer_camera():
    """Kit capture is created before reset using the selected visualizer camera."""
    kit_cfg = _make_visualizer_cfg("kit", eye=(1.0, 2.0, 3.0), lookat=(4.0, 5.0, 6.0))
    capture = MagicMock()
    capture.render_rgb_array.return_value = _FRAME
    create_capture = MagicMock(return_value=capture)
    capture_cfg = MagicMock()
    capture_cfg_type = MagicMock(return_value=capture_cfg)
    capture_module = SimpleNamespace(create_isaacsim_kit_perspective_video=create_capture)
    cfg_module = SimpleNamespace(IsaacsimKitPerspectiveVideoCfg=capture_cfg_type)

    with patch.dict(
        sys.modules,
        {
            "isaaclab_physx.video_recording.isaacsim_kit_perspective_video": capture_module,
            "isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg": cfg_module,
        },
    ):
        recorder = VideoRecorder(_make_cfg(), _make_scene([kit_cfg]))
        create_capture.assert_called_once_with(capture_cfg)
        frame = recorder.render_rgb_array()

    assert frame is _FRAME
    capture_cfg_type.assert_called_once_with(
        eye=(1.0, 2.0, 3.0),
        lookat=(4.0, 5.0, 6.0),
        window_width=1280,
        window_height=720,
    )


def test_renderer_source_creates_standalone_newton_capture():
    """Renderer-selected Newton video uses the recorder camera and standalone capture."""
    newton_cfg = _make_visualizer_cfg("newton")
    capture = MagicMock()
    capture.render_rgb_array.return_value = _FRAME
    create_capture = MagicMock(return_value=capture)
    capture_cfg = MagicMock()
    capture_cfg_type = MagicMock(return_value=capture_cfg)
    capture_module = SimpleNamespace(create_newton_gl_perspective_video=create_capture)
    cfg_module = SimpleNamespace(NewtonGlPerspectiveVideoCfg=capture_cfg_type)

    with patch.dict(
        sys.modules,
        {
            "isaaclab_newton.video_recording.newton_gl_perspective_video": capture_module,
            "isaaclab_newton.video_recording.newton_gl_perspective_video_cfg": cfg_module,
        },
    ):
        recorder = VideoRecorder(
            _make_cfg(backend_source="renderer"),
            _make_scene([newton_cfg], "NewtonPhysicsManager"),
        )
        create_capture.assert_called_once_with(capture_cfg)
        frame = recorder.render_rgb_array()

    assert frame is _FRAME
    capture_cfg_type.assert_called_once_with(
        window_width=1280,
        window_height=720,
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
    )


def test_newton_visualizer_capture_requires_initialized_visualizer():
    """The selected Newton visualizer must be initialized before its framebuffer is read."""
    newton_cfg = _make_visualizer_cfg("newton")
    recorder = VideoRecorder(_make_cfg(), _make_scene([newton_cfg], "NewtonPhysicsManager"))

    with pytest.raises(RuntimeError, match="not initialized"):
        recorder.render_rgb_array()


def test_render_rgb_array_is_disabled_for_other_render_modes():
    """The recorder remains dormant unless Gym requests RGB arrays."""
    recorder = VideoRecorder(_make_cfg(env_render_mode=None), _make_scene())

    assert recorder.render_rgb_array() is None
    assert recorder._capture is None


def test_capture_errors_propagate():
    """Capture failures are not hidden by VideoRecorder."""
    capture = MagicMock()
    capture.render_rgb_array.side_effect = RuntimeError("capture failed")
    recorder = VideoRecorder(_make_cfg(), _make_scene())
    recorder._capture = capture

    with pytest.raises(RuntimeError, match="capture failed"):
        recorder.render_rgb_array()
