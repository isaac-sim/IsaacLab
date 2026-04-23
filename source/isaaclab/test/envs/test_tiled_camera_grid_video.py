# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for TiledCameraGridVideoCapture.

These tests run without a full isaaclab install: the module is loaded directly
from source and all Isaac Sim / torch / isaaclab dependencies are mocked out.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the module under test without requiring the isaaclab package to be
# installed.  The file has only stdlib + numpy at module-level; all Isaac Sim
# imports are lazy (inside methods) so they can be mocked per-test.
# ---------------------------------------------------------------------------
_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "isaaclab" / "envs" / "utils" / "tiled_camera_grid_video.py"
)
_spec = importlib.util.spec_from_file_location("_tcgv", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

TiledCameraGridVideoCapture = _module.TiledCameraGridVideoCapture
_tiled_camera_renderer_type = _module._tiled_camera_renderer_type
_tiled_camera_has_rgb_cfg = _module._tiled_camera_has_rgb_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCamera:
    """Minimal stand-in for isaaclab.sensors.camera.Camera.

    Using a real class (not MagicMock) is required so that
    ``isinstance(sensor, Camera)`` checks inside _find_video_camera work
    correctly when Camera is patched to this class.
    """

    pass


def _fake_camera_module():
    """Return a mock of isaaclab.sensors.camera with Camera = FakeCamera."""
    m = MagicMock()
    m.Camera = FakeCamera
    return m


def _make_rgb_tensor(n_envs: int, h: int = 8, w: int = 8):
    """Return a numpy array shaped (n_envs, h, w, 3) as a fake torch tensor."""
    arr = np.zeros((n_envs, h, w, 3), dtype=np.uint8)
    # Wrap in an object that behaves like a torch tensor for the code under test.
    tensor = MagicMock()
    tensor.shape = arr.shape
    tensor.__getitem__ = lambda self, key: _SlicedTensor(arr[key])
    tensor.contiguous.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.numpy.return_value = arr
    return tensor


class _SlicedTensor:
    """Minimal slice result that supports .contiguous().cpu().numpy()."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr
        self.shape = arr.shape

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, key):
        return _SlicedTensor(self._arr[key])


def _make_sensor(*, rgb: bool = True, renderer_type: str = "default", h: int = 8, w: int = 8, n_envs: int = 4):
    """Return a FakeCamera instance with data/cfg attributes set up."""
    sensor = FakeCamera()
    if rgb:
        tensor = _make_rgb_tensor(n_envs, h, w)
        sensor.data = SimpleNamespace(output={"rgb": tensor})
    else:
        sensor.data = SimpleNamespace(output={})
    sensor.is_initialized = True
    sensor.cfg = SimpleNamespace(
        data_types=["rgb"] if rgb else [],
        renderer_cfg=SimpleNamespace(renderer_type=renderer_type),
    )
    return sensor


def _make_capture(*, scene_sensors=None, fallback=None, preferred=None, num_tiles=-1):
    """Construct a TiledCameraGridVideoCapture with __init__ bypassed."""
    cap = object.__new__(TiledCameraGridVideoCapture)
    cap._scene = MagicMock()
    cap._scene.sensors = scene_sensors or {}
    cap._video_num_tiles = num_tiles
    cap._preferred_renderer_types = preferred
    cap._fallback_tiled_camera = fallback
    return cap


# ---------------------------------------------------------------------------
# _tiled_camera_renderer_type / _tiled_camera_has_rgb_cfg helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_renderer_type_returns_default_when_no_cfg(self):
        assert _tiled_camera_renderer_type(object()) == "default"

    def test_renderer_type_reads_from_renderer_cfg(self):
        sensor = SimpleNamespace(cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="newton_warp")))
        assert _tiled_camera_renderer_type(sensor) == "newton_warp"

    def test_has_rgb_cfg_true_for_rgb(self):
        sensor = SimpleNamespace(cfg=SimpleNamespace(data_types=["rgb", "depth"]))
        assert _tiled_camera_has_rgb_cfg(sensor) is True

    def test_has_rgb_cfg_true_for_rgba(self):
        sensor = SimpleNamespace(cfg=SimpleNamespace(data_types=["rgba"]))
        assert _tiled_camera_has_rgb_cfg(sensor) is True

    def test_has_rgb_cfg_false_for_depth_only(self):
        sensor = SimpleNamespace(cfg=SimpleNamespace(data_types=["depth"]))
        assert _tiled_camera_has_rgb_cfg(sensor) is False

    def test_has_rgb_cfg_false_when_no_cfg(self):
        assert _tiled_camera_has_rgb_cfg(object()) is False


# ---------------------------------------------------------------------------
# _find_video_camera — no preferred_renderer_types
# ---------------------------------------------------------------------------


class TestFindVideoCameraUnfiltered:
    """Tests for _find_video_camera when preferred_renderer_types is None."""

    def test_finds_camera_instance_in_scene(self):
        """Critical fix for PR #5162: isinstance(sensor, Camera) must match Camera instances."""
        sensor = _make_sensor(rgb=True)
        cap = _make_capture(scene_sensors={"cam": sensor})

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is sensor

    def test_skips_non_camera_sensors(self):
        """Non-Camera sensors (e.g. ContactSensor) must be ignored."""
        non_camera = MagicMock(spec=object)  # not a FakeCamera instance
        cap = _make_capture(scene_sensors={"contact": non_camera})

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is None

    def test_skips_camera_without_rgb_output(self):
        """A Camera with no rgb/rgba in data.output is not selected."""
        sensor = _make_sensor(rgb=False)
        cap = _make_capture(scene_sensors={"cam": sensor})

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is None

    def test_uses_fallback_when_no_scene_camera(self):
        """Falls back to _fallback_tiled_camera when scene has no suitable Camera."""
        fallback = _make_sensor(rgb=True)
        cap = _make_capture(scene_sensors={}, fallback=fallback)

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is fallback

    def test_scene_camera_preferred_over_fallback(self):
        """A valid scene Camera is chosen over the fallback."""
        scene_sensor = _make_sensor(rgb=True)
        fallback = _make_sensor(rgb=True)
        cap = _make_capture(scene_sensors={"cam": scene_sensor}, fallback=fallback)

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is scene_sensor

    def test_result_cached_on_second_call(self):
        """_find_video_camera caches its result in _video_camera."""
        sensor = _make_sensor(rgb=True)
        cap = _make_capture(scene_sensors={"cam": sensor})

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            first = cap._find_video_camera()
            second = cap._find_video_camera()  # should hit the cache

        assert first is second is sensor


# ---------------------------------------------------------------------------
# _find_video_camera — with preferred_renderer_types
# ---------------------------------------------------------------------------


class TestFindVideoCameraFiltered:
    """Tests for _find_video_camera when preferred_renderer_types is set."""

    def test_finds_camera_with_matching_renderer(self):
        sensor = _make_sensor(rgb=True, renderer_type="isaac_rtx")
        cap = _make_capture(scene_sensors={"cam": sensor}, preferred=("isaac_rtx", "ovrtx"))

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is sensor

    def test_skips_camera_with_wrong_renderer(self):
        sensor = _make_sensor(rgb=True, renderer_type="newton_warp")
        cap = _make_capture(scene_sensors={"cam": sensor}, preferred=("isaac_rtx",))

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            with pytest.raises(RuntimeError, match="no Camera with RGB"):
                cap._find_video_camera()

    def test_raises_when_no_matching_camera(self):
        cap = _make_capture(scene_sensors={}, preferred=("isaac_rtx",))

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            with pytest.raises(RuntimeError, match="no Camera with RGB"):
                cap._find_video_camera()

    def test_fallback_included_in_candidates(self):
        """Fallback camera with matching renderer is a valid candidate."""
        fallback = _make_sensor(rgb=True, renderer_type="newton_warp")
        cap = _make_capture(scene_sensors={}, fallback=fallback, preferred=("newton_warp",))

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result is fallback

    def test_both_renderers_accepted(self):
        sensor_rtx = _make_sensor(rgb=True, renderer_type="isaac_rtx")
        sensor_ovrtx = _make_sensor(rgb=True, renderer_type="ovrtx")
        cap = _make_capture(
            scene_sensors={"rtx": sensor_rtx, "ovrtx": sensor_ovrtx},
            preferred=("isaac_rtx", "ovrtx"),
        )

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            result = cap._find_video_camera()

        assert result in (sensor_rtx, sensor_ovrtx)


# ---------------------------------------------------------------------------
# render_rgb_array — grid layout
# ---------------------------------------------------------------------------


class TestRenderRgbArray:
    def _primed_capture(self, n_envs: int, h: int, w: int, num_tiles: int = -1):
        """Return a capture already past _find_video_camera (camera pre-set)."""

        sensor = _make_sensor(rgb=True, n_envs=n_envs, h=h, w=w)
        cap = _make_capture(scene_sensors={"cam": sensor}, num_tiles=num_tiles)

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            cap._find_video_camera()

        return cap, sensor

    def test_output_shape_4_envs(self):
        """4 envs on 8x8 tiles → 2×2 grid → 16×16 output."""
        cap, _ = self._primed_capture(n_envs=4, h=8, w=8)
        out = cap.render_rgb_array()
        assert out.shape == (16, 16, 3)

    def test_output_shape_1_env(self):
        """1 env on 4x4 tile → 1×1 grid → 4×4 output."""
        cap, _ = self._primed_capture(n_envs=1, h=4, w=4)
        out = cap.render_rgb_array()
        assert out.shape == (4, 4, 3)

    def test_output_shape_non_square_envs(self):
        """3 envs → ceil(sqrt(3))=2 grid → 2×2 with 1 blank pad → 2H×2W."""
        cap, _ = self._primed_capture(n_envs=3, h=6, w=6)
        out = cap.render_rgb_array()
        assert out.shape == (12, 12, 3)

    def test_num_tiles_caps_envs(self):
        """video_num_tiles=2 with 4 available → 2×1 grid (ceil(sqrt(2))=2) → 2H×2W."""
        cap, _ = self._primed_capture(n_envs=4, h=8, w=8, num_tiles=2)
        out = cap.render_rgb_array()
        # ceil(sqrt(2)) = 2 → 16×16
        assert out.shape == (16, 16, 3)

    def test_raises_when_no_camera_found(self):
        """render_rgb_array raises RuntimeError when _find_video_camera returns None."""
        cap = _make_capture(scene_sensors={})

        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            with pytest.raises(RuntimeError, match="no Camera sensor"):
                cap.render_rgb_array()

    def test_rgba_stripped_to_rgb(self):
        """rgba output has the alpha channel stripped before gridding."""
        n_envs, h, w = 1, 4, 4
        arr_rgba = np.ones((n_envs, h, w, 4), dtype=np.uint8) * 255

        tensor = MagicMock()
        tensor.shape = arr_rgba.shape

        def _getitem(self_inner, key):
            sliced = arr_rgba[key]
            t = MagicMock()
            t.shape = sliced.shape
            t.__getitem__ = lambda s, k: _SlicedTensor(sliced[k])
            t.contiguous.return_value = t
            t.cpu.return_value = t
            t.numpy.return_value = sliced
            return t

        tensor.__getitem__ = _getitem

        sensor = FakeCamera()
        sensor.data = SimpleNamespace(output={"rgba": tensor})
        sensor.is_initialized = True
        sensor.cfg = SimpleNamespace(
            data_types=["rgba"],
            renderer_cfg=SimpleNamespace(renderer_type="default"),
        )

        cap = _make_capture(scene_sensors={"cam": sensor})
        with patch.dict(sys.modules, {"isaaclab.sensors.camera": _fake_camera_module()}):
            cap._find_video_camera()

        out = cap.render_rgb_array()
        assert out.shape[-1] == 3


# ---------------------------------------------------------------------------
# _spawn_fallback_cameras — uses Camera, not TiledCamera
# ---------------------------------------------------------------------------


class TestSpawnFallbackCameras:
    def test_spawns_camera_not_tiled_camera(self):
        """After PR #5162, _spawn_fallback_cameras must instantiate Camera."""
        camera_cfg = MagicMock()
        camera_cfg.offset.rot = [0.0, 0.0, 0.0, 1.0]
        camera_cfg.offset.convention = "ros"
        camera_cfg.offset.pos = (0.0, 0.0, 2.0)
        camera_cfg.height = 64
        camera_cfg.width = 64
        camera_cfg.spawn.vertical_aperture = 1.0
        camera_cfg.spawn.horizontal_aperture = 1.0

        scene = MagicMock()
        scene.num_envs = 2

        fake_camera_instance = MagicMock()
        MockCamera = MagicMock(return_value=fake_camera_instance)

        fake_cam_module = MagicMock()
        fake_cam_module.Camera = MockCamera

        fake_math_module = MagicMock()
        fake_math_module.convert_camera_frame_orientation_convention.return_value = MagicMock(
            squeeze=lambda dim: MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([0.0, 0.0, 0.0, 1.0])))
        )

        fake_cfg_replaced = MagicMock()
        camera_cfg.replace.return_value = fake_cfg_replaced

        with patch.dict(
            sys.modules,
            {
                "isaaclab.sensors.camera": fake_cam_module,
                "isaaclab.utils.math": fake_math_module,
            },
        ):
            result = TiledCameraGridVideoCapture._spawn_fallback_cameras(camera_cfg, scene)

        # Camera() must have been called — not TiledCamera()
        MockCamera.assert_called_once_with(fake_cfg_replaced)
        assert result is fake_camera_instance
