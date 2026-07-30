# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
import importlib.util
import sys
from enum import IntFlag
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from pxr import Gf


class _PoseValidityFlags(IntFlag):
    POSITION_VALID = 1
    ORIENTATION_VALID = 2
    POSITION_TRACKED = 4
    ORIENTATION_TRACKED = 8


class _EventType:
    post_sync_update = "post_sync_update"
    xr_display_disabled = "xr_display_disabled"


class _TransformSource:
    def __init__(self, matrix: Gf.Matrix4d):
        self.source = matrix


class _SpatialSource:
    @staticmethod
    def new_transform_matrix_source(matrix: Gf.Matrix4d) -> _TransformSource:
        return _TransformSource(matrix)

    @staticmethod
    def new_prim_path_source(path: str):
        return SimpleNamespace(kind="prim_path", value=path)

    @staticmethod
    def new_translation_source(value):
        return SimpleNamespace(kind="translation", value=value)

    @staticmethod
    def new_look_at_camera_source():
        return SimpleNamespace(kind="look_at_camera")


class _Subscription:
    def __init__(self, bus: _MessageBus, token: int):
        self._bus = bus
        self._token = token

    def __del__(self):
        self._bus.release(self._token)


class _MessageBus:
    def __init__(self):
        self._next_token = 0
        self._listeners: dict[int, tuple[Any, Any]] = {}
        self.released_event_types: list[Any] = []

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    def create_subscription_to_pop_by_type(self, event_type, callback, *, name):
        del name
        token = self._next_token
        self._next_token += 1
        self._listeners[token] = (event_type, callback)
        return _Subscription(self, token)

    def release(self, token: int) -> None:
        listener = self._listeners.pop(token, None)
        if listener is not None:
            self.released_event_types.append(listener[0])

    def emit(self, event_type) -> None:
        for subscribed_type, callback in tuple(self._listeners.values()):
            if subscribed_type == event_type:
                callback(SimpleNamespace(type=event_type))


class _InputDevice:
    def __init__(self, pose_desc):
        self.pose_desc = pose_desc

    def get_virtual_world_pose_desc(self, path: str):
        assert path == ""
        return self.pose_desc


class _StageCoordinateSystem:
    def __init__(
        self,
        *,
        right: tuple[float, float, float] = (1.0, 0.0, 0.0),
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
        forward: tuple[float, float, float] = (0.0, 0.0, -1.0),
        meters_per_unit: float = 1.0,
        up_axis: str = "y",
    ):
        self._right = right
        self._up = up
        self._forward = forward
        self.meters_per_unit = meters_per_unit
        self.up_axis = up_axis

    def get_right_vector(self):
        return self._right

    def get_up_vector(self):
        return self._up

    def get_forward_vector(self):
        return self._forward


class _XrCore:
    def __init__(
        self,
        *,
        display_enabled: bool = True,
        input_device: _InputDevice | None = None,
        head_device: _InputDevice | None = None,
        meters_per_unit: float = 1.0,
        up_axis: str = "y",
        stage_coordinate_system: _StageCoordinateSystem | None = None,
    ):
        self.display_enabled = display_enabled
        self.input_device = input_device
        self.head_device = head_device
        self.coordinate_system = SimpleNamespace(meters_per_unit=meters_per_unit, up_axis=up_axis)
        self.stage_coordinate_system = stage_coordinate_system or _StageCoordinateSystem()
        self.message_bus = _MessageBus()
        self.reorient_calls: list[tuple[Gf.Matrix4d, bool]] = []

    def get_message_bus(self):
        return self.message_bus

    def is_xr_display_enabled(self):
        return self.display_enabled

    def get_input_device(self, path: str):
        if path == "displayDevice":
            return self.input_device
        if path == "/user/head":
            return self.head_device
        return None

    def get_coordinate_system(self):
        return self.coordinate_system

    def get_stage_coordinate_system(self):
        return self.stage_coordinate_system

    def reorient_transform_matrix_up_right(self, matrix: Gf.Matrix4d, y_up: bool):
        self.reorient_calls.append((matrix, y_up))
        return matrix


def _pose(flags: _PoseValidityFlags, translation=(0.0, 0.0, 0.0)):
    matrix = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*translation))
    return SimpleNamespace(validity_flags=flags, pose_matrix=matrix)


def _matrix_values(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)])


def _translation(matrix: Gf.Matrix4d) -> tuple[float, float, float]:
    return tuple(float(value) for value in matrix.ExtractTranslation())


def _module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


@pytest.fixture
def scene_ui_module(monkeypatch):
    """Load the Scene UI implementation against fake Kit and OpenXR modules."""
    omni = _module("omni")
    gpu_foundation = _module("omni.gpu_foundation_factory")
    ui = _module("omni.ui")
    kit = _module("omni.kit")
    scene_view = _module("omni.kit.scene_view")
    scene_view_xr = _module("omni.kit.scene_view.xr")
    scene_view_xr_utils = _module("omni.kit.scene_view.xr_utils")
    kit_xr = _module("omni.kit.xr")
    kit_xr_core = _module("omni.kit.xr.core")

    class _Widget:
        pass

    class _XrCoreType:
        @staticmethod
        def get_singleton():
            raise AssertionError("Tests must inject their XRCore instance.")

    gpu_foundation.TextureFormat = SimpleNamespace(RGBA8_UNORM="rgba8")
    ui.Widget = _Widget
    scene_view_xr.XRSceneView = object()
    scene_view_xr_utils.SpatialSource = _SpatialSource
    scene_view_xr_utils.UiContainer = object
    scene_view_xr_utils.UpdatePolicy = SimpleNamespace(ALWAYS="always")
    scene_view_xr_utils.WidgetComponent = object
    kit_xr_core.XRCore = _XrCoreType
    kit_xr_core.XRCoreEventType = _EventType
    kit_xr_core.XRPoseValidityFlags = _PoseValidityFlags

    omni.gpu_foundation_factory = gpu_foundation
    omni.ui = ui
    omni.kit = kit
    kit.scene_view = scene_view
    kit.xr = kit_xr
    scene_view.xr = scene_view_xr
    scene_view.xr_utils = scene_view_xr_utils
    kit_xr.core = kit_xr_core

    fake_modules = {
        "omni": omni,
        "omni.gpu_foundation_factory": gpu_foundation,
        "omni.ui": ui,
        "omni.kit": kit,
        "omni.kit.scene_view": scene_view,
        "omni.kit.scene_view.xr": scene_view_xr,
        "omni.kit.scene_view.xr_utils": scene_view_xr_utils,
        "omni.kit.xr": kit_xr,
        "omni.kit.xr.core": kit_xr_core,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).parents[1] / "isaaclab_teleop" / "camera_feed_kit_scene_ui.py"
    module_name = "_isaaclab_teleop_camera_feed_kit_scene_ui_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    loaded_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, loaded_module)
    spec.loader.exec_module(loaded_module)
    return loaded_module


@pytest.mark.parametrize(
    "input_device",
    [
        None,
        _InputDevice(_pose(_PoseValidityFlags.POSITION_VALID)),
    ],
)
def test_viewer_start_invalid_or_missing_pose_stays_hidden(scene_ui_module, input_device):
    core = _XrCore(input_device=input_device)
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    source = _TransformSource(Gf.Matrix4d(1.0))
    visibility = []

    token = anchor.register(source, (0.0, 0.0), 0.8, visibility.append)

    assert visibility == [False]
    assert not anchor.captured
    assert _translation(source.source) == pytest.approx((0.0, 0.0, 0.0))
    anchor.unregister(token)


def test_viewer_start_accepts_valid_pose_without_tracked_flags(scene_ui_module):
    flags = _PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID
    pose_desc = _pose(flags, translation=(1.0, 2.0, 3.0))
    core = _XrCore(input_device=_InputDevice(pose_desc))
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    source = _TransformSource(Gf.Matrix4d(1.0))
    visibility = []

    token = anchor.register(source, (0.0, 0.0), 0.0, visibility.append)

    assert visibility == [False]
    assert not anchor.captured

    core.message_bus.emit(_EventType.post_sync_update)

    assert visibility == [False, True]
    assert anchor.captured
    assert core.reorient_calls == [(pose_desc.pose_matrix, True)]
    assert _translation(source.source) == pytest.approx((1.0, 2.0, 3.0))
    anchor.unregister(token)


def test_viewer_start_freezes_first_valid_pose_across_later_motion(scene_ui_module):
    flags = _PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID
    device = _InputDevice(_pose(flags, translation=(1.0, 2.0, 3.0)))
    core = _XrCore(input_device=device)
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    source = _TransformSource(Gf.Matrix4d(1.0))

    token = anchor.register(source, (0.1, -0.2), 0.5, lambda _ready: None)
    core.message_bus.emit(_EventType.post_sync_update)
    frozen_matrix = _matrix_values(source.source).copy()
    device.pose_desc = _pose(flags, translation=(9.0, 8.0, 7.0))

    core.message_bus.emit(_EventType.post_sync_update)

    np.testing.assert_allclose(_matrix_values(source.source), frozen_matrix)
    assert len(core.reorient_calls) == 1
    anchor.unregister(token)


def test_viewer_start_panels_share_anchor_and_convert_offsets_to_stage_units(scene_ui_module):
    flags = _PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID
    pose_desc = _pose(flags)
    core = _XrCore(
        display_enabled=False,
        input_device=_InputDevice(pose_desc),
        meters_per_unit=0.01,
    )
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    first_source = _TransformSource(Gf.Matrix4d(1.0))
    second_source = _TransformSource(Gf.Matrix4d(1.0))
    first_visibility = []
    second_visibility = []

    first_token = anchor.register(first_source, (0.1, 0.2), 0.5, first_visibility.append)
    second_token = anchor.register(second_source, (-0.3, 0.4), 0.8, second_visibility.append)
    core.display_enabled = True
    core.message_bus.emit(_EventType.post_sync_update)

    assert anchor._upright_pose is pose_desc.pose_matrix
    assert len(core.reorient_calls) == 1
    assert first_visibility == [False, True]
    assert second_visibility == [False, True]
    assert _translation(first_source.source) == pytest.approx((10.0, 20.0, -50.0))
    assert _translation(second_source.source) == pytest.approx((-30.0, 40.0, -80.0))
    np.testing.assert_allclose(_matrix_values(first_source.source)[:3, :3], np.eye(3))
    np.testing.assert_allclose(_matrix_values(second_source.source)[:3, :3], np.eye(3))

    anchor.unregister(first_token)
    anchor.unregister(second_token)


def test_viewer_start_uses_pose_local_axes_in_z_up_stage(scene_ui_module):
    flags = _PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID
    # Local +X/+Y/+Z map to stage +X/+Z/-Y. The viewer therefore looks along
    # stage +Y from an eye height of 1.6 m.
    pose_matrix = Gf.Matrix4d(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 1.6, 1.0),
    )
    core = _XrCore(
        input_device=_InputDevice(SimpleNamespace(validity_flags=flags, pose_matrix=pose_matrix)),
        up_axis="z",
        stage_coordinate_system=_StageCoordinateSystem(
            up=(0.0, 0.0, 1.0),
            forward=(0.0, -1.0, 0.0),
            up_axis="z",
        ),
    )
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    source = _TransformSource(Gf.Matrix4d(1.0))

    token = anchor.register(source, (0.1, 0.2), 0.8, lambda _ready: None)
    core.message_bus.emit(_EventType.post_sync_update)

    assert _translation(source.source) == pytest.approx((0.1, 0.8, 1.8))
    np.testing.assert_allclose(_matrix_values(source.source)[:3, :3], _matrix_values(pose_matrix)[:3, :3])
    anchor.unregister(token)


def test_viewer_start_disconnect_hides_resets_and_recaptures(scene_ui_module):
    flags = _PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID
    device = _InputDevice(_pose(flags, translation=(1.0, 2.0, 3.0)))
    core = _XrCore(input_device=device)
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    source = _TransformSource(Gf.Matrix4d(1.0))
    visibility = []

    token = anchor.register(source, (0.0, 0.0), 0.0, visibility.append)
    core.message_bus.emit(_EventType.post_sync_update)
    first_matrix = _matrix_values(source.source).copy()
    core.display_enabled = False
    core.message_bus.emit(_EventType.xr_display_disabled)

    assert not anchor.captured
    assert visibility[-1] is False
    np.testing.assert_allclose(_matrix_values(source.source), first_matrix)

    core.message_bus.emit(_EventType.post_sync_update)
    assert visibility[-1] is False
    assert len(core.reorient_calls) == 1

    device.pose_desc = _pose(flags, translation=(4.0, 5.0, 6.0))
    core.display_enabled = True
    core.message_bus.emit(_EventType.post_sync_update)

    assert anchor.captured
    assert visibility == [False, True, False, True]
    assert len(core.reorient_calls) == 2
    assert _translation(source.source) == pytest.approx((4.0, 5.0, 6.0))
    anchor.unregister(token)


def test_viewer_start_unregister_releases_listeners_after_last_panel(scene_ui_module):
    core = _XrCore(display_enabled=False)
    anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core)
    first_source = _TransformSource(Gf.Matrix4d(1.0))
    second_source = _TransformSource(Gf.Matrix4d(1.0))

    first_token = anchor.register(first_source, (0.0, 0.0), 0.8, lambda _ready: None)
    second_token = anchor.register(second_source, (0.1, 0.1), 0.8, lambda _ready: None)
    assert core.message_bus.listener_count == 2

    anchor.unregister(first_token)
    assert core.message_bus.listener_count == 2

    anchor.unregister(second_token)
    gc.collect()

    assert anchor._post_sync_subscription is None
    assert anchor._display_disabled_subscription is None
    assert core.message_bus.listener_count == 0
    assert set(core.message_bus.released_event_types) == {
        _EventType.post_sync_update,
        _EventType.xr_display_disabled,
    }


def test_world_panel_matrix_is_fixed_and_normalizes_quaternion(scene_ui_module):
    scaled_descriptor = SimpleNamespace(
        world_position_m=(1.0, 2.0, 3.0),
        world_orientation_xyzw=(0.0, 0.0, 2.0, 2.0),
        offset_m=(0.5, -0.25),
    )
    unit = np.sqrt(0.5)
    unit_descriptor = SimpleNamespace(
        world_position_m=(1.0, 2.0, 3.0),
        world_orientation_xyzw=(0.0, 0.0, unit, unit),
        offset_m=(0.5, -0.25),
    )

    scaled_matrix = scene_ui_module._world_panel_matrix(scaled_descriptor, 0.01)
    repeated_matrix = scene_ui_module._world_panel_matrix(scaled_descriptor, 0.01)
    unit_matrix = scene_ui_module._world_panel_matrix(unit_descriptor, 0.01)

    np.testing.assert_allclose(_matrix_values(scaled_matrix), _matrix_values(repeated_matrix), atol=1.0e-12)
    np.testing.assert_allclose(_matrix_values(scaled_matrix), _matrix_values(unit_matrix), atol=1.0e-12)
    np.testing.assert_allclose(
        _matrix_values(scaled_matrix)[:3, :3],
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        atol=1.0e-12,
    )
    assert _translation(scaled_matrix) == pytest.approx((125.0, 250.0, 300.0))


@pytest.mark.parametrize(
    ("position", "orientation"),
    [
        (None, (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, float("nan"), 1.0)),
    ],
)
def test_world_panel_matrix_rejects_missing_position_or_invalid_quaternion(
    scene_ui_module,
    position,
    orientation,
):
    descriptor = SimpleNamespace(
        world_position_m=position,
        world_orientation_xyzw=orientation,
        offset_m=(0.0, 0.0),
    )

    with pytest.raises(ValueError):
        scene_ui_module._world_panel_matrix(descriptor, 1.0)


@pytest.mark.parametrize("meters_per_unit", [0.0, -1.0, float("nan")])
def test_world_panel_matrix_rejects_invalid_stage_scale(scene_ui_module, meters_per_unit):
    descriptor = SimpleNamespace(
        world_position_m=(0.0, 0.0, 0.0),
        world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        offset_m=(0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="meters_per_unit"):
        scene_ui_module._world_panel_matrix(descriptor, meters_per_unit)


@pytest.mark.parametrize(
    ("placement", "expected_meters_per_unit", "expected_width", "expected_height"),
    [
        ("viewer_start", 0.02, 24.0, 16.0),
        ("head_locked", 0.02, 24.0, 16.0),
        ("world", 0.5, 0.96, 0.64),
    ],
)
def test_panel_converts_metric_geometry_in_selected_coordinate_system(
    scene_ui_module,
    monkeypatch,
    placement,
    expected_meters_per_unit,
    expected_width,
    expected_height,
):
    core = _XrCore(
        display_enabled=False,
        meters_per_unit=0.02,
        stage_coordinate_system=_StageCoordinateSystem(meters_per_unit=0.5, up_axis="z"),
    )
    monkeypatch.setattr(scene_ui_module, "XRCore", SimpleNamespace(get_singleton=lambda: core))
    component = SimpleNamespace()
    component_type = Mock(return_value=component)
    monkeypatch.setattr(scene_ui_module, "WidgetComponent", component_type)
    monkeypatch.setattr(scene_ui_module.ui, "ByteImageProvider", Mock(return_value=object()), raising=False)

    class _Container:
        def __init__(self, _view, _component, *, space_stack):
            self.space_stack = space_stack
            self.root = SimpleNamespace(clear=Mock())
            self.show = Mock()
            self.hide = Mock()

    monkeypatch.setattr(scene_ui_module, "UiContainer", _Container)
    descriptor = SimpleNamespace(
        label=None,
        width_m=0.48,
        offset_m=(0.1, -0.2),
        distance_m=0.8,
        placement=placement,
        world_position_m=(1.0, 2.0, 3.0) if placement == "world" else None,
        world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    viewer_start_anchor = scene_ui_module.KitSceneUiViewerStartAnchor(core) if placement == "viewer_start" else None

    panel = scene_ui_module.KitSceneUiCameraFeedPanel(
        descriptor,
        image_width=720,
        image_height=480,
        viewer_start_anchor=viewer_start_anchor,
    )

    kwargs = component_type.call_args.kwargs
    assert kwargs["width"] == pytest.approx(expected_width)
    assert kwargs["height"] == pytest.approx(expected_height)
    assert kwargs["resolution_scale"] == pytest.approx(1500.0)
    assert kwargs["unit_to_pixel_scale"] == pytest.approx(expected_meters_per_unit)
    if placement == "head_locked":
        translation = panel._container.space_stack[1].value
        assert tuple(translation) == pytest.approx((5.0, -10.0, -40.0))
    elif placement == "world":
        assert _translation(panel._container.space_stack[0].source) == pytest.approx((2.2, 3.6, 6.0))
    panel.close()
