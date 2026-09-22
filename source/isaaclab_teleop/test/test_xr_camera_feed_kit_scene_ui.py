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
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, Usd


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


class _EditContext:
    def __init__(self, stage, edit_target):
        self._stage = stage
        self._edit_target = edit_target
        self._previous_edit_target = None

    def __enter__(self):
        self._previous_edit_target = self._stage._test_edit_target
        self._stage._test_edit_target = self._edit_target
        events = getattr(self._stage, "_test_edit_context_events", None)
        if isinstance(events, list):
            events.append(("enter", self._edit_target))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stage._test_edit_target = self._previous_edit_target
        events = getattr(self._stage, "_test_edit_context_events", None)
        if isinstance(events, list):
            events.append(("exit", self._previous_edit_target))


class _ScenePartitionAttributeSpec:
    def __init__(self, default=None, *, has_default=False):
        self._default = default
        self._has_default = has_default

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self._default = value
        self._has_default = True

    def HasDefaultValue(self):
        return self._has_default

    def ClearDefaultValue(self):
        self._default = None
        self._has_default = False


class _ScenePartitionPrimSpec:
    def __init__(self, layer, prim_path):
        self._layer = layer
        self._prim_path = prim_path

    def RemoveProperty(self, attribute_spec):
        for path, candidate in tuple(self._layer.attributes.items()):
            if path.GetPrimPath() == self._prim_path and candidate is attribute_spec:
                del self._layer.attributes[path]


class _ScenePartitionLayer:
    def __init__(self):
        self.attributes = {}

    def GetAttributeAtPath(self, path):
        return self.attributes.get(Sdf.Path(path))

    def GetPrimAtPath(self, path):
        path = Sdf.Path(path)
        if any(attribute_path.GetPrimPath() == path for attribute_path in self.attributes):
            return _ScenePartitionPrimSpec(self, path)
        return None


class _ScenePartitionAttribute:
    def __init__(self, prim, name):
        self._prim = prim
        self._name = name

    @property
    def _path(self):
        return Sdf.Path(self._prim.path).AppendProperty(self._name)

    def IsValid(self):
        return self._prim.IsValid() and (
            self._name in self._prim.root_values or self._path in self._prim.stage.session_layer.attributes
        )

    def Get(self):
        spec = self._prim.stage.session_layer.GetAttributeAtPath(self._path)
        if spec is not None and spec.HasDefaultValue():
            return spec.default
        return self._prim.root_values.get(self._name)

    def Set(self, value):
        assert self._prim.stage._test_edit_target is self._prim.stage.session_layer
        spec = self._prim.stage.session_layer.GetAttributeAtPath(self._path)
        if spec is None:
            spec = _ScenePartitionAttributeSpec()
            self._prim.stage.session_layer.attributes[self._path] = spec
        spec.default = value
        self._prim.stage.set_counts[self._path] = self._prim.stage.set_counts.get(self._path, 0) + 1
        return True


class _ScenePartitionPrim:
    def __init__(self, stage, path, *, valid=False, root_values=None):
        self.stage = stage
        self.path = path
        self.valid = valid
        self.root_values = dict(root_values or {})

    def IsValid(self):
        return self.valid

    def GetAttribute(self, name):
        return _ScenePartitionAttribute(self, name)

    def CreateAttribute(self, name, value_type):
        assert value_type == Sdf.ValueTypeNames.Token
        assert self.stage._test_edit_target is self.stage.session_layer
        path = Sdf.Path(self.path).AppendProperty(name)
        self.stage.session_layer.attributes.setdefault(path, _ScenePartitionAttributeSpec())
        return _ScenePartitionAttribute(self, name)

    def RemoveProperty(self, name):
        assert self.stage._test_edit_target is self.stage.session_layer
        self.stage.session_layer.attributes.pop(Sdf.Path(self.path).AppendProperty(name), None)
        return True


class _ScenePartitionStage:
    def __init__(self):
        self.root_edit_target = object()
        self.session_layer = _ScenePartitionLayer()
        self._test_edit_target = self.root_edit_target
        self._test_edit_context_events = []
        self.prims = {}
        self.set_counts = {}

    def add_prim(self, path, *, root_values=None):
        prim = _ScenePartitionPrim(self, path, valid=True, root_values=root_values)
        self.prims[path] = prim
        return prim

    def remove_prim(self, path):
        self.prims.pop(path, None)

    def set_session_default(self, prim_path, attribute_name, value):
        path = Sdf.Path(prim_path).AppendProperty(attribute_name)
        self.session_layer.attributes[path] = _ScenePartitionAttributeSpec(value, has_default=True)

    def GetPrimAtPath(self, path):
        path = str(path)
        return self.prims.get(path, _ScenePartitionPrim(self, path))

    def GetSessionLayer(self):
        return self.session_layer


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
    loaded_module.Usd = SimpleNamespace(EditContext=_EditContext)
    return loaded_module


def _install_replicator(monkeypatch, annotator):
    replicator = _module("omni.replicator")
    replicator_core = _module("omni.replicator.core")
    get_annotator = Mock(return_value=annotator)
    replicator_core.AnnotatorRegistry = SimpleNamespace(get_annotator=get_annotator)
    replicator.core = replicator_core
    sys.modules["omni"].replicator = replicator
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator_core)
    return get_annotator


def test_scene_partition_authors_late_and_recreated_prims_once_in_session_layer(scene_ui_module):
    stage = _ScenePartitionStage()
    partition = scene_ui_module._KitSceneUiScenePartition(stage_getter=lambda: stage)
    camera_attr_path = Sdf.Path(scene_ui_module._XR_CAMERA_PATH).AppendProperty("omni:scenePartition")
    ui_attr_path = Sdf.Path(scene_ui_module._SCENE_UI_ROOT_PATH).AppendProperty("primvars:omni:scenePartition")

    token = partition.acquire()

    assert stage.session_layer.attributes == {}

    camera = stage.add_prim(scene_ui_module._XR_CAMERA_PATH)
    partition.refresh()

    assert camera.GetAttribute("omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert stage.set_counts == {camera_attr_path: 1}
    assert stage._test_edit_context_events == [
        ("enter", stage.session_layer),
        ("exit", stage.root_edit_target),
    ]
    assert stage._test_edit_target is stage.root_edit_target

    partition.refresh()
    assert stage.set_counts == {camera_attr_path: 1}

    ui_root = stage.add_prim(scene_ui_module._SCENE_UI_ROOT_PATH)
    partition.refresh()

    assert ui_root.GetAttribute("primvars:omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert stage.set_counts == {camera_attr_path: 1, ui_attr_path: 1}

    stage.remove_prim(scene_ui_module._XR_CAMERA_PATH)
    stage.remove_prim(scene_ui_module._SCENE_UI_ROOT_PATH)
    recreated_camera = stage.add_prim(scene_ui_module._XR_CAMERA_PATH)
    recreated_ui = stage.add_prim(scene_ui_module._SCENE_UI_ROOT_PATH)
    partition.refresh()

    assert recreated_camera.GetAttribute("omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert recreated_ui.GetAttribute("primvars:omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert stage.set_counts == {camera_attr_path: 1, ui_attr_path: 1}

    partition.release(token)

    assert stage.session_layer.attributes == {}


def test_scene_partition_real_usd_keeps_root_layer_clean_and_restores_session_opinions(
    scene_ui_module,
    monkeypatch,
):
    stage = Usd.Stage.CreateInMemory()
    camera = stage.DefinePrim(scene_ui_module._XR_CAMERA_PATH, "Camera")
    ui_root = stage.DefinePrim(scene_ui_module._SCENE_UI_ROOT_PATH, "Xform")
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        ui_root.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("")
    root_layer_before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(scene_ui_module, "Usd", Usd)
    partition = scene_ui_module._KitSceneUiScenePartition(stage_getter=lambda: stage)

    token = partition.acquire()

    assert stage.GetRootLayer().ExportToString() == root_layer_before
    assert camera.GetAttribute("omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert ui_root.GetAttribute("primvars:omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert session_layer.GetAttributeAtPath(camera.GetPath().AppendProperty("omni:scenePartition")) is not None

    partition.release(token)

    assert stage.GetRootLayer().ExportToString() == root_layer_before
    assert session_layer.GetAttributeAtPath(camera.GetPath().AppendProperty("omni:scenePartition")) is None
    assert ui_root.GetAttribute("primvars:omni:scenePartition").Get() == ""


def test_scene_partition_real_usd_teardown_restores_remaining_targets_after_failure(
    scene_ui_module,
    monkeypatch,
):
    stage = Usd.Stage.CreateInMemory()
    camera = stage.DefinePrim(scene_ui_module._XR_CAMERA_PATH, "Camera")
    ui_root = stage.DefinePrim(scene_ui_module._SCENE_UI_ROOT_PATH, "Xform")
    session_layer = stage.GetSessionLayer()
    monkeypatch.setattr(scene_ui_module, "Usd", Usd)
    partition = scene_ui_module._KitSceneUiScenePartition(stage_getter=lambda: stage)
    token = partition.acquire()
    restore_target = partition._restore_target
    camera_attribute_path = camera.GetPath().AppendProperty("omni:scenePartition")

    def fail_camera_restore(stage, session_layer, attribute_path, state):
        if attribute_path == camera_attribute_path:
            raise RuntimeError("camera restore failed")
        restore_target(stage, session_layer, attribute_path, state)

    monkeypatch.setattr(partition, "_restore_target", fail_camera_restore)
    with pytest.raises(RuntimeError, match="camera restore failed"):
        partition.release(token)

    assert session_layer.GetAttributeAtPath(camera_attribute_path).default == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert session_layer.GetAttributeAtPath(ui_root.GetPath().AppendProperty("primvars:omni:scenePartition")) is None
    assert partition._authored == {}
    assert partition._current_stage is None
    assert partition._registrations == set()


def test_scene_partition_multiple_panels_share_owner_and_restore_prior_state(scene_ui_module):
    stage = _ScenePartitionStage()
    camera = stage.add_prim(
        scene_ui_module._XR_CAMERA_PATH,
        root_values={"omni:scenePartition": ""},
    )
    ui_root = stage.add_prim(scene_ui_module._SCENE_UI_ROOT_PATH)
    stage.set_session_default(
        scene_ui_module._SCENE_UI_ROOT_PATH,
        "primvars:omni:scenePartition",
        "",
    )
    partition = scene_ui_module._KitSceneUiScenePartition(stage_getter=lambda: stage)

    first = partition.acquire()
    second = partition.acquire()

    assert camera.GetAttribute("omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION
    assert ui_root.GetAttribute("primvars:omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION

    partition.release(first)

    assert camera.GetAttribute("omni:scenePartition").Get() == scene_ui_module._XR_CAMERA_PIP_PARTITION

    partition.release(second)

    assert camera.GetAttribute("omni:scenePartition").Get() == ""
    assert ui_root.GetAttribute("primvars:omni:scenePartition").Get() == ""
    assert stage._test_edit_target is stage.root_edit_target


@pytest.mark.parametrize(
    ("prim_path", "attribute_name"),
    [
        ("/_xr/stage/xrCamera", "omni:scenePartition"),
        ("/ui", "primvars:omni:scenePartition"),
    ],
)
def test_scene_partition_refuses_to_replace_nonempty_target_partition(
    scene_ui_module,
    prim_path,
    attribute_name,
):
    stage = _ScenePartitionStage()
    for target_path, target_attribute in scene_ui_module._KitSceneUiScenePartition._TARGETS:
        stage.add_prim(
            target_path,
            root_values={target_attribute: "existing_partition"} if target_path == prim_path else None,
        )
    partition = scene_ui_module._KitSceneUiScenePartition(stage_getter=lambda: stage)

    with pytest.raises(RuntimeError, match="cannot replace existing scene partition"):
        partition.acquire()

    assert stage.GetPrimAtPath(prim_path).GetAttribute(attribute_name).Get() == "existing_partition"
    assert stage.session_layer.attributes == {}
    assert partition._registrations == set()


def test_scene_partition_cleanup_preserves_external_changes(scene_ui_module):
    stage = _ScenePartitionStage()
    stage.add_prim(scene_ui_module._XR_CAMERA_PATH)
    partition = scene_ui_module._KitSceneUiScenePartition(stage_getter=lambda: stage)
    token = partition.acquire()

    stage.set_session_default(scene_ui_module._XR_CAMERA_PATH, "omni:scenePartition", "external_partition")
    partition.release(token)

    camera = stage.GetPrimAtPath(scene_ui_module._XR_CAMERA_PATH)
    assert camera.GetAttribute("omni:scenePartition").Get() == "external_partition"


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


def test_head_locked_follows_full_viewer_pose_in_pose_local_axes(scene_ui_module):
    flags = _PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID
    first_pose = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(1.0, 2.0, 3.0))
    device = _InputDevice(SimpleNamespace(validity_flags=flags, pose_matrix=first_pose))
    core = _XrCore(input_device=device, meters_per_unit=0.01)
    anchor = scene_ui_module.KitSceneUiHeadLockedAnchor(core)
    source = _TransformSource(Gf.Matrix4d(1.0))
    visibility = []

    token = anchor.register(source, (0.1, -0.2), 0.8, visibility.append)
    core.message_bus.emit(_EventType.post_sync_update)

    first_expected = Gf.Matrix4d().SetTranslate(Gf.Vec3d(10.0, -20.0, -80.0)) * first_pose
    np.testing.assert_allclose(_matrix_values(source.source), _matrix_values(first_expected))
    assert visibility == [False, True]
    assert core.reorient_calls == []

    second_pose = Gf.Matrix4d(
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (4.0, 5.0, 6.0, 1.0),
    )
    device.pose_desc = SimpleNamespace(validity_flags=flags, pose_matrix=second_pose)
    core.message_bus.emit(_EventType.post_sync_update)

    second_expected = Gf.Matrix4d().SetTranslate(Gf.Vec3d(10.0, -20.0, -80.0)) * second_pose
    np.testing.assert_allclose(_matrix_values(source.source), _matrix_values(second_expected))
    assert visibility == [False, True]
    assert core.reorient_calls == []
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
    head_locked_anchor = scene_ui_module.KitSceneUiHeadLockedAnchor(core) if placement == "head_locked" else None

    panel = scene_ui_module.KitSceneUiCameraFeedPanel(
        descriptor,
        image_width=720,
        image_height=480,
        viewer_start_anchor=viewer_start_anchor,
        head_locked_anchor=head_locked_anchor,
    )

    kwargs = component_type.call_args.kwargs
    assert kwargs["width"] == pytest.approx(expected_width)
    assert kwargs["height"] == pytest.approx(expected_height)
    assert kwargs["resolution_scale"] == pytest.approx(1500.0)
    assert kwargs["unit_to_pixel_scale"] == pytest.approx(expected_meters_per_unit)
    if placement == "head_locked":
        core.input_device = _InputDevice(
            _pose(_PoseValidityFlags.POSITION_VALID | _PoseValidityFlags.ORIENTATION_VALID)
        )
        core.display_enabled = True
        core.message_bus.emit(_EventType.post_sync_update)
        assert len(panel._container.space_stack) == 1
        assert _translation(panel._container.space_stack[0].source) == pytest.approx((5.0, -10.0, -40.0))
    elif placement == "world":
        assert _translation(panel._container.space_stack[0].source) == pytest.approx((2.2, 3.6, 6.0))
    panel.close()


@pytest.mark.parametrize("use_scene_partition", [False, True])
def test_presenters_share_module_scene_partition_lifecycle(scene_ui_module, monkeypatch, use_scene_partition):
    core = _XrCore(display_enabled=False)
    monkeypatch.setattr(scene_ui_module, "XRCore", SimpleNamespace(get_singleton=lambda: core))
    monkeypatch.setattr(scene_ui_module, "WidgetComponent", Mock(return_value=object()))
    monkeypatch.setattr(scene_ui_module.ui, "ByteImageProvider", Mock(return_value=object()), raising=False)

    class _Container:
        def __init__(self, _view, _component, *, space_stack):
            del space_stack
            self.root = SimpleNamespace(clear=Mock())
            self.show = Mock()
            self.hide = Mock()

    monkeypatch.setattr(scene_ui_module, "UiContainer", _Container)
    scene_partition = Mock()
    scene_partition.acquire.side_effect = [101, 102]
    monkeypatch.setattr(scene_ui_module, "_shared_scene_partition", scene_partition)
    get_partition = Mock(return_value=scene_partition)
    monkeypatch.setattr(scene_ui_module, "_get_shared_scene_partition", get_partition)
    first_presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()
    second_presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()
    descriptor = SimpleNamespace(
        label=None,
        width_m=0.48,
        offset_m=(0.0, 0.0),
        distance_m=0.8,
        placement="head_locked",
        world_position_m=None,
        world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        use_scene_partition=use_scene_partition,
    )

    first = first_presenter.create_panel(descriptor, 720, 480)
    second = second_presenter.create_panel(descriptor, 720, 480)

    assert first_presenter._scene_partition is (scene_partition if use_scene_partition else None)
    assert second_presenter._scene_partition is (scene_partition if use_scene_partition else None)
    assert get_partition.call_count == (2 if use_scene_partition else 0)
    assert scene_partition.acquire.call_count == (2 if use_scene_partition else 0)
    assert scene_partition.refresh.call_count == (2 if use_scene_partition else 0)

    first.close()
    second.close()

    assert scene_partition.release.call_args_list == ([call(101), call(102)] if use_scene_partition else [])


def test_presenter_frame_subscription_refreshes_partition_and_always_updates_feed(
    scene_ui_module,
    monkeypatch,
    caplog,
):
    subscription = object()
    subscribe = Mock(return_value=subscription)
    monkeypatch.setattr(scene_ui_module, "_subscribe_to_kit_frame_updates", subscribe)
    events = []
    scene_partition = Mock()
    scene_partition.refresh.side_effect = lambda: events.append("partition")
    feed_callback = Mock(side_effect=lambda _event: events.append("feed"))
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter(scene_partition)

    result = presenter.subscribe_to_frame_updates(feed_callback)
    wrapped_callback = subscribe.call_args.args[0]
    event = object()
    wrapped_callback(event)

    assert result is subscription
    assert events == ["partition", "feed"]
    feed_callback.assert_called_once_with(event)

    scene_partition.refresh.side_effect = RuntimeError("partition update failed")
    feed_callback.reset_mock()
    wrapped_callback(event)
    wrapped_callback(event)

    assert feed_callback.call_count == 2
    assert caplog.text.count("could not update its SceneUI scene partition") == 1

    scene_partition.refresh.side_effect = None
    wrapped_callback(event)
    scene_partition.refresh.side_effect = RuntimeError("partition update failed again")
    wrapped_callback(event)

    assert caplog.text.count("could not update its SceneUI scene partition") == 2


def test_panel_upload_uses_gpu_provider_for_any_cuda_device(scene_ui_module):
    provider = Mock()
    panel = object.__new__(scene_ui_module.KitSceneUiCameraFeedPanel)
    panel._closed = False
    panel._provider = provider
    image = SimpleNamespace(
        device=torch.device("cuda:1"),
        shape=(8, 12, 4),
        data_ptr=Mock(return_value=1234),
    )

    panel.upload(image)

    provider.set_bytes_data_from_gpu.assert_called_once_with(1234, [12, 8], "rgba8")
    provider.set_bytes_data.assert_not_called()


def test_panel_upload_uses_cpu_provider_without_cuda(scene_ui_module):
    provider = Mock()
    panel = object.__new__(scene_ui_module.KitSceneUiCameraFeedPanel)
    panel._closed = False
    panel._provider = provider
    image = torch.zeros((8, 12, 4), dtype=torch.uint8)

    panel.upload(image)

    provider.set_bytes_data.assert_called_once()
    assert provider.set_bytes_data.call_args.args[1:] == ([12, 8], "rgba8")
    provider.set_bytes_data_from_gpu.assert_not_called()


def test_image_source_is_absent_without_existing_render_product(scene_ui_module):
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", SimpleNamespace())

    assert source is None


def test_image_source_attaches_rgb_cuda_annotator_and_detaches(scene_ui_module, monkeypatch):
    annotator = Mock()
    raw_frame = object()
    annotator.get_data.return_value = raw_frame
    get_annotator = _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(
        _render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/ExistingCamera"))
    )
    fallback = torch.empty((8, 12, 4), dtype=torch.uint8)
    direct = SimpleNamespace(
        shape=fallback.shape,
        dtype=torch.uint8,
        device=torch.device("cuda:1"),
        is_contiguous=lambda: True,
        numel=fallback.numel,
    )
    convert = Mock(return_value=direct)
    monkeypatch.setattr(scene_ui_module, "convert_to_torch", convert)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", camera)
    assert source is not None
    image = source.get_image(tuple(fallback.shape))
    source.close()

    get_annotator.assert_called_once_with("rgb", device="cuda", do_array_copy=False)
    annotator.attach.assert_called_once_with(["/Render/ExistingCamera"])
    convert.assert_called_once_with(raw_frame)
    assert image is direct
    annotator.detach.assert_called_once_with(["/Render/ExistingCamera"])


def test_image_source_authors_feed_settings_after_annotator_attach(scene_ui_module, monkeypatch):
    events = []
    edit_context_events = []
    annotator = Mock()
    annotator.attach.side_effect = lambda paths: events.append(("attach", paths))
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    cfg = SimpleNamespace(enable_dlss_ray_reconstruction=False, dlss_exec_mode="quality")
    stage = Mock()
    root_edit_target = object()
    session_edit_target = object()
    stage._test_edit_target = root_edit_target
    stage._test_edit_context_events = edit_context_events
    stage.GetSessionLayer.return_value = session_edit_target
    render_product = stage.GetPrimAtPath.return_value
    render_product.IsValid.return_value = True
    render_product.GetTypeName.return_value = "RenderProduct"

    def apply_schema(schema):
        assert stage._test_edit_target is session_edit_target
        events.append(("apply", schema))
        return True

    render_product.ApplyAPI.side_effect = apply_schema
    attributes = {}
    for name in ("omni:rtx:newDenoiser:enabled", "omni:rtx:post:dlss:execMode"):
        attribute = Mock()
        attribute.IsValid.return_value = True

        def set_attribute(value, attribute_name=name):
            assert stage._test_edit_target is session_edit_target
            events.append(("set", attribute_name, value))
            return True

        attribute.Set.side_effect = set_attribute
        attributes[name] = attribute
    render_product.GetAttribute.side_effect = attributes.__getitem__
    monkeypatch.setattr(scene_ui_module, "get_current_stage", lambda: stage)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", camera, cfg)

    assert source is not None
    assert events == [
        ("attach", ["/Render/Camera"]),
        ("apply", "OmniRtxDebugSettingsAPI_1"),
        ("set", "omni:rtx:newDenoiser:enabled", False),
        ("apply", "OmniRtxSettingsRtAPI_1"),
        ("set", "omni:rtx:post:dlss:execMode", "quality"),
    ]
    assert edit_context_events == [("enter", session_edit_target), ("exit", root_edit_target)]
    assert stage._test_edit_target is root_edit_target
    stage.GetSessionLayer.assert_called_once_with()
    source.close()


def test_image_source_default_policy_does_not_access_stage(scene_ui_module, monkeypatch):
    annotator = Mock()
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    get_current_stage = Mock()
    monkeypatch.setattr(scene_ui_module, "get_current_stage", get_current_stage)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source(
        "robot_pov_cam",
        camera,
        SimpleNamespace(enable_dlss_ray_reconstruction=None, dlss_exec_mode=None),
    )

    assert source is not None
    get_current_stage.assert_not_called()
    source.close()


def test_image_source_keeps_cuda_source_when_optional_policy_fails(scene_ui_module, monkeypatch, caplog):
    annotator = Mock()
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    stage = Mock()
    root_edit_target = object()
    session_edit_target = object()
    stage._test_edit_target = root_edit_target
    stage.GetSessionLayer.return_value = session_edit_target
    render_product = stage.GetPrimAtPath.return_value
    render_product.IsValid.return_value = True
    render_product.GetTypeName.return_value = "RenderProduct"

    def reject_schema(_schema):
        assert stage._test_edit_target is session_edit_target
        return False

    render_product.ApplyAPI.side_effect = reject_schema
    monkeypatch.setattr(scene_ui_module, "get_current_stage", lambda: stage)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source(
        "robot_pov_cam",
        camera,
        SimpleNamespace(enable_dlss_ray_reconstruction=False, dlss_exec_mode=None),
    )

    assert source is not None
    assert "could not apply render-product settings" in caplog.text
    assert stage._test_edit_target is root_edit_target
    annotator.detach.assert_not_called()
    source.close()
    annotator.detach.assert_called_once_with(["/Render/Camera"])


def test_image_source_keeps_successful_partial_policy_when_second_field_fails(
    scene_ui_module,
    monkeypatch,
    caplog,
):
    annotator = Mock()
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    stage = Mock()
    render_product = stage.GetPrimAtPath.return_value
    render_product.IsValid.return_value = True
    render_product.GetTypeName.return_value = "RenderProduct"
    render_product.ApplyAPI.side_effect = [True, False]
    ray_reconstruction_attribute = render_product.GetAttribute.return_value
    ray_reconstruction_attribute.IsValid.return_value = True
    ray_reconstruction_attribute.Set.return_value = True
    monkeypatch.setattr(scene_ui_module, "get_current_stage", lambda: stage)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source(
        "robot_pov_cam",
        camera,
        SimpleNamespace(enable_dlss_ray_reconstruction=False, dlss_exec_mode="quality"),
    )

    assert source is not None
    ray_reconstruction_attribute.Set.assert_called_once_with(False)
    assert "ray_reconstruction=False, dlss_exec_mode='quality'" in caplog.text
    assert "OmniRtxSettingsRtAPI_1" in caplog.text
    annotator.detach.assert_not_called()
    source.close()
    annotator.detach.assert_called_once_with(["/Render/Camera"])


def test_presenter_keeps_cpu_image_and_warns_once_when_no_render_product_gpu_is_known(scene_ui_module, caplog):
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()
    image = torch.empty((8, 12, 4), dtype=torch.uint8)

    upload = presenter.prepare_upload_image("robot_pov_cam", image)
    repeated_upload = presenter.prepare_upload_image("robot_pov_cam", image)

    assert upload is image
    assert repeated_upload is image
    assert caplog.text.count("per-frame host transfer may reduce PiP performance") == 1


def test_presenter_stages_cpu_fallback_on_previous_render_product_gpu(scene_ui_module, monkeypatch):
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()
    image = torch.empty((8, 12, 4), dtype=torch.uint8)
    previous_source = SimpleNamespace(device=torch.device("cuda:1"))
    staging = SimpleNamespace(device=torch.device("cuda:1"))
    empty_like = Mock(return_value=staging)
    monkeypatch.setattr(torch, "empty_like", empty_like)

    upload = presenter.prepare_upload_image(
        "robot_pov_cam",
        image,
        previous_source=previous_source,
    )

    assert upload is staging
    empty_like.assert_called_once_with(
        image,
        device=torch.device("cuda:1"),
        memory_format=torch.contiguous_format,
    )


def test_replicator_output_uses_dlpack_without_array_conversion(scene_ui_module, monkeypatch):
    output = SimpleNamespace(__dlpack__=Mock())
    direct = object()
    from_dlpack = Mock(return_value=direct)
    convert = Mock()
    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", from_dlpack)
    monkeypatch.setattr(scene_ui_module, "convert_to_torch", convert)

    image = scene_ui_module._replicator_output_to_torch(output)

    assert image is direct
    from_dlpack.assert_called_once_with(output)
    convert.assert_not_called()


def test_image_source_falls_back_when_annotator_has_no_frame(scene_ui_module, monkeypatch):
    annotator = Mock()
    annotator.get_data.return_value = None
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    fallback = torch.empty((8, 12, 4), dtype=torch.uint8)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", camera)
    assert source is not None

    assert source.get_image(tuple(fallback.shape)) is None
    source.close()


def test_image_source_treats_empty_annotator_frame_as_not_ready(scene_ui_module, monkeypatch, caplog):
    annotator = Mock()
    annotator.get_data.return_value = object()
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    fallback = torch.empty((8, 12, 4), dtype=torch.uint8)
    monkeypatch.setattr(
        scene_ui_module,
        "_replicator_output_to_torch",
        Mock(return_value=torch.empty(0, dtype=torch.uint8)),
    )
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", camera)

    assert source.get_image(tuple(fallback.shape)) is None
    assert "incompatible CUDA annotator frame" not in caplog.text
    source.close()


def test_image_source_falls_back_when_replicator_attach_fails(scene_ui_module, monkeypatch, caplog):
    annotator = Mock()
    annotator.attach.side_effect = RuntimeError("attach failed")
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", camera)

    assert source is None
    annotator.detach.assert_called_once_with(["/Render/Camera"])
    assert "Falling back to the Camera RGBA buffer" in caplog.text


def test_image_source_authors_feed_policy_when_replicator_attach_falls_back(
    scene_ui_module,
    monkeypatch,
):
    events = []
    annotator = Mock()

    def fail_attach(paths):
        events.append(("attach", paths))
        raise RuntimeError("attach failed")

    annotator.attach.side_effect = fail_attach
    annotator.detach.side_effect = lambda paths: events.append(("detach", paths))
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    stage = Mock()
    render_product = stage.GetPrimAtPath.return_value
    render_product.IsValid.return_value = True
    render_product.GetTypeName.return_value = "RenderProduct"
    render_product.ApplyAPI.side_effect = lambda schema: events.append(("apply", schema)) or True
    attribute = render_product.GetAttribute.return_value
    attribute.IsValid.return_value = True
    attribute.Set.side_effect = lambda value: events.append(("set", value)) or True
    monkeypatch.setattr(scene_ui_module, "get_current_stage", lambda: stage)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source(
        "robot_pov_cam",
        camera,
        SimpleNamespace(enable_dlss_ray_reconstruction=False, dlss_exec_mode=None),
    )

    assert source is None
    assert events == [
        ("attach", ["/Render/Camera"]),
        ("detach", ["/Render/Camera"]),
        ("apply", "OmniRtxDebugSettingsAPI_1"),
        ("set", False),
    ]


def test_image_source_falls_back_when_replicator_read_fails(scene_ui_module, monkeypatch, caplog):
    annotator = Mock()
    annotator.get_data.side_effect = RuntimeError("read failed")
    _install_replicator(monkeypatch, annotator)
    camera = SimpleNamespace(_render_data=SimpleNamespace(render_product=SimpleNamespace(path="/Render/Camera")))
    fallback = torch.empty((8, 12, 4), dtype=torch.uint8)
    presenter = scene_ui_module._KitSceneUiCameraFeedPresenter()

    source = presenter.create_image_source("robot_pov_cam", camera)
    assert source is not None

    assert source.get_image(tuple(fallback.shape)) is None
    assert source.get_image(tuple(fallback.shape)) is None
    assert caplog.text.count("could not read its CUDA annotator") == 1
    source.close()
