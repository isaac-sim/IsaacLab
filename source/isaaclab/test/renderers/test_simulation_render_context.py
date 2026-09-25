# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for simulation-owned renderers and their rendering orchestration."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.benchmark.stepping import RENDER_PROFILE_SCOPE, profile_renderers
from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.renderers.render_context import RenderContext
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, SensorBase
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sim import BackendCfg, SimulationContext
from isaaclab.utils.warp import ProxyArray

pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def _renderer(cfg):
    renderer = Mock(spec=BaseRenderer)
    renderer.visual_material_writer = None
    return renderer


@pytest.fixture
def sim():
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    sim._render_context = RenderContext(sim._backend_registry)
    return sim


def test_renderer_registry_sharing_and_early_clone_requirements(sim):
    constructor = Mock(side_effect=_renderer)
    cfg = IsaacRtxRendererCfg(class_type=constructor, cloning_contexts=("example:CloneContext",))
    renderer = sim.get_or_create_backend(cfg)

    assert sim.get_or_create_backend(cfg) is renderer
    assert sim.get_or_create_backend(cfg.replace()) is renderer
    constructor.assert_called_once_with(cfg)
    assert constructor.call_args.args[0] is cfg
    assert sim.render_context.clone_contexts == set(cfg.cloning_contexts)
    renderer.initialize.assert_not_called()

    assert sim.get_or_create_backend(cfg.replace(semantic_filter="class:robot")) is not renderer
    assert sim.get_or_create_backend(NewtonWarpRendererCfg(class_type=_renderer)) is not renderer
    sim.get_or_create_backend(BackendCfg(class_type=lambda cfg: object()))
    assert sim.render_context.renderer_types == ("isaac_rtx", "isaac_rtx", "newton_warp")


def test_renderer_initializes_once_before_or_after_physics_ready(sim):
    cfg = RendererCfg(class_type=_renderer)
    first = sim.get_or_create_backend(cfg)
    sim.get_or_create_backend(BackendCfg(class_type=lambda cfg: object()))
    sim.render_context.ensure_initialize()
    sim.render_context.ensure_initialize()
    first.initialize.assert_called_once_with()

    second_cfg = cfg.replace(renderer_type="second")
    second = sim.get_or_create_backend(second_cfg)
    second.initialize.assert_called_once_with()
    assert sim.get_or_create_backend(second_cfg) is second
    sim.render_context.ensure_initialize()
    first.initialize.assert_called_once_with()
    second.initialize.assert_called_once_with()


def test_conflicting_global_settings_are_rejected_before_construction(sim):
    constructor = Mock(side_effect=_renderer)
    cfg = IsaacRtxRendererCfg(class_type=constructor)
    renderer = sim.get_or_create_backend(cfg)
    conflicting = cfg.replace(global_settings=cfg.global_settings.replace(enable_shadows=False))

    with pytest.raises(ValueError, match="global settings differ"):
        sim.get_or_create_backend(conflicting)

    constructor.assert_called_once_with(cfg)
    assert sim.get_or_create_backend(cfg) is renderer
    assert sim.render_context.renderer_types == ("isaac_rtx",)


@pytest.mark.parametrize("has_materials", [False, True])
def test_finalized_consumers_allow_cache_hits_but_reject_new_renderers_with_materials(sim, has_materials):
    constructor = Mock(side_effect=_renderer)
    cfg = RendererCfg(class_type=constructor)
    renderer = sim.get_or_create_backend(cfg)
    if has_materials:
        sim.render_context.register_visual_material(
            SimpleNamespace(
                channels=("roughness",),
                _material_paths=("/World/Material",),
                _shader_paths=("/World/Material/Shader",),
                _input_names={"roughness": "roughness"},
                _values={"roughness": torch.zeros(1)},
                _offsets={},
            )
        )
    sim.render_context.finalize_consumers([])

    assert sim.get_or_create_backend(cfg.replace()) is renderer
    late_cfg = cfg.replace(renderer_type="late")
    if has_materials:
        with pytest.raises(RuntimeError, match="before rendering consumers are finalized"):
            sim.get_or_create_backend(late_cfg)
        constructor.assert_called_once_with(cfg)
    else:
        assert sim.get_or_create_backend(late_cfg) is not renderer
        assert constructor.call_count == 2


def test_close_backend_removes_renderer_from_orchestration(sim):
    cfg = RendererCfg(class_type=_renderer)
    renderer = sim.get_or_create_backend(cfg)
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.update_scene_state(1)

    sim.close_backend(renderer)
    renderer.close.assert_called_once_with()
    assert not sim.render_context.renderer_types
    sim.render_context.update_scene_state(2)
    renderer.update_transforms.assert_called_once_with()
    renderer.update_geometries.assert_called_once_with()
    with pytest.raises(RuntimeError, match="renderer must be registered"):
        sim.render_context.ensure_prepare_stage(None, 4)

    replacement = sim.get_or_create_backend(cfg)
    assert replacement is not renderer
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.update_scene_state(2)
    replacement.prepare_stage.assert_called_once_with(None, 4)
    replacement.update_transforms.assert_called_once_with()
    replacement.update_geometries.assert_called_once_with()
    sim.render_context.close()
    renderer.close.assert_called_once_with()
    replacement.close.assert_not_called()


def test_prepare_stage_is_idempotent_and_checks_env_count_until_reset(sim):
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.ensure_prepare_stage(None, 4)
    renderer.prepare_stage.assert_called_once_with(None, 4)
    with pytest.raises(RuntimeError, match="different num_envs"):
        sim.render_context.ensure_prepare_stage(None, 8)

    sim.render_context.reset_stage_prepare_flag()
    sim.render_context.ensure_prepare_stage(None, 8)
    assert renderer.prepare_stage.call_args_list == [call(None, 4), call(None, 8)]


def test_scene_state_does_not_skip_writes_within_a_physics_step(sim):
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    for step in (1, 1, 2):
        sim.render_context.update_scene_state(step)
    assert renderer.update_transforms.call_count == 3
    assert renderer.update_geometries.call_count == 3

    sim.render_context.update_scene_state(2)
    assert renderer.update_transforms.call_count == 4
    assert renderer.update_geometries.call_count == 4


@pytest.mark.parametrize("profile", [False, True])
def test_render_into_camera_call_order_and_profile_output(sim, capsys, profile):
    """Profiling preserves call order and collects timings without printing."""
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    data, camera = object(), CameraData()

    with profile_renderers(sim.render_context, active=profile) as timings:
        sim.render_context.render_into_camera(renderer, data, camera, physics_step_count=1)
        sim.render_context.render_into_camera(renderer, data, camera, physics_step_count=1)

    assert renderer.mock_calls == [
        call.update_transforms(),
        call.update_geometries(),
        call.render_batch([data]),
        call.read_output(data, camera),
        call.update_transforms(),
        call.update_geometries(),
        call.render_batch([data]),
        call.read_output(data, camera),
    ]
    assert len(timings) == (2 if profile else 0)
    assert all(scope == RENDER_PROFILE_SCOPE and elapsed >= 0.0 for scope, elapsed in timings)
    assert RENDER_PROFILE_SCOPE not in capsys.readouterr().out


def test_default_render_batch_preserves_single_camera_render_contract():
    """Existing render implementations handle batches in order, including empty batches."""
    renderer = SimpleNamespace(render=Mock())
    render_data = (object(), object())

    BaseRenderer.render_batch(renderer, ())
    renderer.render.assert_not_called()
    BaseRenderer.render_batch(renderer, render_data)
    assert renderer.render.call_args_list == [call(data) for data in render_data]


def test_render_into_cameras_groups_renderers_and_reads_each_output(sim):
    first = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    second = sim.get_or_create_backend(RendererCfg(class_type=_renderer, renderer_type="second"))
    data = [object() for _ in range(3)]
    cameras = [CameraData() for _ in data]

    sim.render_context.render_into_cameras([], physics_step_count=1)
    first.render_batch.assert_not_called()
    second.render_batch.assert_not_called()
    sim.render_context.render_into_cameras(
        [(first, data[0], cameras[0]), (second, data[1], cameras[1]), (first, data[2], cameras[2])],
        physics_step_count=1,
    )

    first.render_batch.assert_called_once_with([data[0], data[2]])
    second.render_batch.assert_called_once_with([data[1]])
    assert first.read_output.call_args_list == [call(data[0], cameras[0]), call(data[2], cameras[2])]
    second.read_output.assert_called_once_with(data[1], cameras[1])
    for renderer in (first, second):
        renderer.render.assert_not_called()
        renderer.update_transforms.assert_called_once_with()
        renderer.update_geometries.assert_called_once_with()


def test_legacy_render_profile_scope_warns_and_preserves_import():
    """The old scope import remains available during its deprecation period."""
    with pytest.warns(DeprecationWarning, match="isaaclab.benchmark.stepping.RENDER_PROFILE_SCOPE"):
        from isaaclab.renderers.render_context import RENDER_PROFILE_SCOPE as legacy_scope

    assert legacy_scope == RENDER_PROFILE_SCOPE


@pytest.mark.parametrize("fail_writer", [False, True])
def test_context_close_only_releases_writers_and_resets_bookkeeping(sim, fail_writer):
    cfg = RendererCfg(class_type=_renderer, cloning_contexts=("example:CloneContext",))
    renderer = sim.get_or_create_backend(cfg)
    context = sim.render_context
    context.ensure_initialize()
    context.ensure_prepare_stage(None, 4)
    context.update_scene_state(1)
    writers = (Mock(), Mock())
    if fail_writer:
        writers[0].close.side_effect = RuntimeError("writer failed")
    context._visual_material_writers = writers

    if fail_writer:
        with pytest.raises(RuntimeError, match=r"1 material writer\(s\) failed to close"):
            context.close()
    else:
        context.close()
    context.close()
    for writer in writers:
        writer.close.assert_called_once_with()
    renderer.close.assert_not_called()
    assert sim.get_or_create_backend(cfg) is renderer
    assert not context.clone_contexts

    context.ensure_initialize()
    context.ensure_prepare_stage(None, 8)
    context.update_scene_state(1)
    assert renderer.initialize.call_count == renderer.prepare_stage.call_count == 2
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 2


class _CpuCamera(Camera):
    """Exercise capture timing with CPU buffers and an in-memory pose source."""

    def __init__(self, renderer, name, update_period=0.0):
        self.cfg = SimpleNamespace(update_period=update_period, update_latest_camera_pose=True)
        self._device = "cpu"
        self._num_envs = 2
        self._is_initialized = True
        self._is_visualizing = False
        self._renderer = renderer
        self._render_data = SimpleNamespace(name=name, pose=None)
        self._data = CameraData()
        self._data.create_buffers(2, "cpu")
        self._data.info = {}
        self._frame = ProxyArray(wp.zeros(2, dtype=wp.int64, device="cpu"))
        self._ALL_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
        self._ALL_ENV_MASK = wp.ones(2, dtype=wp.bool, device="cpu")
        self._is_outdated = wp.ones(2, dtype=wp.bool, device="cpu")
        self._timestamp = wp.zeros(2, device="cpu")
        self._timestamp_last_update = wp.zeros(2, device="cpu")
        self._data_generation = 0
        self._data_generation_last_update = -1
        self.pose = 0.0
        self._view = SimpleNamespace(count=2, xform_world_space_writer=self._pose_writer)
        self.update(0.0)

    def __del__(self):
        pass

    @contextlib.contextmanager
    def _pose_writer(self):
        def set_poses(positions, orientations, indices):
            self.pose = float(positions.numpy()[0, 0])

        yield SimpleNamespace(set_poses=set_poses)

    def _update_poses(self, env_ids=None, env_mask=None, frame_op=0):
        self._render_data.pose = self.pose
        self._update_camera_state(env_ids=env_ids, env_mask=env_mask, frame_op=frame_op)


class _CpuSensor(SensorBase):
    """Exercise generic sensor updates without a rendering backend."""

    def __init__(self, name, batches=None):
        self.cfg = SimpleNamespace(update_period=0.0)
        self._device = "cpu"
        self._num_envs = 2
        self._is_initialized = True
        self._is_visualizing = False
        self._is_outdated = wp.ones(2, dtype=wp.bool, device="cpu")
        self._timestamp = wp.zeros(2, device="cpu")
        self._timestamp_last_update = wp.zeros(2, device="cpu")
        self._data_generation = 0
        self._data_generation_last_update = -1
        self._data = np.zeros(2, dtype=int)
        self.name = name
        self.batches = batches
        self.captures = []

    def __del__(self):
        pass

    @property
    def data(self):
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        pass

    def _update_buffers_impl(self, env_mask):
        mask = env_mask.numpy()
        self.captures.append(mask)
        self._data[mask] += 1


class _BatchSensor(_CpuSensor):
    @property
    def supports_batch_update(self):
        return True

    @staticmethod
    def _update_buffers_batch_impl(sensors):
        sensors[0].batches.append([sensor.name for sensor in sensors])
        for sensor in sensors:
            sensor._update_buffers_impl(sensor._is_outdated)


@pytest.fixture
def camera_batch_context(sim, monkeypatch):
    ctx = sim.render_context
    sim._physics_step_count = 1
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    renderer = sim.get_or_create_backend(NewtonWarpRendererCfg(class_type=_renderer))
    batches = []
    renderer.render.side_effect = lambda rd: batches.append([(rd.name, rd.pose)])
    renderer.render_batch = Mock(side_effect=lambda requests: batches.append([(rd.name, rd.pose) for rd in requests]))
    renderer.read_output.side_effect = lambda rd, data: data.info.update(pose=rd.pose)
    return ctx, renderer, batches


def _camera_scene(cameras, lazy=False, **sensors):
    return SimpleNamespace(
        sim=SimulationContext.instance(),
        cfg=SimpleNamespace(lazy_sensor_update=lazy),
        _sensors={**{camera._render_data.name: camera for camera in cameras}, **sensors},
        **{
            name: {}
            for name in (
                "_articulations",
                "_cable_objects",
                "_deformable_objects",
                "_rigid_objects",
                "_rigid_object_collections",
                "_surface_grippers",
            )
        },
    )


def test_camera_eager_updates_render_shared_batch_with_current_poses(camera_batch_context):
    """Eager scene updates batch initialized cameras and still update other sensor types."""
    _, renderer, batches = camera_batch_context
    cameras = [_CpuCamera(renderer, "wide"), _CpuCamera(renderer, "tele")]
    cameras[0].pose, cameras[1].pose = 1.0, 2.0
    uninitialized = _CpuCamera(renderer, "uninitialized")
    uninitialized._is_initialized = False
    other_sensor = Mock(supports_batch_update=False)
    scene = _camera_scene([*cameras, uninitialized], other=other_sensor)

    InteractiveScene.update(scene, 0.01)
    assert batches == [[("wide", 1.0), ("tele", 2.0)]]
    renderer.render.assert_not_called()
    other_sensor.update.assert_called_once_with(0.01, force_recompute=True)
    assert cameras[0].data.info["pose"] == 1.0
    assert cameras[1].data.info["pose"] == 2.0
    assert batches == [[("wide", 1.0), ("tele", 2.0)]]
    for camera in cameras:
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [1, 1])
        np.testing.assert_allclose(camera._timestamp_last_update.numpy(), [0.01, 0.01])
    np.testing.assert_array_equal(uninitialized.frame.warp.numpy(), [0, 0])


def test_camera_lazy_reads_leave_peer_cameras_outdated(camera_batch_context):
    """Reading a lazy camera captures only that camera, even when a peer shares its renderer."""
    _, renderer, batches = camera_batch_context
    cameras = [_CpuCamera(renderer, "wide"), _CpuCamera(renderer, "tele")]
    cameras[0].pose, cameras[1].pose = 1.0, 2.0
    other_sensor = Mock(supports_batch_update=False)
    scene = _camera_scene(cameras, lazy=True, other=other_sensor)

    InteractiveScene.update(scene, 0.01)
    assert not batches
    other_sensor.update.assert_called_once_with(0.01, force_recompute=False)
    assert cameras[0].data.info["pose"] == 1.0
    assert batches == [[("wide", 1.0)]]
    np.testing.assert_array_equal(cameras[1].frame.warp.numpy(), [0, 0])
    np.testing.assert_array_equal(cameras[1]._timestamp_last_update.numpy(), [0.0, 0.0])

    assert cameras[1].data.info["pose"] == 2.0
    for camera in cameras:
        assert camera.data.info["pose"] == camera.pose
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [1, 1])
        np.testing.assert_allclose(camera._timestamp_last_update.numpy(), [0.01, 0.01])
    assert batches == [[("wide", 1.0)], [("tele", 2.0)]]
    renderer.render.assert_not_called()


def test_eager_scene_batches_multiple_sensor_families(camera_batch_context):
    """Sensor families batch together unless an earlier update already refreshed their data."""
    _, renderer, camera_batches = camera_batch_context
    sensor_batches = []

    class InheritedSensor(_BatchSensor):
        pass

    class OtherSensor(_BatchSensor):
        @staticmethod
        def _update_buffers_batch_impl(sensors):
            _BatchSensor._update_buffers_batch_impl(sensors)

    class ObserverSensor(_CpuSensor):
        def _update_buffers_impl(self, env_mask):
            np.testing.assert_array_equal(observed.data, [1, 1])
            super()._update_buffers_impl(env_mask)

    observed = _BatchSensor("observed", sensor_batches)
    observer = ObserverSensor("observer")
    first = _BatchSensor("first", sensor_batches)
    second = InheritedSensor("second", sensor_batches)
    other = OtherSensor("other", sensor_batches)
    cameras = [_CpuCamera(renderer, "wide"), _CpuCamera(renderer, "tele")]
    scene = _camera_scene(cameras, observed=observed, observer=observer, first=first, other=other, second=second)

    InteractiveScene.update(scene, 0.25)

    assert camera_batches == [[("wide", 0.0), ("tele", 0.0)]]
    assert sensor_batches == [["first", "second"], ["other"]]
    for sensor in (observed, observer, first, second, other):
        np.testing.assert_array_equal(sensor.data, [1, 1])
        np.testing.assert_allclose(sensor._timestamp.numpy(), [0.25, 0.25])
        np.testing.assert_allclose(sensor._timestamp_last_update.numpy(), [0.25, 0.25])
        assert len(sensor.captures) == 1


def test_sensor_batch_advances_time_once_and_preserves_update_hooks():
    """Batch updates advance sensors in order and preserve visualization refreshes."""
    updates = []

    class DefaultBatchSensor(_CpuSensor):
        @property
        def supports_batch_update(self):
            return True

        def update(self, dt, force_recompute=False):
            updates.append((self.name, dt, force_recompute))
            super().update(dt, force_recompute=force_recompute)

    batched = DefaultBatchSensor("batched")
    visualized = DefaultBatchSensor("visualized")
    visualized._is_visualizing = True
    uninitialized = DefaultBatchSensor("uninitialized")
    uninitialized._is_initialized = False
    sensors = [batched, visualized, uninitialized]

    SensorBase.update_batch([], 0.25)
    SensorBase.update_batch(sensors, 0.25)

    assert updates == [
        ("batched", 0.25, False),
        ("visualized", 0.25, False),
        ("uninitialized", 0.25, False),
    ]
    for sensor in (batched, visualized):
        np.testing.assert_array_equal(sensor.data, [1, 1])
        np.testing.assert_allclose(sensor._timestamp.numpy(), [0.25, 0.25])
        np.testing.assert_allclose(sensor._timestamp_last_update.numpy(), [0.25, 0.25])
        assert len(sensor.captures) == 1
    assert not uninitialized.captures
    np.testing.assert_array_equal(uninitialized._timestamp_last_update.numpy(), [0.0, 0.0])


def test_sensor_batch_rejects_unsupported_input_before_updating():
    """Invalid batches leave all sensors untouched, including preceding supported sensors."""
    batches = []
    sensors = [_BatchSensor("supported", batches), _CpuSensor("unsupported")]

    with pytest.raises(ValueError, match="supports_batch_update"):
        SensorBase.update_batch(sensors, 0.25)

    assert not batches
    for sensor in sensors:
        assert not sensor.captures
        np.testing.assert_array_equal(sensor._timestamp.numpy(), [0.0, 0.0])


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("hook", ["_update_buffers_impl", "_update_outdated_buffers"])
def test_camera_updates_preserve_custom_buffer_hooks(camera_batch_context, monkeypatch, lazy, hook):
    """Custom camera capture hooks run in both eager and lazy scenes."""
    _, renderer, _ = camera_batch_context
    custom_updates = Mock()

    class CustomCamera(_CpuCamera):
        pass

    def custom_update(self, *args, **kwargs):
        custom_updates()
        return getattr(Camera, hook)(self, *args, **kwargs)

    monkeypatch.setattr(CustomCamera, hook, custom_update)
    custom = CustomCamera(renderer, "custom")
    peer = _CpuCamera(renderer, "peer")
    InteractiveScene.update(_camera_scene([custom, peer], lazy=lazy), 0.01)
    assert custom_updates.call_count == (0 if lazy else 1)

    assert custom.data.info["pose"] == 0.0
    renderer.render.assert_not_called()
    if lazy:
        custom_updates.assert_called_once_with()
        renderer.render_batch.assert_called_once_with([custom._render_data])
    else:
        assert renderer.render_batch.call_args_list == [call([custom._render_data]), call([peer._render_data])]


def test_camera_batch_respects_period_partial_reset_and_updated_pose(camera_batch_context):
    """A fresh peer stays cached while due or reset cameras capture their current poses."""
    _, renderer, batches = camera_batch_context
    fast = _CpuCamera(renderer, "fast")
    slow = _CpuCamera(renderer, "slow", update_period=1.0)
    scene = _camera_scene([fast, slow])

    InteractiveScene.update(scene, 0.1)
    InteractiveScene.update(scene, 0.2)
    assert batches == [[("fast", 0.0), ("slow", 0.0)], [("fast", 0.0)]]
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [1, 1])
    np.testing.assert_allclose(slow._timestamp_last_update.numpy(), [0.1, 0.1])

    slow.pose = 3.0
    slow.reset(env_mask=wp.array([False, True], dtype=wp.bool, device="cpu"))
    InteractiveScene.update(scene, 0.1)
    assert batches[-1] == [("fast", 0.0), ("slow", 3.0)]
    assert slow.data.info["pose"] == 3.0
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [1, 1])
    np.testing.assert_allclose(slow._timestamp.numpy(), [0.4, 0.1])
    np.testing.assert_allclose(slow._timestamp_last_update.numpy(), [0.1, 0.1])

    InteractiveScene.update(scene, 0.9)
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [2, 1])
    np.testing.assert_allclose(slow._timestamp_last_update.numpy(), [1.3, 0.1])
    renderer.render.assert_not_called()


@pytest.mark.parametrize("failure", ["render_batch", "read_output"])
def test_failed_eager_capture_can_retry_one_camera(camera_batch_context, failure):
    """Failed captures leave timestamps outdated, and lazy retries do not capture peer cameras."""
    _, renderer, batches = camera_batch_context
    cameras = [_CpuCamera(renderer, "wide"), _CpuCamera(renderer, "tele")]
    scene = _camera_scene(cameras)
    original = getattr(renderer, failure)

    setattr(renderer, failure, Mock(side_effect=RuntimeError("capture failed")))
    with pytest.raises(RuntimeError, match="capture failed"):
        InteractiveScene.update(scene, 0.1)
    for camera in cameras:
        np.testing.assert_array_equal(camera._timestamp_last_update.numpy(), [0.0, 0.0])
    setattr(renderer, failure, original)
    batches.clear()

    assert cameras[0].data.info["pose"] == 0.0
    assert batches == [[("wide", 0.0)]]
    np.testing.assert_allclose(cameras[0]._timestamp_last_update.numpy(), [0.1, 0.1])
    np.testing.assert_array_equal(cameras[1]._timestamp_last_update.numpy(), [0.0, 0.0])
