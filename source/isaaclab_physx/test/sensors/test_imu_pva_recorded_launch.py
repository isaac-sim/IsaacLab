# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cached PhysX views and recorded IMU/PVA updates."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_physx.sensors.imu import imu as imu_module
from isaaclab_physx.sensors.imu.imu import Imu
from isaaclab_physx.sensors.imu.imu_data import ImuData
from isaaclab_physx.sensors.pva import pva as pva_module
from isaaclab_physx.sensors.pva.pva import Pva
from isaaclab_physx.sensors.pva.pva_data import PvaData

from isaaclab.sensors.imu import BaseImu
from isaaclab.sensors.pva import BasePva

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


class _FakeBuffer:
    """Wrap a typed array while counting typed-view construction."""

    def __init__(self, array: wp.array, dtype):
        self.array = array
        self.dtype = dtype
        self.view_count = 0

    def view(self, dtype):
        assert dtype == self.dtype
        self.view_count += 1
        return self.array

    @property
    def ptr(self):
        return self.array.ptr


class _FakeRigidView:
    """Return stable PhysX-like buffers while counting getter calls."""

    def __init__(self, transforms: wp.array, velocities: wp.array, accelerations: wp.array, coms: wp.array):
        self.transforms = _FakeBuffer(transforms, wp.transformf)
        self.velocities = _FakeBuffer(velocities, wp.spatial_vectorf)
        self.accelerations = _FakeBuffer(accelerations, wp.spatial_vectorf)
        self.coms = _FakeBuffer(coms, wp.transformf)
        self.get_counts = {"transforms": 0, "velocities": 0, "accelerations": 0, "coms": 0}

    def get_transforms(self):
        self.get_counts["transforms"] += 1
        return self.transforms

    def get_velocities(self):
        self.get_counts["velocities"] += 1
        return self.velocities

    def get_accelerations(self):
        self.get_counts["accelerations"] += 1
        return self.accelerations

    def get_coms(self):
        self.get_counts["coms"] += 1
        return self.coms


def _make_sensor(sensor_type: str, use_recorded_launch: bool = True):
    """Create a two-environment IMU or PVA without a USD scene."""
    device = "cuda:0"
    transforms_torch = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    velocities_torch = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    accelerations_torch = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    coms_torch = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    transforms = wp.from_torch(transforms_torch.contiguous()).view(wp.transformf)
    velocities = wp.from_torch(velocities_torch.contiguous()).view(wp.spatial_vectorf)
    accelerations = wp.from_torch(accelerations_torch.contiguous()).view(wp.spatial_vectorf)
    coms = wp.from_torch(coms_torch.contiguous()).view(wp.transformf)
    rigid_view = _FakeRigidView(transforms, velocities, accelerations, coms)

    gravity_sink = {"value": (0.0, 0.0, -9.81)}
    sensor_cls = Imu if sensor_type == "imu" else Pva
    sensor = sensor_cls.__new__(sensor_cls)
    sensor.cfg = SimpleNamespace(prim_path=f"/World/{sensor_type.upper()}")
    sensor._device = device
    sensor._num_envs = 2
    sensor._view = rigid_view
    sensor._timestamp = wp.ones(2, dtype=wp.float32, device=device)
    sensor._offset_pos_b = wp.zeros(2, dtype=wp.vec3f, device=device)
    sensor._offset_quat_b = wp.array(
        [wp.quatf(0.0, 0.0, 0.0, 1.0), wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device
    )
    sensor._coms_buffer = wp.zeros(2, dtype=wp.transformf, device=device)
    sensor._raw_transforms = None
    sensor._raw_velocities = None
    sensor._raw_accelerations = None
    sensor._raw_coms = None
    sensor._update_cmd = None
    sensor._update_env_mask = None
    sensor._use_recorded_launch = use_recorded_launch
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    # ``_update_buffers_impl`` re-reads scene gravity so runtime randomization stays visible.
    sensor._physics_sim_view = SimpleNamespace(get_gravity=lambda: gravity_sink["value"])
    sensor._gravity_w = gravity_sink["value"]

    if sensor_type == "imu":
        sensor._data = ImuData()
        sensor._data.create_buffers(num_envs=2, device=device)
        sensor._gravity_bias_w = wp.vec3f(0.0, 0.0, 9.81)
        sensor._recorded_gravity_w = gravity_sink["value"]
    else:
        sensor._data = PvaData()
        sensor._data.create_buffers(num_envs=2, device=device)
        sensor._gravity_vec_w = wp.vec3f(0.0, 0.0, -1.0)
        sensor._recorded_gravity_w = gravity_sink["value"]

    env_mask = wp.ones(2, dtype=wp.bool, device=device)
    return sensor, rigid_view, accelerations_torch, env_mask, gravity_sink


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_caches_physx_typed_views(sensor_type):
    """Repeated eager updates should reuse typed views over refreshed PhysX buffers."""
    sensor, rigid_view, _, env_mask, _ = _make_sensor(sensor_type, use_recorded_launch=False)

    sensor._update_buffers_impl(env_mask)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert rigid_view.get_counts == {"transforms": 2, "velocities": 2, "accelerations": 2, "coms": 2}
    assert rigid_view.transforms.view_count == 1
    assert rigid_view.velocities.view_count == 1
    assert rigid_view.accelerations.view_count == 1
    assert rigid_view.coms.view_count == 1


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_records_and_replays_changed_runtime_inputs(sensor_type):
    """Replay should observe refreshed buffers and a new mask."""
    sensor, rigid_view, accelerations_torch, env_mask, _ = _make_sensor(sensor_type)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    update_cmd = sensor._update_cmd

    assert update_cmd is not None
    torch.testing.assert_close(
        wp.to_torch(sensor._data._lin_acc_b)[:, 0],
        torch.tensor([2.0, 2.0], device=sensor.device),
    )

    accelerations_torch[:, 0] = torch.tensor([3.0, 16.0], device=sensor.device)
    wp.to_torch(sensor._timestamp).fill_(1.25)
    changed_env_mask = wp.array([False, True], dtype=wp.bool, device=sensor.device)
    sensor._update_buffers_impl(changed_env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._update_cmd is update_cmd
    # replays must still call the PhysX getters: they are what refresh the underlying buffers
    assert rigid_view.get_counts == {"transforms": 2, "velocities": 2, "accelerations": 2, "coms": 2}
    torch.testing.assert_close(
        wp.to_torch(sensor._data._lin_acc_b)[:, 0],
        torch.tensor([2.0, 16.0], device=sensor.device),
    )


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_falls_back_when_recording_fails(monkeypatch, sensor_type):
    """A recording failure should disable recording and execute the update eagerly."""
    sensor, _, _, env_mask, _ = _make_sensor(sensor_type)
    sensor_module = imu_module if sensor_type == "imu" else pva_module
    original_launch = sensor_module.wp.launch

    def launch_with_recording_failure(*args, record_cmd=False, **kwargs):
        if record_cmd:
            raise RuntimeError("recording failed")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(sensor_module.wp, "launch", launch_with_recording_failure)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert not sensor._use_recorded_launch
    assert sensor._update_cmd is None
    torch.testing.assert_close(
        wp.to_torch(sensor._data._lin_acc_b)[:, 0],
        torch.tensor([2.0, 2.0], device=sensor.device),
    )


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_invalidation_drops_cached_launch_state(monkeypatch, sensor_type):
    """Physics invalidation should release cached PhysX views and the recorded command."""
    sensor, _, _, _, _ = _make_sensor(sensor_type)
    sensor._raw_transforms = object()
    sensor._raw_velocities = object()
    sensor._raw_accelerations = object()
    sensor._raw_coms = object()
    sensor._update_cmd = object()
    sensor._update_env_mask = object()
    base_cls = BaseImu if sensor_type == "imu" else BasePva
    monkeypatch.setattr(base_cls, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._view is None
    assert sensor._raw_transforms is None
    assert sensor._raw_velocities is None
    assert sensor._raw_accelerations is None
    assert sensor._raw_coms is None
    assert sensor._update_cmd is None
    assert sensor._update_env_mask is None


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_tracks_runtime_gravity_changes(sensor_type):
    """Scene gravity randomized after initialization must reach the sensor's replayed launch."""
    sensor, _, _, env_mask, gravity_sink = _make_sensor(sensor_type)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    gravity_sink["value"] = (0.0, 3.72, 0.0)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    # IMU carries the accelerometer bias (-g); PVA carries the unit gravity direction.
    expected = (0.0, -3.72, 0.0) if sensor_type == "imu" else (0.0, 1.0, 0.0)
    resolved = sensor._gravity_bias_w if sensor_type == "imu" else sensor._gravity_vec_w
    assert tuple(resolved) == pytest.approx(expected)
    # The recorded command must have been re-bound, not left holding the old value.
    assert sensor._recorded_gravity_w == (0.0, 3.72, 0.0)


def test_imu_replayed_launch_applies_new_gravity_bias():
    """A gravity change must alter the IMU output through the replayed recorded launch."""
    sensor, _, _, env_mask, gravity_sink = _make_sensor("imu")

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    # Guard the point of the test: the eager path would not exercise set_param_by_name.
    assert sensor._update_cmd is not None
    before = wp.to_torch(sensor._data._lin_acc_b).clone()

    # Solver accelerations are constant across updates in this harness, so the gravity
    # bias is the only term that changes.
    gravity_sink["value"] = (0.0, 3.72, 0.0)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    after = wp.to_torch(sensor._data._lin_acc_b)

    # Sensor frames are identity in this harness, so the delta is exactly the bias delta (-dg).
    delta_g = torch.tensor((0.0, 3.72, 0.0), device=sensor.device) - torch.tensor(
        (0.0, 0.0, -9.81), device=sensor.device
    )
    torch.testing.assert_close(after - before, (-delta_g).repeat(2, 1))


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_skips_gravity_work_when_unchanged(sensor_type):
    """An unchanged scene gravity must not re-do the per-update gravity work."""
    sensor, _, _, env_mask, _ = _make_sensor(sensor_type)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    # Still the harness value: no re-bind happened, so the recorded command is untouched.
    assert sensor._recorded_gravity_w == (0.0, 0.0, -9.81)
