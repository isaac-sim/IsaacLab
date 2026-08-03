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
from isaaclab.utils.warp import ProxyArray

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

    def __init__(self, transforms: wp.array, velocities: wp.array, coms: wp.array):
        self.transforms = _FakeBuffer(transforms, wp.transformf)
        self.velocities = _FakeBuffer(velocities, wp.spatial_vectorf)
        self.coms = _FakeBuffer(coms, wp.transformf)
        self.get_counts = {"transforms": 0, "velocities": 0, "coms": 0}

    def get_transforms(self):
        self.get_counts["transforms"] += 1
        return self.transforms

    def get_velocities(self):
        self.get_counts["velocities"] += 1
        return self.velocities

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
    coms_torch = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    transforms = wp.from_torch(transforms_torch.contiguous()).view(wp.transformf)
    velocities = wp.from_torch(velocities_torch.contiguous()).view(wp.spatial_vectorf)
    coms = wp.from_torch(coms_torch.contiguous()).view(wp.transformf)
    rigid_view = _FakeRigidView(transforms, velocities, coms)

    sensor_cls = Imu if sensor_type == "imu" else Pva
    sensor = sensor_cls.__new__(sensor_cls)
    sensor.cfg = SimpleNamespace(prim_path=f"/World/{sensor_type.upper()}")
    sensor._device = device
    sensor._num_envs = 2
    sensor._view = rigid_view
    sensor._dt = 0.5
    sensor._timestamp = wp.ones(2, dtype=wp.float32, device=device)
    sensor._offset_pos_b = wp.zeros(2, dtype=wp.vec3f, device=device)
    sensor._offset_quat_b = wp.array(
        [wp.quatf(0.0, 0.0, 0.0, 1.0), wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device
    )
    sensor._prev_lin_vel_w = wp.zeros(2, dtype=wp.vec3f, device=device)
    sensor._coms_buffer = wp.zeros(2, dtype=wp.transformf, device=device)
    sensor._raw_transforms = None
    sensor._raw_velocities = None
    sensor._raw_coms = None
    sensor._update_cmd = None
    sensor._update_env_mask = None
    sensor._update_inv_dt = None
    sensor._use_recorded_launch = use_recorded_launch
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None

    if sensor_type == "imu":
        sensor._data = ImuData()
        sensor._data.create_buffers(num_envs=2, device=device)
        sensor._gravity_bias_w = wp.zeros(2, dtype=wp.vec3f, device=device)
    else:
        sensor._data = PvaData()
        sensor._data.create_buffers(num_envs=2, device=device)
        sensor._prev_ang_vel_w = wp.zeros(2, dtype=wp.vec3f, device=device)
        sensor.GRAVITY_VEC_W = ProxyArray(wp.zeros(2, dtype=wp.vec3f, device=device))

    env_mask = wp.ones(2, dtype=wp.bool, device=device)
    return sensor, rigid_view, velocities_torch, env_mask


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_caches_physx_typed_views(sensor_type):
    """Repeated eager updates should reuse typed views over refreshed PhysX buffers."""
    sensor, rigid_view, _, env_mask = _make_sensor(sensor_type, use_recorded_launch=False)

    sensor._update_buffers_impl(env_mask)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert rigid_view.get_counts == {"transforms": 2, "velocities": 2, "coms": 2}
    assert rigid_view.transforms.view_count == 1
    assert rigid_view.velocities.view_count == 1
    assert rigid_view.coms.view_count == 1


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_records_and_replays_changed_runtime_inputs(sensor_type):
    """Replay should observe refreshed buffers, a new mask, and a changed timestep."""
    sensor, rigid_view, velocities_torch, env_mask = _make_sensor(sensor_type)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    update_cmd = sensor._update_cmd

    assert update_cmd is not None
    torch.testing.assert_close(
        wp.to_torch(sensor._data._lin_acc_b)[:, 0],
        torch.tensor([2.0, 2.0], device=sensor.device),
    )

    velocities_torch[:, 0] = torch.tensor([3.0, 5.0], device=sensor.device)
    sensor._dt = 0.25
    changed_env_mask = wp.array([False, True], dtype=wp.bool, device=sensor.device)
    sensor._update_buffers_impl(changed_env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._update_cmd is update_cmd
    # replays must still call the PhysX getters: they are what refresh the underlying buffers
    assert rigid_view.get_counts == {"transforms": 2, "velocities": 2, "coms": 2}
    torch.testing.assert_close(
        wp.to_torch(sensor._data._lin_acc_b)[:, 0],
        torch.tensor([2.0, 16.0], device=sensor.device),
    )


@pytest.mark.parametrize("sensor_type", ["imu", "pva"])
def test_sensor_falls_back_when_recording_fails(monkeypatch, sensor_type):
    """A recording failure should disable recording and execute the update eagerly."""
    sensor, _, _, env_mask = _make_sensor(sensor_type)
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
    sensor, _, _, _ = _make_sensor(sensor_type)
    sensor._raw_transforms = object()
    sensor._raw_velocities = object()
    sensor._raw_coms = object()
    sensor._update_cmd = object()
    sensor._update_env_mask = object()
    sensor._update_inv_dt = 1.0
    base_cls = BaseImu if sensor_type == "imu" else BasePva
    monkeypatch.setattr(base_cls, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._view is None
    assert sensor._raw_transforms is None
    assert sensor._raw_velocities is None
    assert sensor._raw_coms is None
    assert sensor._update_cmd is None
    assert sensor._update_env_mask is None
    assert sensor._update_inv_dt is None
