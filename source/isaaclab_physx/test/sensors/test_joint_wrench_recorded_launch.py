# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cached PhysX views and recorded JointWrench sensor updates."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_physx.sensors.joint_wrench import joint_wrench_sensor as joint_wrench_module
from isaaclab_physx.sensors.joint_wrench.joint_wrench_sensor import JointWrenchSensor
from isaaclab_physx.sensors.joint_wrench.joint_wrench_sensor_data import JointWrenchSensorData

from isaaclab.sensors.joint_wrench import BaseJointWrenchSensor


class _FakeArticulationView:
    """Return one stable PhysX-like wrench buffer while counting typed-view construction."""

    def __init__(self, wrenches: wp.array):
        self.wrenches = wrenches
        self.get_count = 0
        self.view_count = 0

    def get_link_incoming_joint_force(self):
        self.get_count += 1
        return self

    def view(self, dtype):
        assert dtype == wp.spatial_vectorf
        self.view_count += 1
        return self.wrenches


def _make_joint_wrench_sensor(use_recorded_launch: bool = True):
    """Create a one-environment JointWrench sensor without a USD scene."""
    device = "cuda:0"
    wrenches_torch = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]],
        dtype=torch.float32,
        device=device,
    )
    wrenches = wp.from_torch(wrenches_torch.contiguous()).view(wp.spatial_vectorf)
    root_view = _FakeArticulationView(wrenches)

    sensor = JointWrenchSensor.__new__(JointWrenchSensor)
    sensor.cfg = SimpleNamespace(prim_path="/World/Robot")
    sensor._device = device
    sensor._num_envs = 1
    sensor._num_bodies = 1
    sensor._root_view = root_view
    sensor._joint_pos_b = wp.zeros(1, dtype=wp.vec3f, device=device)
    sensor._joint_quat_b = wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
    sensor._timestamp = wp.ones(1, dtype=wp.float32, device=device)
    sensor._data = JointWrenchSensorData()
    sensor._data.create_buffers(num_envs=1, num_bodies=1, device=device)
    sensor._raw_incoming_joint_wrench = None
    sensor._update_cmd = None
    sensor._use_recorded_launch = use_recorded_launch
    sensor._physics_sim_view = None
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None

    env_mask = wp.ones(1, dtype=wp.bool, device=device)
    return sensor, root_view, wrenches_torch, env_mask


def test_joint_wrench_caches_physx_wrench_view():
    """Repeated eager updates should reuse one typed view over the refreshed PhysX buffer."""
    sensor, root_view, _, env_mask = _make_joint_wrench_sensor(use_recorded_launch=False)

    sensor._update_buffers_impl(env_mask)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert root_view.get_count == 2
    assert root_view.view_count == 1


def test_joint_wrench_records_and_replays_update_launch():
    """The first recorded update should execute, and later updates should replay refreshed data."""
    sensor, _, wrenches_torch, env_mask = _make_joint_wrench_sensor()

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    update_cmd = sensor._update_cmd

    assert update_cmd is not None
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[0, 0],
        torch.tensor([1.0, 2.0, 3.0], device=sensor.device),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._data._torque)[0, 0],
        torch.tensor([4.0, 5.0, 6.0], device=sensor.device),
    )

    wrenches_torch[0, 0, 0] = 7.0
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._update_cmd is update_cmd
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[0, 0],
        torch.tensor([7.0, 2.0, 3.0], device=sensor.device),
    )


def test_joint_wrench_falls_back_when_recording_fails(monkeypatch):
    """A recording failure should disable recording and execute the current update eagerly."""
    sensor, _, _, env_mask = _make_joint_wrench_sensor()
    original_launch = joint_wrench_module.wp.launch

    def launch_with_recording_failure(*args, record_cmd=False, **kwargs):
        if record_cmd:
            raise RuntimeError("recording failed")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(joint_wrench_module.wp, "launch", launch_with_recording_failure)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert not sensor._use_recorded_launch
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[0, 0],
        torch.tensor([1.0, 2.0, 3.0], device=sensor.device),
    )


def test_joint_wrench_invalidation_drops_cached_launch_state(monkeypatch):
    """Physics invalidation should release the cached PhysX view and recorded command."""
    sensor, _, _, _ = _make_joint_wrench_sensor()
    sensor._raw_incoming_joint_wrench = object()
    sensor._update_cmd = object()
    monkeypatch.setattr(BaseJointWrenchSensor, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._root_view is None
    assert sensor._raw_incoming_joint_wrench is None
    assert sensor._update_cmd is None
