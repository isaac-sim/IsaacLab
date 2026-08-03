# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVPhysX RayCaster backend glue."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_ovphysx.sensors.ray_caster import ray_caster as ray_caster_module


class _FakeBinding:
    shape = (3, 7)

    def read(self, dst):
        pass

    def destroy(self):
        pass


class _FakePhysx:
    def __init__(self):
        self.calls = []

    def create_tensor_binding(self, *, pattern, tensor_type):
        self.calls.append((pattern, tensor_type))
        return _FakeBinding()


class _DummyRayCaster(ray_caster_module._OvPhysxRayCasterMixin):
    def __init__(self):
        self.cfg = SimpleNamespace(prim_path="/World/envs/env_.*/Robot/base/ray")
        self._device = "cpu"
        self._resolved = (
            "/World/envs/env_.*/Robot/base",
            (0.1, 0.2, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _resolve_rigid_body_ancestor_expr(self):
        return self._resolved

    def _initialize_static_pose_tracking(self, prims):
        raise AssertionError("dynamic clone-plan sources should not fall back to static USD pose tracking")


def test_initialize_pose_tracking_uses_shared_rigid_body_resolver_without_destination_usd(monkeypatch):
    """RayCaster should use SensorBase clone-plan resolution when destination USD prims are missing."""
    fake_tensor_type = object()
    fake_tensor_types = SimpleNamespace(RIGID_BODY_POSE=fake_tensor_type)
    fake_physx = _FakePhysx()

    monkeypatch.setitem(sys.modules, "isaaclab_ovphysx.tensor_types", fake_tensor_types)
    monkeypatch.setattr(ray_caster_module.sim_utils, "find_matching_prims", lambda _path: [])
    monkeypatch.setattr(ray_caster_module.OvPhysxManager, "get_physx_instance", staticmethod(lambda: fake_physx))

    sensor = _DummyRayCaster()

    sensor._initialize_pose_tracking()

    assert fake_physx.calls == [("/World/envs/env_*/Robot/base", fake_tensor_type)]
    assert sensor.count == 3
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_pos_wp),
        torch.tensor([[0.1, 0.2, 0.3]] * 3, dtype=torch.float32),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_quat_wp),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
    )
