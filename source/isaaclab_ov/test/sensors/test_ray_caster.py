# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVPhysX RayCaster backend glue."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.sensors.ray_caster import ray_caster as ray_caster_module


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
        self.cfg = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Robot/base/ray")
        self._device = "cpu"
        self._resolved = (
            "/World/envs/env_[^/]+/Robot/base",
            (0.1, 0.2, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _resolve_rigid_body_ancestor_expr(self):
        return self._resolved


def test_initialize_pose_tracking_binds_body_glob_and_replicates_offset(monkeypatch):
    """The resolved rigid-body expression is bound as a glob, and its fixed offset is replicated per frame."""
    fake_physx = _FakePhysx()

    monkeypatch.setattr(ray_caster_module.OvPhysxManager, "get_physx_instance", staticmethod(lambda: fake_physx))

    sensor = _DummyRayCaster()

    sensor._initialize_pose_tracking()

    assert fake_physx.calls == [("/World/envs/env_*/Robot/base", TT.RIGID_BODY_POSE)]
    assert sensor.count == 3
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_pos_wp),
        torch.tensor([[0.1, 0.2, 0.3]] * 3, dtype=torch.float32),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_quat_wp),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
    )
