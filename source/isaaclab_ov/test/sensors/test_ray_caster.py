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
from isaaclab_ov.sensors.ray_caster import ray_caster as ray_caster_module
from isaaclab_ov.sensors.ray_caster.ray_caster import RayCaster

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.utils.warp import CapturedKernelUpdate

from .conftest import (
    CountingReadView,
    assert_invalidation_drops_captured_graph,
    assert_update_refused_inside_outer_capture,
    make_identity_quat_poses,
    requires_cuda,
)


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

    def _initialize_static_pose_tracking(self, prims):
        raise AssertionError("dynamic clone-plan sources should not fall back to static USD pose tracking")


def test_initialize_pose_tracking_uses_shared_rigid_body_resolver_without_destination_usd(monkeypatch):
    """RayCaster should use SensorBase clone-plan resolution when destination USD prims are missing."""
    fake_tensor_type = object()
    fake_tensor_types = SimpleNamespace(RIGID_BODY_POSE=fake_tensor_type)
    fake_physx = _FakePhysx()

    monkeypatch.setitem(sys.modules, "isaaclab_ov.tensor_types", fake_tensor_types)
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


# ---------------------------------------------------------------------------
# CUDA-graph capture (scene-free unit tests)
# ---------------------------------------------------------------------------


@wp.kernel
def _copy_translation_under_mask(
    transforms: wp.array(dtype=wp.transformf),
    mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.vec3f),
):
    """Copy the tracked translation into ``out`` for masked-in envs only."""
    env_id = wp.tid()
    if mask[env_id]:
        out[env_id] = wp.transform_get_translation(transforms[env_id])


class _FakeBodyView(CountingReadView):
    """Live ovphysx-style pose binding: fill the caller's staging buffer, counting reads."""

    def __init__(self, poses_torch: torch.Tensor):
        super().__init__(poses_torch)
        self.shape = (poses_torch.shape[0], 7)

    def destroy(self) -> None:
        """Match the native binding teardown hook; nothing to release for the fake."""


def _make_graphed_ray_caster(num_envs: int = 2):
    """Create a RayCaster without a USD scene, wired for graph unit tests.

    Uses a *live* fake body view (non-None) so the prefetch/read path is real
    and the read count is observable.
    """
    device = "cuda:0"
    translations = torch.arange(1, num_envs * 3 + 1, dtype=torch.float32, device=device).reshape(num_envs, 3)
    body_view = _FakeBodyView(make_identity_quat_poses(translations))

    sensor = RayCaster.__new__(RayCaster)
    sensor.cfg = SimpleNamespace(prim_path="/World/RC")
    sensor._device = device
    sensor._num_envs = num_envs
    sensor._ovphysx_body_view = body_view
    sensor._view_count = num_envs
    sensor._pose_buf = wp.zeros((num_envs, 7), dtype=wp.float32, device=device)
    sensor._pose_buf_transformf = wp.array(
        ptr=sensor._pose_buf.ptr,
        shape=(num_envs,),
        dtype=wp.transformf,
        device=str(sensor._pose_buf.device),
        copy=False,
    )
    sensor._transforms_prefetched = False
    sensor._test_out = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
    sensor._update_graph = CapturedKernelUpdate(device, owner="ray caster at '/World/RC'")

    env_mask = wp.ones(num_envs, dtype=wp.bool, device=device)
    return sensor, body_view, env_mask


def _test_update_impl(self, env_mask):
    """Stand-in for the core kernel-only ``BaseRayCaster._update_buffers_impl``.

    Consumes the tracked transforms through a single deterministic warp kernel,
    so the captured graph exercises the same prefetched-buffer contract.
    """
    transforms = self._get_view_transforms_wp()
    wp.launch(
        _copy_translation_under_mask,
        dim=self._num_envs,
        inputs=[transforms, env_mask, self._test_out],
        device=self._device,
    )


@requires_cuda
def test_ray_caster_graph_replay_sees_refreshed_reads_and_mask(monkeypatch):
    """Replays must consume freshly read body poses and in-place mask changes."""
    monkeypatch.setattr(BaseRayCaster, "_update_buffers_impl", _test_update_impl)
    sensor, body_view, env_mask = _make_graphed_ray_caster(num_envs=2)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor._device)
    assert sensor._update_graph.is_captured
    assert body_view.read_count == 1
    out_before = wp.to_torch(sensor._test_out).clone()

    body_view.source_torch[:, :3] *= 10.0
    wp.to_torch(env_mask)[:] = torch.tensor([False, True], device=sensor._device)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor._device)

    assert body_view.read_count == 2  # fetch still runs eagerly on replay
    out_after = wp.to_torch(sensor._test_out)
    torch.testing.assert_close(out_after[0], out_before[0])  # masked-off env untouched
    assert not torch.allclose(out_after[1], out_before[1])  # masked-on env refreshed


@requires_cuda
def test_ray_caster_refuses_update_inside_outer_capture(monkeypatch):
    """The update must raise before reading OvPhysX when an outer capture is active."""
    monkeypatch.setattr(BaseRayCaster, "_update_buffers_impl", _test_update_impl)
    sensor, body_view, env_mask = _make_graphed_ray_caster()
    assert_update_refused_inside_outer_capture(sensor, lambda: sensor._update_buffers_impl(env_mask), body_view)


@requires_cuda
def test_ray_caster_invalidation_drops_captured_graph(monkeypatch):
    """Invalidation must invalidate the update graph alongside the native handles."""
    monkeypatch.setattr(BaseRayCaster, "_update_buffers_impl", _test_update_impl)
    sensor, _, env_mask = _make_graphed_ray_caster()
    assert_invalidation_drops_captured_graph(
        sensor, lambda: sensor._update_buffers_impl(env_mask), BaseRayCaster, monkeypatch
    )
