# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cached PhysX transforms and graphed standard RayCaster updates."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_physx.sensors.ray_caster import ray_caster as ray_caster_module
from isaaclab_physx.sensors.ray_caster.ray_caster import RayCaster

from isaaclab.sensors.ray_caster import BaseRayCaster

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


@wp.kernel
def _copy_transform_position(
    transforms: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    output: wp.array(dtype=wp.vec3f),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        output[env_id] = wp.transform_get_translation(transforms[env_id])


class _FakeTransformView:
    """Refresh one stable PhysX-like transform buffer and count conversions."""

    def __init__(self, transforms: torch.Tensor):
        self.transforms = transforms
        self.get_count = 0
        self.conversion_count = 0

    def get_transforms(self):
        self.get_count += 1
        return self

    def contiguous(self):
        self.conversion_count += 1
        return self.transforms


def _make_ray_caster(use_graph: bool = True, cache_raw_transforms: bool = True):
    """Create a two-environment standard RayCaster without a USD scene."""
    device = "cuda:0"
    transforms_torch = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    ).contiguous()
    transforms = wp.from_torch(transforms_torch).view(wp.transformf)
    transform_view = _FakeTransformView(transforms_torch)

    sensor = RayCaster.__new__(RayCaster)
    sensor.cfg = SimpleNamespace(prim_path="/World/Sensor")
    sensor._device = device
    sensor._physx_body_view = transform_view
    sensor._raw_transforms = transforms if cache_raw_transforms else None
    sensor._compute_graph = None
    sensor._use_graph = use_graph
    sensor._env_mask = None
    sensor._transforms_prefetched = False
    sensor._test_output = wp.zeros(2, dtype=wp.vec3f, device=device)
    sensor._view = object()
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None

    wp.load_module(device=device)
    env_mask = wp.array([True, False], dtype=wp.bool, device=device)
    return sensor, transform_view, transforms_torch, env_mask


def _replace_base_update_with_test_kernel(monkeypatch) -> None:
    """Replace the four production kernels with one deterministic test kernel."""

    def compute(sensor, env_mask):
        wp.launch(
            _copy_transform_position,
            dim=2,
            inputs=[sensor._raw_transforms, env_mask, sensor._test_output],
            device=sensor._device,
        )

    monkeypatch.setattr(BaseRayCaster, "_update_buffers_impl", compute)


def test_ray_caster_caches_physx_transform_view():
    """Repeated eager reads should reuse one typed view over the refreshed PhysX buffer."""
    sensor, transform_view, _, _ = _make_ray_caster(use_graph=False, cache_raw_transforms=False)

    sensor._get_view_transforms_wp()
    sensor._get_view_transforms_wp()

    assert transform_view.get_count == 2
    assert transform_view.conversion_count == 1


def test_ray_caster_updates_eagerly_when_graphs_are_disabled(monkeypatch):
    """Graph-disabled updates should stay eager while reusing the cached transform view."""
    _replace_base_update_with_test_kernel(monkeypatch)
    sensor, transform_view, transforms_torch, env_mask = _make_ray_caster(use_graph=False, cache_raw_transforms=False)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._compute_graph is None
    torch.testing.assert_close(
        wp.to_torch(sensor._test_output),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sensor.device),
    )

    transforms_torch[1, 0] = 4.0
    wp.to_torch(env_mask)[:] = torch.tensor([False, True], device=sensor.device)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._compute_graph is None
    assert transform_view.get_count == 2
    assert transform_view.conversion_count == 1
    torch.testing.assert_close(
        wp.to_torch(sensor._test_output),
        torch.tensor([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], device=sensor.device),
    )


def test_ray_caster_refuses_update_inside_outer_graph_capture(monkeypatch):
    """Updating the sensor during an outer capture should raise before touching PhysX."""
    _replace_base_update_with_test_kernel(monkeypatch)
    sensor, transform_view, _, env_mask = _make_ray_caster(cache_raw_transforms=False)
    device = wp.get_device(sensor.device)

    zeros = wp.zeros_like(sensor._test_output)
    with wp.ScopedCapture(device=device):
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            sensor._update_buffers_impl(env_mask)
        # record something so the outer capture does not end empty
        wp.copy(sensor._test_output, zeros)

    assert sensor._compute_graph is None
    assert transform_view.get_count == 0


def test_ray_caster_captures_and_replays_update_graph(monkeypatch):
    """Graph replay should consume changed transforms and contents of the stable environment mask."""
    _replace_base_update_with_test_kernel(monkeypatch)
    sensor, _, transforms_torch, env_mask = _make_ray_caster()

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    compute_graph = sensor._compute_graph

    assert compute_graph is not None
    torch.testing.assert_close(
        wp.to_torch(sensor._test_output),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sensor.device),
    )

    transforms_torch[1, 0] = 3.0
    env_mask_torch = wp.to_torch(env_mask)
    env_mask_torch[:] = torch.tensor([False, True], device=sensor.device)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._compute_graph is compute_graph
    torch.testing.assert_close(
        wp.to_torch(sensor._test_output),
        torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device=sensor.device),
    )


def test_ray_caster_falls_back_when_graph_capture_fails(monkeypatch):
    """A graph-capture failure should disable graphs and execute the current update eagerly."""
    _replace_base_update_with_test_kernel(monkeypatch)
    sensor, _, _, env_mask = _make_ray_caster()

    class FailingCapture:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("capture failed")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(ray_caster_module.wp, "ScopedCapture", FailingCapture)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert not sensor._use_graph
    torch.testing.assert_close(
        wp.to_torch(sensor._test_output)[0],
        torch.tensor([1.0, 0.0, 0.0], device=sensor.device),
    )


def test_ray_caster_invalidation_drops_cached_graph_state(monkeypatch):
    """Physics invalidation should release the cached PhysX view and CUDA graph."""
    sensor, _, _, _ = _make_ray_caster()
    sensor._compute_graph = object()
    sensor._env_mask = object()
    sensor._transforms_prefetched = True
    monkeypatch.setattr(BaseRayCaster, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._view is None
    assert sensor._physx_body_view is None
    assert sensor._raw_transforms is None
    assert sensor._compute_graph is None
    assert sensor._env_mask is None
    assert not sensor._transforms_prefetched
