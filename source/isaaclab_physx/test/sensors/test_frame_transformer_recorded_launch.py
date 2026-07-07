# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cached PhysX views and recorded FrameTransformer updates."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab.sensors.frame_transformer import BaseFrameTransformer
from isaaclab_physx.sensors.frame_transformer import frame_transformer as frame_transformer_module
from isaaclab_physx.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab_physx.sensors.frame_transformer.frame_transformer_data import FrameTransformerData


class _FakeTransformView:
    """Return one stable PhysX-like transform buffer while counting typed-view construction."""

    def __init__(self, transforms: wp.array):
        self.transforms = transforms
        self.get_count = 0
        self.view_count = 0

    def get_transforms(self):
        self.get_count += 1
        return self

    def view(self, dtype):
        assert dtype == wp.transformf
        self.view_count += 1
        return self.transforms


def _make_frame_transformer(use_recorded_launch: bool = True):
    """Create a one-environment FrameTransformer without a USD scene."""
    device = "cuda:0"
    transforms_torch = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    transforms = wp.from_torch(transforms_torch.contiguous()).view(wp.transformf)
    transform_view = _FakeTransformView(transforms)

    sensor = FrameTransformer.__new__(FrameTransformer)
    sensor.cfg = SimpleNamespace(prim_path="/World/Source")
    sensor._device = device
    sensor._num_envs = 1
    sensor._num_target_frames = 1
    sensor._frame_physx_view = transform_view
    sensor._source_raw_indices = wp.array([0], dtype=wp.int32, device=device)
    sensor._target_raw_indices = wp.array([[1]], dtype=wp.int32, device=device)
    sensor._source_offset_pos_wp = wp.zeros(1, dtype=wp.vec3f, device=device)
    sensor._source_offset_quat_wp = wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
    sensor._target_offset_pos_wp = wp.zeros(1, dtype=wp.vec3f, device=device)
    sensor._target_offset_quat_wp = wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
    sensor._data = FrameTransformerData()
    sensor._data.create_buffers(num_envs=1, num_target_frames=1, target_frame_names=["target"], device=device)
    sensor._raw_transforms = None
    sensor._update_cmd = None
    sensor._use_recorded_launch = use_recorded_launch
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None

    env_mask = wp.ones(1, dtype=wp.bool, device=device)
    return sensor, transform_view, transforms_torch, env_mask


def test_frame_transformer_caches_physx_transform_view():
    """Repeated eager updates should reuse one typed view over the refreshed PhysX buffer."""
    sensor, transform_view, _, env_mask = _make_frame_transformer(use_recorded_launch=False)

    sensor._update_buffers_impl(env_mask)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert transform_view.get_count == 2
    assert transform_view.view_count == 1


def test_frame_transformer_records_and_replays_update_launch():
    """The first recorded update should execute, and later updates should replay refreshed data."""
    sensor, _, transforms_torch, env_mask = _make_frame_transformer()

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    update_cmd = sensor._update_cmd

    assert update_cmd is not None
    torch.testing.assert_close(
        wp.to_torch(sensor._data._target_pos_source)[0, 0],
        torch.tensor([1.0, 0.0, 0.0], device=sensor.device),
    )

    transforms_torch[1, 0] = 2.0
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._update_cmd is update_cmd
    torch.testing.assert_close(
        wp.to_torch(sensor._data._target_pos_source)[0, 0],
        torch.tensor([2.0, 0.0, 0.0], device=sensor.device),
    )


def test_frame_transformer_falls_back_when_recording_fails(monkeypatch):
    """A recording failure should disable recording and execute the current update eagerly."""
    sensor, _, _, env_mask = _make_frame_transformer()
    original_launch = frame_transformer_module.wp.launch

    def launch_with_recording_failure(*args, record_cmd=False, **kwargs):
        if record_cmd:
            raise RuntimeError("recording failed")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(frame_transformer_module.wp, "launch", launch_with_recording_failure)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert not sensor._use_recorded_launch
    torch.testing.assert_close(
        wp.to_torch(sensor._data._target_pos_source)[0, 0],
        torch.tensor([1.0, 0.0, 0.0], device=sensor.device),
    )


def test_frame_transformer_invalidation_drops_cached_launch_state(monkeypatch):
    """Physics invalidation should release the cached PhysX view and recorded command."""
    sensor, _, _, _ = _make_frame_transformer()
    sensor._raw_transforms = object()
    sensor._update_cmd = object()
    monkeypatch.setattr(BaseFrameTransformer, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._frame_physx_view is None
    assert sensor._raw_transforms is None
    assert sensor._update_cmd is None
