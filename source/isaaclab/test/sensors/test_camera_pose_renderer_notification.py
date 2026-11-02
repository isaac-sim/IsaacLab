# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose refresh for renderers presented outside :meth:`Camera.update`."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.sensors.camera import Camera
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit

_NUM_CAMERAS = 2


class _RecordingRenderer:
    """Record the camera poses each renderer notification carries."""

    def __init__(self):
        self.poses = []

    def update_camera(self, render_data, positions, orientations, intrinsics):
        self.poses.append((render_data, positions.torch.clone(), orientations.torch.clone()))

    def cleanup(self, render_data):
        pass


def _assert_same_rotations(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare unit quaternions up to sign, since ``q`` and ``-q`` describe one rotation."""
    torch.testing.assert_close((actual * expected).sum(dim=-1).abs(), torch.ones(actual.shape[0]))


class _CpuCamera(Camera):
    """A CPU camera whose pose source is an in-memory world pose."""

    def __init__(self):
        self.cfg = SimpleNamespace(update_period=0.0, update_latest_camera_pose=False)
        self._device = "cpu"
        # No capture is pending, so reading ``data`` returns the buffers without rendering.
        self._data_dirty = False
        self._sensor_prims = [object()] * _NUM_CAMERAS
        self._renderer = _RecordingRenderer()
        self._render_data = object()
        self._data = CameraData()
        self._data.create_buffers(_NUM_CAMERAS, "cpu")
        self._frame = ProxyArray(wp.zeros(_NUM_CAMERAS, dtype=wp.int64, device="cpu"))
        self._ALL_INDICES = wp.array(range(_NUM_CAMERAS), dtype=wp.int32, device="cpu")
        self._ALL_ENV_MASK = wp.ones(_NUM_CAMERAS, dtype=wp.bool, device="cpu")
        self.positions = torch.zeros(_NUM_CAMERAS, 3)
        self.orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * _NUM_CAMERAS)
        self._view = SimpleNamespace(count=_NUM_CAMERAS, get_world_poses=self._world_poses, close=lambda: None)

    def __del__(self):
        pass

    def _world_poses(self, env_ids):
        indices = slice(None) if env_ids is None else env_ids.numpy()
        positions = wp.from_torch(self.positions[indices].contiguous(), dtype=wp.vec3f)
        orientations = wp.from_torch(self.orientations[indices].contiguous(), dtype=wp.quatf)
        return ProxyArray(positions), ProxyArray(orientations)


def test_refresh_world_poses_updates_data_and_renderer_without_capturing():
    """A refresh exposes moved poses to the data and the renderer, but captures no frame."""
    camera = _CpuCamera()
    camera.positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    camera.orientations = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    camera.refresh_world_poses()

    # The data and renderer carry the world-convention orientation that a direct pose query reports.
    _, world_orientations = camera.get_world_poses(convention="world")
    torch.testing.assert_close(camera.data.pos_w.torch, camera.positions)
    _assert_same_rotations(camera.data.quat_w_world.torch, world_orientations)
    [(render_data, positions, orientations)] = camera._renderer.poses
    assert render_data is camera._render_data
    torch.testing.assert_close(positions, camera.positions)
    _assert_same_rotations(orientations, world_orientations)
    assert camera.frame.torch.tolist() == [0] * _NUM_CAMERAS

    # A partial refresh leaves the other camera's cached pose unchanged.
    moved = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    camera.positions = moved
    camera.refresh_world_poses(env_ids=[1])

    torch.testing.assert_close(camera.data.pos_w.torch, torch.tensor([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]))
    assert len(camera._renderer.poses) == 2
    assert camera.frame.torch.tolist() == [0] * _NUM_CAMERAS
