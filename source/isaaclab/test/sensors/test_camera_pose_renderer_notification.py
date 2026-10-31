# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pin the camera pose-refresh contract that external renderer borrowers rely on."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import warp as wp

from isaaclab.sensors.camera.camera import Camera


def test_camera_pose_refresh_notifies_its_renderer():
    """``Camera._update_poses`` must forward the refreshed world pose to its renderer.

    The OVXR adapter refreshes selected PiP camera poses through this method before each scene
    publication; a refactor that stops notifying the renderer would leave those panels stale.
    """
    camera = Camera.__new__(Camera)
    pose = SimpleNamespace(warp=wp.zeros(1, dtype=wp.vec3f, device="cpu"))
    orientation = SimpleNamespace(warp=wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device="cpu"))
    camera._sensor_prims = [object()]
    camera._view = SimpleNamespace(get_world_poses=lambda _env_ids: (pose, orientation))
    camera._update_camera_state = MagicMock()
    camera._render_data = object()
    camera._renderer = MagicMock()
    camera._data = SimpleNamespace(pos_w="positions", quat_w_world="orientations", intrinsic_matrices="intrinsics")

    camera._update_poses()

    camera._update_camera_state.assert_called_once()
    assert camera._update_camera_state.call_args.kwargs["update_pose"] is True
    camera._renderer.update_camera.assert_called_once_with(
        camera._render_data, "positions", "orientations", "intrinsics"
    )
