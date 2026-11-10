# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the array inputs that the camera pose setters accept."""

import pytest
import torch
import warp as wp

from isaaclab.sensors.camera import Camera
from isaaclab.utils.warp import ProxyArray

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_pose_input_accepts_proxy_array():
    """A ProxyArray such as ``scene.env_origins`` converts through its torch view, also on CPU."""
    camera = Camera.__new__(Camera)
    camera._device = "cpu"
    origins = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    converted = camera._as_device_tensor(ProxyArray(wp.from_torch(origins, dtype=wp.vec3f)), 3)

    torch.testing.assert_close(converted, origins)
