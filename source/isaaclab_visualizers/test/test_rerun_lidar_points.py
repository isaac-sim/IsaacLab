# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame selection of lidar hits logged by the Rerun visualizer."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
from isaaclab_visualizers.rerun.lidar_points import lidar_world_points


def _sensor(points: torch.Tensor, valid: torch.Tensor, frame: str, position=None, rotation=None):
    """Build the lidar attributes ``lidar_world_points`` reads."""
    return SimpleNamespace(
        data=SimpleNamespace(points=SimpleNamespace(torch=points), valid=SimpleNamespace(torch=valid)),
        cfg=SimpleNamespace(frame_of_reference=frame),
        _pos_w=None if position is None else SimpleNamespace(torch=position),
        _quat_w=None if rotation is None else SimpleNamespace(torch=rotation),
    )


def test_lidar_world_points_keeps_finite_hits_for_the_requested_envs():
    """Misses drop out, a sensor-frame hit moves by its pose, and other environments stay out."""
    world = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[20.0, 1.0, 2.0], [float("inf"), float("inf"), float("inf")], [30.0, 0.0, 0.0]],
        ]
    )
    world_valid = torch.tensor([[True, False, True], [True, True, False]])
    selected = lidar_world_points(_sensor(world, world_valid, "WORLD"), [1])
    np.testing.assert_allclose(selected, [[20.0, 1.0, 2.0]])

    half = math.sqrt(0.5)
    sensor_points = torch.tensor([[[1.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]])
    sensor_valid = torch.tensor([[True, True]])
    position = torch.tensor([[2.0, 3.0, 4.0]])
    rotation = torch.tensor([[0.0, 0.0, half, half]])
    transformed = lidar_world_points(
        _sensor(sensor_points, sensor_valid, "SENSOR", position, rotation),
        None,
    )
    np.testing.assert_allclose(transformed, [[2.0, 4.0, 4.0]], atol=1e-6)

    unposed = lidar_world_points(_sensor(sensor_points, sensor_valid, "SENSOR"), None)
    assert unposed.shape == (0, 3)
