# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA gaze fusion agrees with eager geometry and reads live replay inputs."""

import pytest
import torch
from fovi.sensing.projection import CameraModel, gaze_rotation
from isaaclab_fovi.sensors.calibrated_geometry import (
    calibrated_fixation_rotation,
    calibrated_mount_angles,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("projection", ["pinhole", "fisheye"])
def test_calibrated_gaze_fusion_replay(projection: str) -> None:
    distortion = (0.01, -0.003, 0.0002, -0.00001)
    if projection == "pinhole":
        distortion = (0.01, -0.003, 0.0002, -0.00001, 0.0001)
    camera = CameraModel(projection, (120, 192), (72, 71, 96, 60), distortion)
    axis = torch.linspace(-0.1, 1.1, 32, device="cuda")
    fixation = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            calibrated_fixation_rotation(camera, fixation)
            calibrated_mount_angles(camera, fixation)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        rotation, valid = calibrated_fixation_rotation(camera, fixation)
        angles = calibrated_mount_angles(camera, fixation)

    for delta in (0.0, 0.01, -0.02):
        fixation.add_(delta)
        graph.replay()
        pixels = torch.stack((fixation[:, 1] * 192 - 0.5, fixation[:, 0] * 120 - 0.5), -1)
        direction, expected_valid = camera.unproject(pixels)
        x, y, z = direction.unbind(-1)
        expected_angles = torch.stack((-torch.atan2(y, (x * x + z * z).sqrt()), -torch.atan2(x, z)), -1)
        torch.testing.assert_close(rotation, gaze_rotation(direction, "pan_tilt"), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(angles, expected_angles, atol=2e-6, rtol=2e-6)
        assert torch.equal(valid, expected_valid)
