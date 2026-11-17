# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense raster and foveated renderers must use the same physical lens rays."""

import pytest
import torch
from fovi.sensing.projection import CameraModel, gaze_rotation

from isaaclab_fovi.sensors.calibrated_geometry import calibrated_visibility, retinal_directions
from isaaclab_fovi.sensors.dense_calibrated_geometry import dense_camera_rays


@pytest.mark.parametrize(
    "model,distortion",
    [
        ("pinhole", (-0.06, 0.01, 0.001, -0.002, 0.0)),
        ("fisheye", (0.01, -0.003, 0.0001, 0.0)),
    ],
)
def test_dense_and_foveated_rays_agree_under_off_axis_gaze(model: str, distortion: tuple[float, ...]) -> None:
    camera = CameraModel(model, (120, 192), (100, 103, 93.4, 61.2), distortion)
    dense, valid = dense_camera_rays(camera)
    pixels = torch.tensor([[50, 40], [100, 60], [140, 80]])
    assert valid[pixels[:, 1], pixels[:, 0]].all()
    source_rays = dense[pixels[:, 1], pixels[:, 0]].double() * torch.tensor([1, -1, -1])
    rotation = gaze_rotation(torch.tensor([[0.2, -0.1, 1.0]], dtype=torch.float64), "pan_tilt")[0]
    retinal = source_rays @ rotation
    theta = torch.atan2(torch.linalg.vector_norm(retinal[:, :2], dim=-1), retinal[:, 2])
    xy = torch.nn.functional.normalize(retinal[:, :2], dim=-1) * theta[:, None]
    cartesian = xy * torch.tensor([1, -1]) / torch.deg2rad(torch.tensor(60.0))
    rendered = retinal_directions(cartesian, 120) * torch.tensor([1, -1, -1])
    torch.testing.assert_close(rendered @ rotation.T, source_rays, atol=1e-7, rtol=1e-7)
    recovered, projected_valid = camera.project(rendered @ rotation.T)
    torch.testing.assert_close(recovered, pixels.double(), atol=2e-5, rtol=0)
    assert projected_valid.all()
    torch.testing.assert_close(calibrated_visibility(camera, rendered, rotation[None])[0], projected_valid)
