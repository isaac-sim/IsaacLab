# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gaze kernels agree with fovi's camera geometry and with the composition they replace."""

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, gaze_rotation
from isaaclab_fovi.sensors.calibrated_geometry import compose_rotations, convert_camera_frame_quaternion
from isaaclab_fovi.sensors.gaze_kernels import (
    camera_params,
    project_directions,
    publish_grid_kernel,
    set_gaze_kernel,
    torch_stream,
)
from isaaclab_fovi.sensors.native_grid import (
    GRID_STYLE,
    allocate_grid_outputs,
    copy_grid_outputs,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
)

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
wp.init()

CAMERAS = {
    "ideal pinhole": CameraModel("pinhole", (480, 640), (500.0, 500.0, 319.5, 239.5)),
    "distorted pinhole": CameraModel(
        "pinhole",
        (480, 640),
        (520.0, 515.0, 322.0, 241.0),
        distortion=(-0.28, 0.09, 0.001, -0.002, -0.01, 0.02, -0.01, 0.005),
    ),
    "fisheye with image circle": CameraModel(
        "fisheye",
        (800, 800),
        (230.0, 230.0, 399.5, 399.5),
        distortion=(0.02, -0.01, 0.003, -0.001),
        image_circle=(399.5, 399.5, 380.0),
        max_angle_deg=95.0,
    ),
}


def _directions(count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    directions = torch.randn(count, 3, generator=generator)
    directions[:, 2] = directions[:, 2].abs() * 1.5  # mostly forward, some sideways and behind
    directions[: count // 10, 2] *= -1
    return torch.nn.functional.normalize(directions, dim=-1).to(DEVICE)


@pytest.mark.parametrize("name", CAMERAS)
def test_projection_matches_fovi(name):
    camera = CAMERAS[name]
    directions = _directions(20000)
    expected_pixels, expected_valid = camera.project(directions)
    pixels, valid = project_directions(directions, camera_params(camera))
    agree = valid == expected_valid
    # Float32 rounding may flip a sample that lies on the image edge, nothing more.
    assert agree.float().mean() > 0.9995
    assert expected_valid.any() and (~expected_valid).any()
    torch.testing.assert_close(
        pixels[valid & expected_valid], expected_pixels[valid & expected_valid], rtol=1e-4, atol=1e-2
    )


def _set_gaze(angles: torch.Tensor, basis: torch.Tensor):
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
    to_opengl = convert_camera_frame_quaternion(identity, origin="world", target="opengl")[0]
    to_world = convert_camera_frame_quaternion(identity, origin="opengl", target="world")[0]
    rotation = torch.empty(len(angles), 3, 3, device=DEVICE)
    delta = torch.empty(len(angles), 4, device=DEVICE)
    with torch_stream(DEVICE):
        wp.launch(
            set_gaze_kernel,
            dim=len(angles),
            inputs=[
                wp.from_torch(angles.contiguous(), dtype=wp.vec2f),
                wp.mat33f(*basis.flatten().tolist()),
                wp.quatf(*to_opengl.tolist()),
                wp.quatf(*to_world.tolist()),
            ],
            outputs=[wp.from_torch(rotation, dtype=wp.mat33f), wp.from_torch(delta, dtype=wp.quatf)],
        )
    return rotation, delta


def test_gaze_kernel_matches_the_torch_composition():
    generator = torch.Generator().manual_seed(1)
    angles = ((torch.rand(64, 2, generator=generator) * 2 - 1) * 40).to(DEVICE)
    base = torch.nn.functional.normalize(torch.randn(64, 4, generator=generator), dim=-1).to(DEVICE)
    angle = torch.tensor(0.3)
    c, s = float(angle.cos()), float(angle.sin())
    basis = torch.tensor(((c, s, 0.0), (s, -c, 0.0), (0.0, 0.0, -1.0)), device=DEVICE)

    rotation, delta = _set_gaze(angles, basis)
    tilt, pan = angles.deg2rad().unbind(-1)
    direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
    expected_rotation = gaze_rotation(direction, "pan_tilt")
    torch.testing.assert_close(rotation, expected_rotation, atol=1e-5, rtol=0)

    # The torch composition the camera used per frame.
    relative = compose_rotations(compose_rotations(basis, expected_rotation), basis.T)
    opengl = convert_camera_frame_quaternion(base, origin="world", target="opengl")
    composed = quat_from_matrix(compose_rotations(matrix_from_quat(opengl), relative))
    expected = convert_camera_frame_quaternion(composed, origin="opengl", target="world")

    result = torch.empty_like(base)
    for i in range(len(base)):
        product = wp.normalize(wp.quatf(*base[i].tolist()) * wp.quatf(*delta[i].tolist()))
        result[i] = torch.tensor([product[0], product[1], product[2], product[3]])
    torch.testing.assert_close(matrix_from_quat(result), matrix_from_quat(expected), atol=1e-5, rtol=0)


def test_publish_kernel_matches_masking_then_copying_to_the_grid():
    camera = CAMERAS["distorted pinhole"]
    coords = SamplingCoords(
        camera.field_of_view("short"),
        0.591,
        24,
        style=GRID_STYLE,
        field_geometry="spherical",
        device=DEVICE,
    )
    rays = (native_grid_directions(coords) * coords.cartesian.new_tensor((1, -1, -1))).float().contiguous()
    generator = torch.Generator().manual_seed(2)
    angles = ((torch.rand(6, 2, generator=generator) * 2 - 1) * 25).to(DEVICE)
    rotation, _ = _set_gaze(angles, torch.eye(3, device=DEVICE))
    count = len(coords)
    outputs = {
        "rgba": torch.randint(1, 255, (6, count, 1, 4), generator=generator, dtype=torch.uint8).to(DEVICE),
        "depth": (torch.rand(6, count, 1, 1, generator=generator) + 0.5).to(DEVICE),
    }
    indices = grid_sample_indices(coords)
    valid = camera.project(rays @ rotation.transpose(-1, -2))[1]

    # The sequence the kernel replaces: footprint mask, sensor mask, then the grid layout.
    expected = {name: value.clone() for name, value in outputs.items()}
    mask_grid_samples(expected, coords)
    for value in expected.values():
        value.masked_fill_(~valid[:, :, None, None], 0)
    expected_grid = allocate_grid_outputs(expected, coords)
    copy_grid_outputs(expected, expected_grid, indices)

    visible = torch.zeros(6, count, dtype=torch.bool, device=DEVICE)
    for name, value in outputs.items():
        target = torch.zeros_like(expected_grid[name])
        source = value.squeeze(2)
        with torch_stream(DEVICE):
            wp.launch(
                publish_grid_kernel,
                dim=(6, count),
                inputs=[
                    wp.from_torch(source.view(torch.uint8)),
                    wp.from_torch(indices.to(torch.int32).contiguous()),
                    wp.from_torch(coords.valid_mask.contiguous()),
                    wp.from_torch(rays, dtype=wp.vec3f),
                    wp.from_torch(rotation, dtype=wp.mat33f),
                    camera_params(camera),
                    1,
                    int(name == "rgba"),
                    wp.from_torch(visible),
                ],
                outputs=[wp.from_torch(target.view(6, count, -1).view(torch.uint8))],
            )
        # Float32 rounding may flip a sample on the image edge, nothing more.
        same = (target == expected_grid[name]).all(-1)
        assert same.float().mean() > 0.999
    assert (visible == valid).float().mean() > 0.999
    assert valid.any() and (~valid).any()
