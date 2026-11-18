# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gaze kernels agree with fovi's camera geometry and with the composition they replace."""

from typing import Literal

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, gaze_rotation
from isaaclab_fovi.sensors.calibrated_geometry import compose_rotations
from isaaclab_fovi.sensors.gaze_kernels import (
    apply_gaze_indexed_kernel,
    apply_gaze_kernel,
    camera_params,
    project_directions,
    publish_grid_kernel,
    set_gaze_kernel,
    torch_stream,
)
from isaaclab_fovi.sensors.native_grid import GRID_STYLE, grid_sample_indices, native_grid_directions

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

pytestmark = pytest.mark.kitless

DEVICE = next(iter(test_devices(DeviceScope.DEFAULT_CUDA)), "cpu")
"""The run's default GPU, or the CPU when the run selects none."""
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


def convert_camera_frame_quaternion(
    orientation: torch.Tensor,
    *,
    origin: Literal["world", "opengl"],
    target: Literal["world", "opengl", "ros"],
) -> torch.Tensor:
    """Reference conversion of world/OpenGL or OpenGL/ROS poses by exact fixed-basis quaternions.

    Args:
        orientation: Camera quaternions in XYZW order, shape (..., 4).
        origin: Input camera-frame convention.
        target: Opposite world/OpenGL convention, or ROS for OpenGL input.

    Returns:
        Normalized float32 quaternions, shape (..., 4). Quaternion sign may change.
    """
    quat = orientation.float()
    quat = quat / quat.norm(dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(-1)
    # Right multiplication by exact fixed-basis quaternions, without allocating
    # device constants or making the matrix round-trip sensitive to AMP/TF32.
    if origin == "world" and target == "opengl":
        components = (w + x - y + z, -w + x + y + z, -w - x - y + z, w - x + y + z)
    elif origin == "opengl" and target == "world":
        components = (-w + x + y - z, w - x + y - z, w + x + y + z, w + x - y - z)
    elif origin == "opengl" and target == "ros":
        # ROS flips the OpenGL Y/Z axes: right-multiply by a half-turn about X.
        return torch.stack((w, z, -y, -x), dim=-1)
    else:
        raise ValueError(f"Unsupported camera conversion: {origin!r} to {target!r}")
    return torch.stack(components, dim=-1) * 0.5


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


def _set_gaze(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The stored angles and gaze rotations :func:`set_gaze_kernel` writes."""
    stored = torch.empty_like(angles)
    rotation = torch.empty(len(angles), 3, 3, device=DEVICE)
    with torch_stream(DEVICE):
        wp.launch(
            set_gaze_kernel,
            dim=len(angles),
            inputs=[wp.from_torch(angles.contiguous(), dtype=wp.vec2f)],
            outputs=[wp.from_torch(stored, dtype=wp.vec2f), wp.from_torch(rotation, dtype=wp.mat33f)],
        )
    return stored, rotation


def _gaze_frames(basis: torch.Tensor) -> list:
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
    to_opengl = convert_camera_frame_quaternion(identity, origin="world", target="opengl")
    to_world = convert_camera_frame_quaternion(identity, origin="opengl", target="world")[0]
    to_ros = convert_camera_frame_quaternion(to_opengl, origin="opengl", target="ros")[0]
    return [
        wp.mat33f(*basis.flatten().tolist()),
        wp.quatf(*to_opengl[0].tolist()),
        wp.quatf(*to_world.tolist()),
        wp.quatf(*to_ros.tolist()),
    ]


@pytest.mark.parametrize("selection", ["all", "mask", "indices"])
def test_gaze_kernels_match_the_torch_composition(selection):
    generator = torch.Generator().manual_seed(1)
    count = 64
    angles = ((torch.rand(count, 2, generator=generator) * 2 - 1) * 40).to(DEVICE)
    base = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=-1).to(DEVICE)
    angle = torch.tensor(0.3)
    c, s = float(angle.cos()), float(angle.sin())
    basis = torch.tensor(((c, s, 0.0), (s, -c, 0.0), (0.0, 0.0, -1.0)), device=DEVICE)

    stored, set_rotation = _set_gaze(angles)
    tilt, pan = angles.deg2rad().unbind(-1)
    direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
    expected_rotation = gaze_rotation(direction, "pan_tilt")
    torch.testing.assert_close(stored, angles, atol=0, rtol=0)
    torch.testing.assert_close(set_rotation, expected_rotation, atol=1e-5, rtol=0)

    # The torch composition the camera used per frame.
    relative = compose_rotations(compose_rotations(basis, expected_rotation), basis.T)
    opengl = convert_camera_frame_quaternion(base, origin="world", target="opengl")
    composed = quat_from_matrix(compose_rotations(matrix_from_quat(opengl), relative))
    expected = convert_camera_frame_quaternion(composed, origin="opengl", target="world")

    selected = torch.arange(count, device=DEVICE) % 3 == 0
    orientation = base.clone()
    rotation = torch.zeros(count, 3, 3, device=DEVICE)
    mount = torch.zeros(count, 4, device=DEVICE)
    inputs = [wp.from_torch(angles, dtype=wp.vec2f), *_gaze_frames(basis)]
    outputs = [
        wp.from_torch(orientation, dtype=wp.quatf),
        wp.from_torch(rotation, dtype=wp.mat33f),
        wp.from_torch(mount, dtype=wp.quatf),
    ]
    with torch_stream(DEVICE):
        if selection == "indices":
            ids = wp.from_torch(torch.nonzero(selected).flatten().to(torch.int32))
            wp.launch(apply_gaze_indexed_kernel, dim=len(ids), inputs=[*inputs, ids], outputs=outputs)
        else:
            mask = wp.from_torch(selected) if selection == "mask" else wp.zeros(1, dtype=wp.bool, device=DEVICE)
            wp.launch(apply_gaze_kernel, dim=count, inputs=[*inputs, mask, int(selection == "mask")], outputs=outputs)
    turned = torch.ones(count, dtype=torch.bool, device=DEVICE) if selection == "all" else selected
    torch.testing.assert_close(
        matrix_from_quat(orientation[turned]), matrix_from_quat(expected[turned]), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(orientation[~turned], base[~turned], atol=0, rtol=0)
    # The turned cameras record their mount before the gaze, in the ROS convention.
    ros = convert_camera_frame_quaternion(
        convert_camera_frame_quaternion(base, origin="world", target="opengl"), origin="opengl", target="ros"
    )
    torch.testing.assert_close(matrix_from_quat(mount[turned]), matrix_from_quat(ros[turned]), atol=1e-5, rtol=0)
    assert not mount[~turned].any()
    # Only the turned cameras record their gaze rotation; the others keep the one their pose was turned by.
    torch.testing.assert_close(rotation[turned], set_rotation[turned], atol=0, rtol=0)
    assert not rotation[~turned].any()


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
    _, rotation = _set_gaze(angles)
    count = len(coords)
    outputs = {
        "rgba": torch.randint(1, 255, (6, count, 1, 4), generator=generator, dtype=torch.uint8).to(DEVICE),
        "depth": (torch.rand(6, count, 1, 1, generator=generator) + 0.5).to(DEVICE),
    }
    valid = camera.project(rays @ rotation.transpose(-1, -2))[1]
    # Environments 1 and 4 were not rendered this frame and keep their previous image.
    updated = torch.tensor([True, False, True, True, False, True], device=DEVICE)

    # The sequence the kernel replaces: footprint mask, sensor mask, then the upright layout. The layout comes from
    # fovi's own grid reshaping rather than the index permutation the kernel uses.
    keep = valid & coords.valid_mask[None]
    expected_grid = {
        name: coords.as_grid(value.squeeze(2) * keep[..., None], sample_dim=1) for name, value in outputs.items()
    }
    indices = grid_sample_indices(coords).to(torch.int32).contiguous()

    visible = torch.zeros(6, count, dtype=torch.bool, device=DEVICE)
    identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for name, value in outputs.items():
        previous = torch.full_like(expected_grid[name], 7)
        target = previous.clone()
        source = value.squeeze(2)
        with torch_stream(DEVICE):
            wp.launch(
                publish_grid_kernel,
                dim=(6, count),
                inputs=[
                    wp.from_torch(source.view(torch.uint8)),
                    wp.from_torch(indices),
                    wp.from_torch(coords.valid_mask.contiguous()),
                    wp.from_torch(updated),
                    wp.from_torch(rays, dtype=wp.vec3f),
                    wp.from_torch(rotation, dtype=wp.mat33f),
                    identity,
                    camera_params(camera),
                    1,
                    int(name == "rgba"),
                    wp.from_torch(visible),
                ],
                outputs=[wp.from_torch(target.view(6, count, -1).view(torch.uint8))],
            )
        # Float32 rounding may flip a sample on the image edge, nothing more.
        same = (target[updated] == expected_grid[name][updated]).all(-1)
        assert same.float().mean() > 0.999
        assert torch.equal(target[~updated], previous[~updated])
    assert (visible[updated] == valid[updated]).float().mean() > 0.999
    assert not visible[~updated].any()
    assert valid.any() and (~valid).any()
