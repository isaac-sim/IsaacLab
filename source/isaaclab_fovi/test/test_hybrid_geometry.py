# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab_fovi.sensors.fixed_software_fov import (
    fixed_frustum_validity,
    image_fixation_to_camera_angles,
    image_fixation_to_pan_tilt_angles,
    retinal_ray_directions,
)


def test_center_image_fixation_preserves_camera_axis() -> None:
    angles = image_fixation_to_camera_angles(
        torch.tensor([[0.5, 0.5]]),
        image_height=120,
        image_width=160,
        focal_length_px=100.0,
        principal_point=(80.0, 60.0),
    )

    torch.testing.assert_close(angles, torch.zeros_like(angles))


def test_image_rotation_rotates_normalized_fixation_before_steering() -> None:
    fixation = torch.tensor([[0.5, 0.75]])
    unrotated = image_fixation_to_camera_angles(
        fixation,
        image_height=100,
        image_width=100,
        focal_length_px=100.0,
        principal_point=(50.0, 50.0),
    )
    rotated = image_fixation_to_camera_angles(
        fixation,
        image_height=100,
        image_width=100,
        focal_length_px=100.0,
        principal_point=(50.0, 50.0),
        image_rotation_deg=90.0,
    )

    assert unrotated[0, 0].abs() < 1.0e-7
    assert unrotated[0, 1] < 0.0
    assert rotated[0, 0] > 0.0
    assert rotated[0, 1].abs() < 1.0e-7


@pytest.mark.parametrize(
    "fixation",
    [
        (0.2, 0.8),
        (0.75, 0.15),
        (0.1, 0.3),
    ],
)
def test_pan_tilt_image_conversion_centers_off_axis_rays(
    fixation: tuple[float, float],
) -> None:
    image_height = 160
    image_width = 120
    focal_length_px = 180.0
    command = image_fixation_to_pan_tilt_angles(
        torch.tensor([fixation]),
        image_height=image_height,
        image_width=image_width,
        focal_length_px=focal_length_px,
        principal_point=(image_width * 0.5, image_height * 0.5),
    )[0]
    tilt, pan = command.tolist()

    # The mount is +Y-forward with pan about +Z followed by tilt about the
    # panned +X axis. Compare the resulting optical axis against the pinhole
    # ray independently of the implementation under test.
    optical_axis = torch.tensor(
        [
            -math.sin(pan) * math.cos(tilt),
            math.cos(pan) * math.cos(tilt),
            math.sin(tilt),
        ]
    )
    expected_tangent = torch.tensor(
        [
            (fixation[1] * image_width - image_width * 0.5) / focal_length_px,
            (image_height * 0.5 - fixation[0] * image_height) / focal_length_px,
        ]
    )
    actual_tangent = optical_axis[[0, 2]] / optical_axis[1]

    torch.testing.assert_close(actual_tangent, expected_tangent)


def test_fixed_frustum_masks_direct_rays_after_off_center_steering() -> None:
    cartesian = torch.tensor(
        [
            [0.0, 0.0],
            [-0.9, 0.0],
            [0.9, 0.0],
        ]
    )
    directions = retinal_ray_directions(
        cartesian,
        image_height=120,
        image_width=160,
        focal_length_px=100.0,
        retinal_window_short_side_fraction=1.0,
    )
    fixation = image_fixation_to_camera_angles(
        torch.tensor([[0.5, 0.9]]),
        image_height=120,
        image_width=160,
        focal_length_px=100.0,
        principal_point=(80.0, 60.0),
    )

    valid = fixed_frustum_validity(
        directions,
        fixation,
        image_height=120,
        image_width=160,
        focal_length_px=100.0,
        principal_point=(80.0, 60.0),
    )

    assert valid.shape == (1, 3)
    assert valid[0, 0]
    assert valid[0].sum() < valid.shape[1]


@pytest.mark.parametrize("image_rotation_deg", [-90.0, 0.0, 90.0, 180.0])
def test_fixed_frustum_pan_tilt_matches_physical_mount_kinematics(
    image_rotation_deg: float,
) -> None:
    raw_height = 120
    raw_width = 160
    focal_length_px = 100.0
    if round(image_rotation_deg / 90.0) % 2:
        policy_height, policy_width = raw_width, raw_height
    else:
        policy_height, policy_width = raw_height, raw_width
    cartesian = torch.tensor(
        [
            [0.0, 0.0],
            [-1.0, -0.7],
            [-0.4, 0.8],
            [0.6, -0.5],
            [1.0, 0.9],
        ],
        dtype=torch.float64,
    )
    rays = retinal_ray_directions(
        cartesian,
        image_height=raw_height,
        image_width=raw_width,
        focal_length_px=focal_length_px,
        retinal_window_short_side_fraction=1.0,
        retinal_window_reference_side="long",
        image_rotation_deg=image_rotation_deg,
    )
    mount_angles = image_fixation_to_pan_tilt_angles(
        torch.tensor([[0.18, 0.81]], dtype=torch.float64),
        image_height=policy_height,
        image_width=policy_width,
        focal_length_px=focal_length_px,
        principal_point=(policy_width * 0.5, policy_height * 0.5),
    )

    actual = fixed_frustum_validity(
        rays,
        mount_angles,
        image_height=raw_height,
        image_width=raw_width,
        focal_length_px=focal_length_px,
        principal_point=(raw_width * 0.5, raw_height * 0.5),
        angle_model="pan_tilt",
        image_rotation_deg=image_rotation_deg,
    )

    tilt, pan = mount_angles[0].tolist()
    camera_to_policy = R.from_euler("z", -image_rotation_deg, degrees=True)
    policy_to_camera = camera_to_policy.inv()
    policy_rays = camera_to_policy.apply(rays.numpy())
    mount_rays = np.column_stack((policy_rays[:, 0], -policy_rays[:, 2], policy_rays[:, 1]))
    mount_motion = R.from_euler("z", pan) * R.from_euler("x", tilt)
    moved_mount_rays = mount_motion.apply(mount_rays)
    moved_policy_rays = np.column_stack((moved_mount_rays[:, 0], moved_mount_rays[:, 2], -moved_mount_rays[:, 1]))
    moved_camera_rays = policy_to_camera.apply(moved_policy_rays)
    forward = -moved_camera_rays[:, 2]
    pixel_x = focal_length_px * moved_camera_rays[:, 0] / forward + raw_width * 0.5
    pixel_y = raw_height * 0.5 - focal_length_px * moved_camera_rays[:, 1] / forward
    expected = (forward > 0.0) & (pixel_x >= 0.0) & (pixel_x < raw_width) & (pixel_y >= 0.0) & (pixel_y < raw_height)

    assert torch.equal(actual[0], torch.from_numpy(expected))


@pytest.mark.parametrize(
    ("reference_side", "expected_tangent"),
    (("short", 0.6), ("long", 0.8)),
)
def test_retinal_ray_directions_select_reference_side(
    reference_side: str,
    expected_tangent: float,
) -> None:
    directions = retinal_ray_directions(
        torch.tensor([[1.0, 0.0]]),
        image_height=120,
        image_width=160,
        focal_length_px=100.0,
        retinal_window_short_side_fraction=1.0,
        retinal_window_reference_side=reference_side,
    )

    tangent = directions[0, 0] / -directions[0, 2]
    assert tangent.item() == pytest.approx(expected_tangent)


def test_retinal_ray_directions_reject_invalid_reference_side() -> None:
    with pytest.raises(ValueError, match="retinal_window_reference_side"):
        retinal_ray_directions(
            torch.tensor([[0.0, 0.0]]),
            image_height=120,
            image_width=160,
            focal_length_px=100.0,
            retinal_window_short_side_fraction=1.0,
            retinal_window_reference_side="diagonal",
        )


def test_fixed_fov_ray_and_mask_geometry_is_scale_invariant() -> None:
    cartesian = torch.tensor([[0.0, 0.0], [-0.8, 0.1], [0.9, -0.2]])
    fixation = torch.tensor([[0.3, 0.85]])
    reference_directions = None
    reference_validity = None

    for scale in (1, 4, 8):
        height = 120 * scale
        width = 160 * scale
        focal_length_px = 100.0 * scale
        directions = retinal_ray_directions(
            cartesian,
            image_height=height,
            image_width=width,
            focal_length_px=focal_length_px,
            retinal_window_short_side_fraction=1.0,
        )
        fixation_angles = image_fixation_to_camera_angles(
            fixation,
            image_height=height,
            image_width=width,
            focal_length_px=focal_length_px,
            principal_point=(width * 0.5, height * 0.5),
        )
        validity = fixed_frustum_validity(
            directions,
            fixation_angles,
            image_height=height,
            image_width=width,
            focal_length_px=focal_length_px,
            principal_point=(width * 0.5, height * 0.5),
        )

        if reference_directions is None:
            reference_directions = directions
            reference_validity = validity
        else:
            torch.testing.assert_close(directions, reference_directions)
            assert torch.equal(validity, reference_validity)


def test_fixed_frustum_matches_half_open_image_boundaries() -> None:
    directions = torch.tensor(
        [[-0.6, 0.0, -0.8], [0.6, 0.0, -0.8], [0.0, 0.6, -0.8], [0.0, -0.6, -0.8], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    valid = fixed_frustum_validity(
        directions,
        torch.zeros(1, 2, dtype=torch.float64),
        image_height=150,
        image_width=150,
        focal_length_px=100.0,
        principal_point=(75.0, 75.0),
    )
    assert valid.tolist() == [[True, False, True, False, False]]


@pytest.mark.parametrize("principal_point", [(70.0, 45.0), (95.0, 75.0)])
def test_fixation_optical_axis_projects_to_calibrated_image_location(
    principal_point: tuple[float, float],
) -> None:
    fixation = torch.tensor([[0.25, 0.7]], dtype=torch.float64)
    angles = image_fixation_to_camera_angles(
        fixation,
        image_height=120,
        image_width=160,
        focal_length_px=100.0,
        principal_point=principal_point,
    )
    roll, pitch = angles[0].tolist()
    # Independent rotations project the steered central ray into the dense image.
    motion = R.from_euler("x", roll) * R.from_euler("y", pitch)
    direction = motion.apply(np.array([0.0, 0.0, -1.0]))
    pixel_x = 100.0 * direction[0] / -direction[2] + principal_point[0]
    pixel_y = principal_point[1] - 100.0 * direction[1] / -direction[2]
    assert pixel_x == pytest.approx(0.7 * 160)
    assert pixel_y == pytest.approx(0.25 * 120)
