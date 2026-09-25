# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from math import pi as PI

import numpy as np
import pytest
import scipy.spatial.transform as scipy_tf
import torch
import torch.utils.benchmark as benchmark

import isaaclab.utils.math as math_utils
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit

DECIMAL_PRECISION = 5
"""Precision of the test.

This value is used since float operations are inexact. For reference:
https://github.com/pytorch/pytorch/issues/17678
"""


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((5, 4, 3), (10, 2)))
def test_scale_unscale_transform(device, size):
    """Test scale_transform and unscale_transform."""

    inputs = torch.tensor(range(math.prod(size)), device=device, dtype=torch.float32).reshape(size)

    # test with same size
    scale_same = 2.0
    lower_same = -scale_same * torch.ones(size, device=device)
    upper_same = scale_same * torch.ones(size, device=device)
    output_same = math_utils.scale_transform(inputs, lower_same, upper_same)
    expected_output_same = inputs / scale_same
    torch.testing.assert_close(output_same, expected_output_same)
    output_unscale_same = math_utils.unscale_transform(output_same, lower_same, upper_same)
    torch.testing.assert_close(output_unscale_same, inputs)

    # test with broadcasting
    scale_per_batch = 3.0
    lower_per_batch = -scale_per_batch * torch.ones(size[1:], device=device)
    upper_per_batch = scale_per_batch * torch.ones(size[1:], device=device)
    output_per_batch = math_utils.scale_transform(inputs, lower_per_batch, upper_per_batch)
    expected_output_per_batch = inputs / scale_per_batch
    torch.testing.assert_close(output_per_batch, expected_output_per_batch)
    output_unscale_per_batch = math_utils.unscale_transform(output_per_batch, lower_per_batch, upper_per_batch)
    torch.testing.assert_close(output_unscale_per_batch, inputs)

    # test offset between lower and upper
    lower_offset = -3.0 * torch.ones(size[1:], device=device)
    upper_offset = 2.0 * torch.ones(size[1:], device=device)
    output_offset = math_utils.scale_transform(inputs, lower_offset, upper_offset)
    expected_output_offset = (inputs + 0.5) / 2.5
    torch.testing.assert_close(output_offset, expected_output_offset)
    output_unscale_offset = math_utils.unscale_transform(output_offset, lower_offset, upper_offset)
    torch.testing.assert_close(output_unscale_offset, inputs)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((5, 4, 3), (10, 2)))
def test_saturate(device, size):
    "Test saturate of a tensor of differed shapes and device."

    num_elements = math.prod(size)
    input = torch.tensor(range(num_elements), device=device, dtype=torch.float32).reshape(size)

    # testing with same size
    lower_same = -2.0 * torch.ones(size, device=device)
    upper_same = 2.0 * torch.ones(size, device=device)
    output_same = math_utils.saturate(input, lower_same, upper_same)
    assert torch.all(torch.greater_equal(output_same, lower_same)).item()
    assert torch.all(torch.less_equal(output_same, upper_same)).item()
    # testing with broadcasting
    lower_per_batch = -2.0 * torch.ones(size[1:], device=device)
    upper_per_batch = 3.0 * torch.ones(size[1:], device=device)
    output_per_batch = math_utils.saturate(input, lower_per_batch, upper_per_batch)
    assert torch.all(torch.greater_equal(output_per_batch, lower_per_batch)).item()
    assert torch.all(torch.less_equal(output_per_batch, upper_per_batch)).item()


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((5, 4, 3), (10, 2)))
def test_normalize(device, size):
    """Test normalize of a tensor along its last dimension and check the norm of that dimension is close to 1.0."""

    num_elements = math.prod(size)
    input = torch.tensor(range(num_elements), device=device, dtype=torch.float32).reshape(size)
    output = math_utils.normalize(input)
    norm = torch.linalg.norm(output, dim=-1)
    torch.testing.assert_close(norm, torch.ones(size[0:-1], device=device))


@pytest.mark.parametrize("device", test_devices())
def test_copysign(device):
    """Test copysign by copying a sign from both a negative and positive value and
    verify that the new sign is the same.
    """

    size = (10, 2)

    input_mag_pos = 2.0
    input_mag_neg = -3.0

    input = torch.tensor(range(20), device=device, dtype=torch.float32).reshape(size)
    value_pos = math_utils.copysign(input_mag_pos, input)
    value_neg = math_utils.copysign(input_mag_neg, input)
    torch.testing.assert_close(abs(input_mag_pos) * torch.ones_like(input), value_pos)
    torch.testing.assert_close(abs(input_mag_neg) * torch.ones_like(input), value_neg)

    input_neg_dim1 = input.clone()
    input_neg_dim1[:, 1] = -input_neg_dim1[:, 1]
    value_neg_dim1_pos = math_utils.copysign(input_mag_pos, input_neg_dim1)
    value_neg_dim1_neg = math_utils.copysign(input_mag_neg, input_neg_dim1)
    expected_value_neg_dim1_pos = abs(input_mag_pos) * torch.ones_like(input_neg_dim1)
    expected_value_neg_dim1_pos[:, 1] = -expected_value_neg_dim1_pos[:, 1]
    expected_value_neg_dim1_neg = abs(input_mag_neg) * torch.ones_like(input_neg_dim1)
    expected_value_neg_dim1_neg[:, 1] = -expected_value_neg_dim1_neg[:, 1]

    torch.testing.assert_close(expected_value_neg_dim1_pos, value_neg_dim1_pos)
    torch.testing.assert_close(expected_value_neg_dim1_neg, value_neg_dim1_neg)


@pytest.mark.parametrize("device", test_devices())
def test_is_identity_pose(device):
    """Test is_identity_pose method."""
    # Single row identity pose (xyzw format)
    identity_pos = torch.zeros(3, device=device)
    identity_rot = torch.tensor((0.0, 0.0, 0.0, 1.0), device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is True

    # Modified single row pose (not identity)
    identity_pos = torch.tensor([1.0, 0.0, 0.0], device=device)
    identity_rot = torch.tensor((0.0, 1.0, 0.0, 1.0), device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is False

    # Multi-row identity pose (xyzw format)
    identity_pos = torch.zeros(3, 3, device=device)
    identity_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is True

    # Modified multi-row pose (not identity)
    identity_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
    identity_rot = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is False


@pytest.mark.parametrize("device", test_devices())
def test_axis_angle_from_quat(device):
    """Test axis_angle_from_quat method."""
    # Quaternions of the form (2,4) and (2,2,4) in xyzw format
    quats = [
        torch.Tensor([[0.0, 0.0, 0.0, 1.0], [0.142006, 0.0, 0.5206887, 0.8418536]]).to(device),
        torch.Tensor(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.142006, 0.0, 0.5206887, 0.8418536]],
                [[0.0, 0.0, 0.0, 1.0], [0.0995007, 0.0995007, 0.0995007, 0.9850375]],
            ]
        ).to(device),
    ]

    # Angles of the form (2,3) and (2,2,3)
    angles = [
        torch.Tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]]).to(device),
        torch.Tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]], [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]]).to(device),
    ]

    for quat, angle in zip(quats, angles):
        torch.testing.assert_close(math_utils.axis_angle_from_quat(quat), angle)


@pytest.mark.parametrize("device", test_devices())
def test_axis_angle_from_quat_approximation(device):
    """Test the Taylor approximation from axis_angle_from_quat method.

    This test checks for unstable conversions where theta is very small.
    """
    theta = torch.Tensor([0.0000001]).to(device)
    axis = [-0.302286, 0.205494, -0.930803]
    qw = torch.cos(theta / 2)
    quat_vect = [d * torch.sin(theta / 2) for d in axis] + [qw]
    quaternion = torch.tensor(quat_vect, dtype=torch.float32, device=device)
    axis_angle_computed = math_utils.axis_angle_from_quat(quaternion)
    axis_angle_expected = torch.tensor([theta * d for d in axis], dtype=torch.float32, device=device)
    torch.testing.assert_close(axis_angle_computed, axis_angle_expected)


@pytest.mark.parametrize("device", test_devices())
def test_quat_error_magnitude(device):
    """Test quat_error_magnitude method."""
    q1 = torch.Tensor([0, 0, 0, 1]).to(device)
    q2 = torch.Tensor([0, 0, 0, 1]).to(device)
    expected_diff = torch.Tensor([0.0]).to(device)
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    assert math.isclose(q12_diff.item(), expected_diff.item(), rel_tol=1e-5)

    q1 = torch.Tensor([0, 0, 0, 1.0]).to(device)
    q2 = torch.Tensor([0.7071068, 0, 0, 0.7071068]).to(device)
    expected_diff = torch.Tensor([PI / 2]).to(device)
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    assert math.isclose(q12_diff.item(), expected_diff.item(), rel_tol=1e-5)

    q1 = torch.Tensor([0, 0, 0, 1.0]).to(device)
    q2 = torch.Tensor([0.0, 1.0, 0, 0.0]).to(device)
    expected_diff = torch.Tensor([PI]).to(device)
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    assert math.isclose(q12_diff.item(), expected_diff.item(), rel_tol=1e-5)

    q1 = torch.stack(
        [torch.Tensor([0, 0, 0, 1]), torch.Tensor([0, 0, 0, 1.0]), torch.Tensor([0, 0, 0, 1.0])], dim=0
    ).to(device)
    q2 = torch.stack(
        [torch.Tensor([0, 0, 0, 1]), torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([0.0, 1.0, 0, 0.0])],
        dim=0,
    ).to(device)
    expected_diff = (
        torch.stack([torch.Tensor([0.0]), torch.Tensor([PI / 2]), torch.Tensor([PI])], dim=0).flatten().to(device)
    )
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    torch.testing.assert_close(q12_diff, expected_diff)


@pytest.mark.parametrize("device", test_devices())
def test_quat_unique(device):
    """Test quat_unique method."""
    quats = math_utils.random_orientation(num=1024, device=device)
    pos_real_quats = math_utils.quat_unique(quats)
    assert torch.all(pos_real_quats[:, 3] > 0).item()
    non_pos_indices = quats[:, 3] < 0
    torch.testing.assert_close(pos_real_quats[non_pos_indices], -quats[non_pos_indices])
    torch.testing.assert_close(pos_real_quats[~non_pos_indices], quats[~non_pos_indices])


@pytest.mark.parametrize("device", test_devices())
def test_quat_mul_with_quat_unique(device):
    """Test quat_mul method with different quaternions."""
    quats_1 = math_utils.random_orientation(num=1024, device=device)
    quats_2 = math_utils.random_orientation(num=1024, device=device)
    quats_1_pos_real = math_utils.quat_unique(quats_1)
    quats_2_pos_real = math_utils.quat_unique(quats_2)
    quat_result_1 = math_utils.quat_unique(math_utils.quat_mul(quats_1, math_utils.quat_conjugate(quats_2)))
    quat_result_2 = math_utils.quat_unique(
        math_utils.quat_mul(quats_1_pos_real, math_utils.quat_conjugate(quats_2_pos_real))
    )
    quat_result_3 = math_utils.quat_unique(math_utils.quat_mul(quats_1, math_utils.quat_conjugate(quats_2_pos_real)))
    torch.testing.assert_close(quat_result_1, quat_result_2)
    torch.testing.assert_close(quat_result_2, quat_result_3)
    torch.testing.assert_close(quat_result_3, quat_result_1)


@pytest.mark.parametrize("device", test_devices())
def test_quat_error_mag_with_quat_unique(device):
    """Test quat_error_magnitude method with positive real quaternions."""
    quats_1 = math_utils.random_orientation(num=1024, device=device)
    quats_2 = math_utils.random_orientation(num=1024, device=device)
    quats_1_pos_real = math_utils.quat_unique(quats_1)
    quats_2_pos_real = math_utils.quat_unique(quats_2)
    error_1 = math_utils.quat_error_magnitude(quats_1, quats_2)
    error_2 = math_utils.quat_error_magnitude(quats_1_pos_real, quats_2_pos_real)
    error_3 = math_utils.quat_error_magnitude(quats_1, quats_2_pos_real)
    error_4 = math_utils.quat_error_magnitude(quats_1_pos_real, quats_2)
    torch.testing.assert_close(error_1, error_2)
    torch.testing.assert_close(error_2, error_3)
    torch.testing.assert_close(error_3, error_4)
    torch.testing.assert_close(error_4, error_1)


@pytest.mark.parametrize("device", test_devices())
def test_convention_converter(device):
    """Test convert_camera_frame_orientation_convention to and from ros, opengl, and world conventions."""
    quat_ros = torch.tensor([[0.33985114, 0.82047325, -0.42470819, -0.17591989]], device=device)
    quat_opengl = torch.tensor([[0.17591988, 0.42470818, 0.82047324, 0.33985113]], device=device)
    quat_world = torch.tensor([[-0.27984815, -0.1159169, 0.88047623, -0.3647052]], device=device)
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "opengl"), quat_opengl
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "world"), quat_world
    )
    torch.testing.assert_close(math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "ros"), quat_ros)
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "ros"), quat_ros
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world"), quat_world
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "opengl"), quat_opengl
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_world, "world", "ros"), quat_ros
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_world, "world", "opengl"), quat_opengl
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_world, "world", "world"), quat_world
    )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((10, 4), (5, 3, 4)))
def test_convert_quat(device, size):
    """Test convert_quat from xyzw to wxyz and back."""
    quat = torch.zeros(size, device=device)
    quat[..., 0] = 1.0
    value_default = math_utils.convert_quat(quat)
    expected_default = torch.zeros(size, device=device)
    expected_default[..., -1] = 1.0
    torch.testing.assert_close(expected_default, value_default)
    value_to_xyzw = math_utils.convert_quat(quat, to="xyzw")
    expected_to_xyzw = torch.zeros(size, device=device)
    expected_to_xyzw[..., -1] = 1.0
    torch.testing.assert_close(expected_to_xyzw, value_to_xyzw)
    value_to_wxyz = math_utils.convert_quat(quat, to="wxyz")
    expected_to_wxyz = torch.zeros(size, device=device)
    expected_to_wxyz[..., 1] = 1.0
    torch.testing.assert_close(expected_to_wxyz, value_to_wxyz)
    bad_quat = torch.zeros((10, 5), device=device)
    with pytest.raises(ValueError):
        math_utils.convert_quat(bad_quat)
    with pytest.raises(ValueError):
        math_utils.convert_quat(quat, to="xwyz")


@pytest.mark.parametrize("device", test_devices())
def test_quat_conjugate(device):
    """Test quat_conjugate."""
    quat = math_utils.random_orientation(1000, device=device)
    value = math_utils.quat_conjugate(quat)
    expected_imag = -quat[..., :3]
    expected_real = quat[..., 3]
    torch.testing.assert_close(expected_imag, value[..., :3])
    torch.testing.assert_close(expected_real, value[..., 3])


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("num_envs", (1, 10))
@pytest.mark.parametrize(
    "euler_angles",
    [
        [0.0, 0.0, 0.0],
        [math.pi / 2.0, 0.0, 0.0],
        [0.0, math.pi / 2.0, 0.0],
        [0.0, 0.0, math.pi / 2.0],
        [1.5708, -2.75, 0.1],
        [0.1, math.pi, math.pi / 2],
    ],
)
def test_quat_from_euler_xyz(device, num_envs, euler_angles):
    """Test quat_from_euler_xyz against scipy."""
    angles = torch.tensor(euler_angles, device=device).unsqueeze(0).repeat((num_envs, 1))
    quat_value = math_utils.quat_unique(math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2]))
    expected_quat = (
        torch.tensor(
            scipy_tf.Rotation.from_euler("xyz", euler_angles, degrees=False).as_quat(),
            device=device,
            dtype=torch.float,
        )
        .unsqueeze(0)
        .repeat((num_envs, 1))
    )
    torch.testing.assert_close(expected_quat, quat_value)


@pytest.mark.parametrize("device", test_devices())
def test_wrap_to_pi(device):
    """Test wrap_to_pi method."""
    angle = torch.Tensor([0.0]).to(device)
    expected_angle = torch.Tensor([0.0]).to(device)
    torch.testing.assert_close(math_utils.wrap_to_pi(angle), expected_angle)
    angle = torch.Tensor([PI]).to(device)
    expected_angle = torch.Tensor([PI]).to(device)
    torch.testing.assert_close(math_utils.wrap_to_pi(angle), expected_angle)
    angle = torch.Tensor([-PI]).to(device)
    expected_angle = torch.Tensor([-PI]).to(device)
    torch.testing.assert_close(math_utils.wrap_to_pi(angle), expected_angle)
    angle = torch.Tensor([3 * PI, -3 * PI, 4 * PI, -4 * PI]).to(device)
    expected_angle = torch.Tensor([PI, -PI, 0.0, 0.0]).to(device)
    torch.testing.assert_close(math_utils.wrap_to_pi(angle), expected_angle)
    # fmt: off
    angle = torch.Tensor([-2 * PI, - PI - 0.1, -PI, -2.8, 3.1, PI, PI + 0.001, PI + 1, 2 * PI, 2 * PI + 0.1]).to(device)
    expected_angle = torch.Tensor([0.0, PI - 0.1, -PI, -2.8, 3.1 , PI, -PI + 0.001, -PI + 1 , 0.0, 0.1]).to(device)
    # fmt: on
    torch.testing.assert_close(math_utils.wrap_to_pi(angle), expected_angle)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("shape", ((3,), (1024, 3)))
def test_skew_symmetric_matrix(device, shape):
    """Test skew_symmetric_matrix."""
    vec_rand = torch.zeros(shape, device=device)
    vec_rand.uniform_(-1000.0, 1000.0)
    vec_rand_resized = vec_rand.clone().unsqueeze(0) if vec_rand.ndim == 1 else vec_rand.clone()
    mat_value = math_utils.skew_symmetric_matrix(vec_rand)
    expected_shape = (1, 3, 3) if len(shape) == 1 else (shape[0], 3, 3)
    torch.testing.assert_close(
        torch.zeros((expected_shape[0], 3), device=device), torch.diagonal(mat_value, dim1=-2, dim2=-1)
    )
    torch.testing.assert_close(-vec_rand_resized[:, 2], mat_value[:, 0, 1])
    torch.testing.assert_close(vec_rand_resized[:, 1], mat_value[:, 0, 2])
    torch.testing.assert_close(-vec_rand_resized[:, 0], mat_value[:, 1, 2])
    torch.testing.assert_close(vec_rand_resized[:, 2], mat_value[:, 1, 0])
    torch.testing.assert_close(-vec_rand_resized[:, 1], mat_value[:, 2, 0])
    torch.testing.assert_close(vec_rand_resized[:, 0], mat_value[:, 2, 1])


@pytest.mark.parametrize("device", test_devices())
def test_orthogonalize_perspective_depth(device):
    """Test for converting perspective depth to orthogonal depth."""
    perspective_depth = torch.tensor([[[10.0, 0.0, 100.0], [0.0, 3000.0, 0.0], [100.0, 0.0, 100.0]]], device=device)
    intrinsics = torch.tensor([[500.0, 0.0, 5.0], [0.0, 500.0, 5.0], [0.0, 0.0, 1.0]], device=device)
    orthogonal_depth = math_utils.orthogonalize_perspective_depth(perspective_depth, intrinsics)
    expected_orthogonal_depth = torch.tensor(
        [[[9.9990, 0.0000, 99.9932], [0.0000, 2999.8079, 0.0000], [99.9932, 0.0000, 99.9964]]], device=device
    )
    torch.testing.assert_close(orthogonal_depth, expected_orthogonal_depth)


@pytest.mark.parametrize("device", test_devices())
def test_unproject_depth(device):
    """Test unproject_depth against the pinhole camera model."""
    height, width = 3, 4
    fx, fy, cx, cy = 50.0, 40.0, 1.5, 1.0
    depth = torch.rand(height, width, device=device) + 0.5
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)
    points = math_utils.unproject_depth(depth, intrinsics)
    expected = torch.tensor(
        [
            [(u - cx) * depth[v, u].item() / fx, (v - cy) * depth[v, u].item() / fy, depth[v, u].item()]
            for u in range(width)
            for v in range(height)
        ],
        device=device,
    )
    torch.testing.assert_close(points, expected)


@pytest.mark.parametrize("device", test_devices())
def test_combine_frame_transform(device):
    """Test combine_frame_transforms function."""
    pose01 = torch.rand(1, 7, device=device)
    pose01[:, 3:7] = torch.nn.functional.normalize(pose01[..., 3:7], dim=-1)
    pose12 = torch.rand(1, 7, device=device)
    pose12[:, 3:7] = torch.nn.functional.normalize(pose12[..., 3:7], dim=-1)
    pos02, quat02 = math_utils.combine_frame_transforms(
        pose01[..., :3], pose01[..., 3:7], pose12[:, :3], pose12[:, 3:7]
    )
    pos01, quat01 = math_utils.combine_frame_transforms(
        pos02,
        quat02,
        math_utils.quat_rotate(math_utils.quat_inv(pose12[:, 3:7]), -pose12[:, :3]),
        math_utils.quat_inv(pose12[:, 3:7]),
    )
    torch.testing.assert_close(pose01, torch.cat((pos01, quat01), dim=-1))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_interpolate_poses(device, seed):
    """Test interpolate_poses function."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    for _ in range(100):
        mat1 = math_utils.generate_random_transformation_matrix()
        mat2 = math_utils.generate_random_transformation_matrix()
        pos_1, rmat1 = math_utils.unmake_pose(mat1)
        pos_2, rmat2 = math_utils.unmake_pose(mat2)
        key_rots = scipy_tf.Rotation.from_matrix(np.array([rmat1, rmat2]))
        num_steps = np.random.randint(3, 51)
        slerp = scipy_tf.Slerp([0, 1], key_rots)
        expected_quat = slerp(np.linspace(0, 1, num_steps)).as_matrix()
        expected_pos = np.linspace(pos_1, pos_2, num_steps)
        interpolated_poses, _ = math_utils.interpolate_poses(
            math_utils.make_pose(pos_1, rmat1), math_utils.make_pose(pos_2, rmat2), num_steps - 2
        )
        result_pos, result_quat = math_utils.unmake_pose(interpolated_poses)
        np.testing.assert_array_almost_equal(result_quat, expected_quat, decimal=DECIMAL_PRECISION)
        np.testing.assert_array_almost_equal(result_pos, expected_pos, decimal=DECIMAL_PRECISION)


def test_pose_inv():
    """Test pose_inv function."""
    for _ in range(100):
        test_mat = math_utils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * np.pi))
        result = np.array(math_utils.pose_inv(test_mat))
        expected = np.linalg.inv(np.array(test_mat))
        np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)
    test_mats = torch.stack(
        [
            math_utils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * math.pi))
            for _ in range(100)
        ]
    )
    result = np.array(math_utils.pose_inv(test_mats))
    expected = np.linalg.inv(np.array(test_mats))
    np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("device", test_devices())
def test_quat_to_and_from_angle_axis(device):
    """Test axis-angle conversions against scipy."""
    n = 1024
    q_rand = math_utils.quat_unique(math_utils.random_orientation(num=n, device=device))
    rot_vec_value = math_utils.axis_angle_from_quat(q_rand)
    rot_vec_scipy = torch.tensor(
        scipy_tf.Rotation.from_quat(q_rand.cpu().numpy()).as_rotvec(), device=device, dtype=torch.float32
    )
    torch.testing.assert_close(rot_vec_scipy, rot_vec_value)
    torch.testing.assert_close(
        math_utils.axis_angle_from_quat(q_rand.reshape(n // 2, 2, 4)), rot_vec_scipy.reshape(n // 2, 2, 3)
    )
    axis = math_utils.normalize(rot_vec_value.clone())
    angle = torch.linalg.norm(rot_vec_value.clone(), dim=-1)
    q_value = math_utils.quat_unique(math_utils.quat_from_angle_axis(angle, axis))
    torch.testing.assert_close(q_rand, q_value)


@pytest.mark.parametrize("device", test_devices())
def test_quat_box_minus(device):
    """Test quat_box_minus method."""
    axis_angles = torch.tensor([0.0, 0.0, 1.0], device=device)
    angle_a = math.pi - 0.1
    angle_b = -math.pi + 0.1
    quat_a = math_utils.quat_from_angle_axis(torch.tensor([angle_a], device=device), axis_angles)
    quat_b = math_utils.quat_from_angle_axis(torch.tensor([angle_b], device=device), axis_angles)
    axis_diff = math_utils.quat_box_minus(quat_a, quat_b).squeeze(0)
    expected_diff = axis_angles * math_utils.wrap_to_pi(torch.tensor(angle_a - angle_b, device=device))
    torch.testing.assert_close(expected_diff, axis_diff, atol=1e-06, rtol=1e-06)


@pytest.mark.parametrize("device", test_devices())
def test_quat_box_minus_and_quat_box_plus(device):
    """Test consistency of quat_box_plus and quat_box_minus."""
    for n in (2, 10, 100, 1000):
        delta_angle = torch.tensor(
            [
                [0, 0, -math.pi / n],
                [0, -math.pi / n, 0],
                [-math.pi / n, 0, 0],
                [0, 0, math.pi / n],
                [0, math.pi / n, 0],
                [math.pi / n, 0, 0],
            ],
            device=device,
        )
        quat_trajectory = torch.zeros((len(delta_angle), 2 * n + 1, 4), device=device)
        quat_trajectory[:, 0, :] = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(len(delta_angle), 1)
        for i in range(1, 2 * n + 1):
            quat_trajectory[:, i] = math_utils.quat_box_plus(quat_trajectory[:, i - 1], delta_angle)
        torch.testing.assert_close(quat_trajectory[:, 0], quat_trajectory[:, -1], atol=1e-04, rtol=1e-04)
        for i in range(2 * n):
            delta_result = math_utils.quat_box_minus(quat_trajectory[:, i + 1], quat_trajectory[:, i])
            torch.testing.assert_close(delta_result, delta_angle, atol=1e-04, rtol=1e-04)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("t12_inputs", ["True", "False"])
@pytest.mark.parametrize("q12_inputs", ["True", "False"])
def test_combine_frame_transforms(device, t12_inputs, q12_inputs):
    """Test combine_frame_transforms with optional inputs."""
    n = 1024
    t01 = torch.zeros((n, 3), device=device)
    t01.uniform_(-1000.0, 1000.0)
    q01 = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
    mat_01 = torch.eye(4, device=device).unsqueeze(0).repeat(n, 1, 1)
    mat_01[:, 0:3, 3] = t01
    mat_01[:, 0:3, 0:3] = math_utils.matrix_from_quat(q01)
    mat_12 = torch.eye(4, device=device).unsqueeze(0).repeat(n, 1, 1)
    if t12_inputs:
        t12 = torch.zeros((n, 3), device=device)
        t12.uniform_(-1000.0, 1000.0)
        mat_12[:, 0:3, 3] = t12
    else:
        t12 = None
    if q12_inputs:
        q12 = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
        mat_12[:, 0:3, 0:3] = math_utils.matrix_from_quat(q12)
    else:
        q12 = None
    mat_expect = torch.einsum("bij,bjk->bik", mat_01, mat_12)
    expected_translation = mat_expect[:, 0:3, 3]
    expected_quat = math_utils.quat_from_matrix(mat_expect[:, 0:3, 0:3])
    translation_value, quat_value = math_utils.combine_frame_transforms(t01, q01, t12, q12)
    torch.testing.assert_close(expected_translation, translation_value, atol=1e-3, rtol=1e-5)
    torch.testing.assert_close(math_utils.quat_unique(expected_quat), math_utils.quat_unique(quat_value))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("t02_inputs", ["True", "False"])
@pytest.mark.parametrize("q02_inputs", ["True", "False"])
def test_subtract_frame_transforms(device, t02_inputs, q02_inputs):
    """Test subtract_frame_transforms with optional inputs."""
    n = 1024
    t01 = torch.zeros((n, 3), device=device)
    t01.uniform_(-1000.0, 1000.0)
    q01 = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
    if t02_inputs:
        t02 = torch.zeros((n, 3), device=device)
        t02.uniform_(-1000.0, 1000.0)
        t02_expected = t02.clone()
    else:
        t02 = None
        t02_expected = torch.zeros((n, 3), device=device)
    if q02_inputs:
        q02 = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
        q02_expected = q02.clone()
    else:
        q02 = None
        q02_expected = math_utils.default_orientation(n, device=device)
    t12_value, q12_value = math_utils.subtract_frame_transforms(t01, q01, t02, q02)
    t02_compare, q02_compare = math_utils.combine_frame_transforms(t01, q01, t12_value, q12_value)
    torch.testing.assert_close(t02_expected, t02_compare, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(math_utils.quat_unique(q02_expected), math_utils.quat_unique(q02_compare))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("rot_error_type", ("quat", "axis_angle"))
def test_compute_pose_error(device, rot_error_type):
    """Test compute_pose_error for different rot_error_type."""
    n = 1000
    t01 = torch.zeros((n, 3), device=device).uniform_(-1000.0, 1000.0)
    t02 = torch.zeros((n, 3), device=device).uniform_(-1000.0, 1000.0)
    q01 = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
    q02 = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
    diff_pos, diff_rot = math_utils.compute_pose_error(t01, q01, t02, q02, rot_error_type=rot_error_type)
    torch.testing.assert_close(t02 - t01, diff_pos)
    if rot_error_type == "axis_angle":
        torch.testing.assert_close(math_utils.quat_box_minus(q02, q01), diff_rot)
    else:
        axis_angle = math_utils.quat_box_minus(q02, q01)
        axis = math_utils.normalize(axis_angle)
        angle = torch.linalg.norm(axis_angle, dim=-1)
        torch.testing.assert_close(
            math_utils.quat_unique(math_utils.quat_from_angle_axis(angle, axis)), math_utils.quat_unique(diff_rot)
        )


@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_twist_transform(device):
    """Test rigid_body_twist_transform method."""
    num_bodies = 100
    t_AB = torch.randn((num_bodies, 3), device=device)
    q_AB = math_utils.random_orientation(num=num_bodies, device=device)
    v_AA = torch.randn((num_bodies, 3), device=device)
    w_AA = torch.randn((num_bodies, 3), device=device)
    v_BB, w_BB = math_utils.rigid_body_twist_transform(v_AA, w_AA, t_AB, q_AB)
    t_BA = -math_utils.quat_rotate_inverse(q_AB, t_AB)
    q_BA = math_utils.quat_conjugate(q_AB)
    v_AA_, w_AA_ = math_utils.rigid_body_twist_transform(v_BB, w_BB, t_BA, q_BA)
    torch.testing.assert_close(v_AA_, v_AA)
    torch.testing.assert_close(w_AA_, w_AA)


@pytest.mark.parametrize("device", test_devices())
def test_yaw_quat(device):
    """Test yaw_quat."""
    quat_input = torch.tensor([0, 0.7071, 0, 0.7071], device=device)
    cloned_quat_input = quat_input.clone()
    expected_output = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    result = math_utils.yaw_quat(quat_input)
    torch.testing.assert_close(quat_input, cloned_quat_input)
    torch.testing.assert_close(result, expected_output)


@pytest.mark.parametrize("device", test_devices())
def test_quat_slerp(device):
    """Test quat_slerp against scipy."""
    random_rotation_matrices_1 = [math_utils.generate_random_rotation() for _ in range(100)]
    random_rotation_matrices_2 = [math_utils.generate_random_rotation() for _ in range(100)]
    tau_values = np.random.rand(10)
    for rmat1, rmat2 in zip(random_rotation_matrices_1, random_rotation_matrices_2):
        q1 = scipy_tf.Rotation.from_matrix(rmat1).as_quat()
        q2 = scipy_tf.Rotation.from_matrix(rmat2).as_quat()
        slerp = scipy_tf.Slerp([0, 1], scipy_tf.Rotation.from_quat(np.array([q1, q2])))
        q2_tensor = torch.tensor(q2, device=device)
        for tau in tau_values:
            expected = slerp(tau).as_quat()
            result = math_utils.quat_slerp(torch.tensor(q1, device=device), q2_tensor, tau)
            np.testing.assert_array_almost_equal(result.cpu(), expected, decimal=DECIMAL_PRECISION)
        np.testing.assert_array_equal(q2_tensor.cpu().numpy(), q2)


@pytest.mark.parametrize("device", test_devices())
def test_matrix_from_quat(device):
    """Test matrix_from_quat against scipy."""
    n = 1024
    q_rand = math_utils.quat_unique(math_utils.random_orientation(num=n, device=device))
    rot_mat = math_utils.matrix_from_quat(quaternions=q_rand)
    rot_mat_scipy = torch.tensor(
        scipy_tf.Rotation.from_quat(q_rand.cpu().numpy()).as_matrix(), device=device, dtype=torch.float32
    )
    torch.testing.assert_close(rot_mat_scipy, rot_mat)
    q_value = math_utils.quat_unique(math_utils.quat_from_matrix(rot_mat))
    torch.testing.assert_close(q_rand, q_value)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "euler_angles",
    [
        [0.0, 0.0, 0.0],
        [math.pi / 2.0, 0.0, 0.0],
        [0.0, math.pi / 2.0, 0.0],
        [0.0, 0.0, math.pi / 2.0],
        [1.5708, -2.75, 0.1],
        [0.1, math.pi, math.pi / 2],
    ],
)
@pytest.mark.parametrize(
    "convention", ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "ZYZ", "YZY", "XYX", "XZX", "ZXZ", "YXY")
)
def test_matrix_from_euler(device, euler_angles, convention):
    """Test matrix_from_euler against scipy."""
    num_envs = 1024
    angles = torch.tensor(euler_angles, device=device).unsqueeze(0).repeat((num_envs, 1))
    mat_value = math_utils.matrix_from_euler(angles, convention=convention)
    expected_mag = (
        torch.tensor(
            scipy_tf.Rotation.from_euler(convention, euler_angles, degrees=False).as_matrix(),
            device=device,
            dtype=torch.float,
        )
        .unsqueeze(0)
        .repeat((num_envs, 1, 1))
    )
    torch.testing.assert_close(expected_mag, mat_value)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "quat_shape,vec_shape",
    [
        ((1024,), (1024,)),
        ((128, 2, 4), (128, 2, 4)),
        ((), ()),
        ((1,), ()),
        ((), (2, 3)),
        ((2, 3), ()),
        ((2, 3), (2, 1)),
        ((2, 1), (3,)),
        ((2, 3), (3, 2)),
    ],
)
def test_quat_apply(device, inverse, quat_shape, vec_shape):
    """Rotations follow NumPy broadcasting and agree with SciPy, including strided inputs."""
    quat = math_utils.random_orientation(num=2 * math.prod(quat_shape), device=device)[::2].reshape(*quat_shape, 4)
    vec = math_utils.sample_uniform(-1000, 1000, (2 * math.prod(vec_shape), 3), device=device)[::2]
    vec = vec.reshape(*vec_shape, 3)
    apply = math_utils.quat_apply_inverse if inverse else math_utils.quat_apply
    try:
        shape = np.broadcast_shapes(quat_shape, vec_shape) + (3,)
    except ValueError:
        with pytest.raises((RuntimeError, torch.jit.Error)):
            apply(quat, vec)
        return
    quat_np = np.broadcast_to(quat.cpu().numpy(), shape[:-1] + (4,)).reshape(-1, 4)
    vec_np = np.broadcast_to(vec.cpu().numpy(), shape).reshape(-1, 3)
    expected = scipy_tf.Rotation.from_quat(quat_np).apply(vec_np, inverse=inverse).reshape(shape)
    result = apply(quat, vec)
    torch.testing.assert_close(result, torch.tensor(expected, device=device, dtype=vec.dtype), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", test_devices())
def test_quat_inv(device):
    """Test quat_inv method."""
    num = 2048
    q_nonunit = torch.randn(num, 4, device=device) * 5.0
    q_unit = torch.randn(num, 4, device=device)
    q_unit = q_unit / q_unit.norm(dim=-1, keepdim=True)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    for q in (q_nonunit, q_unit):
        q_inv = math_utils.quat_inv(q)
        id_batch = identity.expand_as(q)
        torch.testing.assert_close(math_utils.quat_mul(q, q_inv), id_batch, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(math_utils.quat_mul(q_inv, q), id_batch, atol=1e-4, rtol=1e-4)


def test_quat_apply_benchmarks():
    """Test quat_apply and quat_apply_inverse against old methods and benchmark them."""

    @torch.jit.script
    def bmm_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        shape = q.shape
        q_vec = q[:, :3]
        q_w = q[:, 3]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    @torch.jit.script
    def bmm_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        shape = q.shape
        q_vec = q[:, :3]
        q_w = q[:, 3]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    @torch.jit.script
    def einsum_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_vec = q[..., :3]
        q_w = q[..., 3]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        return a + b + c

    @torch.jit.script
    def einsum_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_vec = q[..., :3]
        q_w = q[..., 3]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        return a - b + c

    for device in test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA):
        q_rand = math_utils.random_orientation(num=1024, device=device)
        v_rand = math_utils.sample_uniform(-1000, 1000, (1024, 3), device=device)
        bmm_result = bmm_quat_rotate(q_rand, v_rand)
        bmm_result_inv = bmm_quat_rotate_inverse(q_rand, v_rand)
        einsum_result = einsum_quat_rotate(q_rand, v_rand)
        einsum_result_inv = einsum_quat_rotate_inverse(q_rand, v_rand)
        new_result = math_utils.quat_apply(q_rand, v_rand)
        new_result_inv = math_utils.quat_apply_inverse(q_rand, v_rand)
        torch.testing.assert_close(bmm_result, new_result, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(bmm_result_inv, new_result_inv, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(einsum_result, new_result, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(einsum_result_inv, new_result_inv, atol=1e-3, rtol=1e-3)

    for device in test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA):
        q_shape = (1024, 2, 5, 4)
        v_shape = (1024, 2, 5, 3)
        num_quats = math.prod(q_shape[:-1])
        q_rand = math_utils.random_orientation(num=num_quats, device=device).reshape(q_shape)
        v_rand = math_utils.sample_uniform(-1000, 1000, v_shape, device=device)

        def iter_quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = math_utils.quat_apply(q[:, i, j], v[:, i, j])
            return out

        def iter_quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = math_utils.quat_apply_inverse(q[:, i, j], v[:, i, j])
            return out

        def iter_bmm_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = bmm_quat_rotate(q[:, i, j], v[:, i, j])
            return out

        def iter_bmm_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = bmm_quat_rotate_inverse(q[:, i, j], v[:, i, j])
            return out

        def iter_einsum_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = einsum_quat_rotate(q[:, i, j], v[:, i, j])
            return out

        def iter_einsum_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = einsum_quat_rotate_inverse(q[:, i, j], v[:, i, j])
            return out

        timers = [
            benchmark.Timer(
                stmt="math_utils.quat_apply(q_rand, v_rand)",
                globals={"math_utils": math_utils, "q_rand": q_rand, "v_rand": v_rand},
            ),
            benchmark.Timer(
                stmt="math_utils.quat_apply_inverse(q_rand, v_rand)",
                globals={"math_utils": math_utils, "q_rand": q_rand, "v_rand": v_rand},
            ),
            benchmark.Timer(
                stmt="iter_quat_apply(q_rand, v_rand)",
                globals={"iter_quat_apply": iter_quat_apply, "q_rand": q_rand, "v_rand": v_rand},
            ),
            benchmark.Timer(
                stmt="iter_quat_apply_inverse(q_rand, v_rand)",
                globals={"iter_quat_apply_inverse": iter_quat_apply_inverse, "q_rand": q_rand, "v_rand": v_rand},
            ),
            benchmark.Timer(
                stmt="iter_bmm_quat_rotate(q_rand, v_rand)",
                globals={"iter_bmm_quat_rotate": iter_bmm_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
            ),
            benchmark.Timer(
                stmt="iter_bmm_quat_rotate_inverse(q_rand, v_rand)",
                globals={
                    "iter_bmm_quat_rotate_inverse": iter_bmm_quat_rotate_inverse,
                    "q_rand": q_rand,
                    "v_rand": v_rand,
                },
            ),
            benchmark.Timer(
                stmt="iter_einsum_quat_rotate(q_rand, v_rand)",
                globals={"iter_einsum_quat_rotate": iter_einsum_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
            ),
            benchmark.Timer(
                stmt="iter_einsum_quat_rotate_inverse(q_rand, v_rand)",
                globals={
                    "iter_einsum_quat_rotate_inverse": iter_einsum_quat_rotate_inverse,
                    "q_rand": q_rand,
                    "v_rand": v_rand,
                },
            ),
        ]
        for timer in timers:
            timer.timeit(number=1000)
        torch.testing.assert_close(math_utils.quat_apply(q_rand, v_rand), iter_quat_apply(q_rand, v_rand))
        torch.testing.assert_close(
            math_utils.quat_apply(q_rand, v_rand), iter_bmm_quat_rotate(q_rand, v_rand), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            math_utils.quat_apply_inverse(q_rand, v_rand), iter_quat_apply_inverse(q_rand, v_rand)
        )
        torch.testing.assert_close(
            math_utils.quat_apply_inverse(q_rand, v_rand),
            iter_bmm_quat_rotate_inverse(q_rand, v_rand),
            atol=1e-3,
            rtol=1e-3,
        )


def test_interpolate_rotations():
    """Test interpolate_rotations against scipy."""
    random_rotation_matrices_1 = [math_utils.generate_random_rotation() for _ in range(100)]
    random_rotation_matrices_2 = [math_utils.generate_random_rotation() for _ in range(100)]
    for rmat1, rmat2 in zip(random_rotation_matrices_1, random_rotation_matrices_2):
        key_rots = scipy_tf.Rotation.from_matrix(np.array([rmat1, rmat2]))
        num_steps = np.random.randint(2, 51)
        slerp = scipy_tf.Slerp([0, 1], key_rots)
        expected = slerp(np.linspace(0, 1, num_steps)).as_matrix()
        result_quat = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1)
        np.testing.assert_array_almost_equal(result_quat.cpu(), expected, decimal=DECIMAL_PRECISION)
        result_axis_angle = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1, axis_angle=True)
        np.testing.assert_array_almost_equal(result_axis_angle.cpu(), expected, decimal=DECIMAL_PRECISION)


def test_euler_xyz_from_quat():
    """Test euler_xyz_from_quat function."""
    quats = [
        torch.Tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.Tensor(
            [[0.3826834, 0.0, 0.0, 0.9238795], [0.0, -0.3826834, 0.0, 0.9238795], [0.0, 0.0, -0.3826834, 0.9238795]]
        ),
        torch.Tensor([[-0.7071068, 0.0, 0.0, 0.7071068], [0.0, 0.0, -0.7071068, 0.7071068]]),
        torch.Tensor([[-0.9238795, 0.0, 0.0, 0.3826834], [0.0, 0.0, -0.9238795, 0.3826834]]),
    ]
    expected_euler_angles = [
        torch.Tensor([[0.0, 0.0, 0.0]]),
        torch.Tensor([[torch.pi / 4, 0.0, 0.0], [0.0, -torch.pi / 4, 0.0], [0.0, 0.0, -torch.pi / 4]]),
        torch.Tensor([[-torch.pi / 2, 0.0, 0.0], [0.0, 0.0, -torch.pi / 2]]),
        torch.Tensor([[-3 * torch.pi / 4, 0.0, 0.0], [0.0, 0.0, -3 * torch.pi / 4]]),
    ]
    for quat, expected in zip(quats, expected_euler_angles):
        output = torch.stack(math_utils.euler_xyz_from_quat(quat), dim=-1)
        torch.testing.assert_close(output, expected)
    for quat, expected in zip(quats, expected_euler_angles):
        wrapped = expected % (2 * torch.pi)
        output = torch.stack(math_utils.euler_xyz_from_quat(quat, wrap_to_2pi=True), dim=-1)
        torch.testing.assert_close(output, wrapped)


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_lookat_along_up_axis_z(device):
    """Camera above target on +Z axis with Z-up should return a valid orthonormal frame."""
    eyes = torch.tensor([[0.0, 0.0, 5.0]], device=device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
    identity = torch.eye(3, device=device).expand(1, 3, 3)
    torch.testing.assert_close(R @ R.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(1, device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_lookat_along_up_axis_y(device):
    """Camera at +Y looking at origin with Y-up should return a valid orthonormal frame."""
    eyes = torch.tensor([[0.0, 5.0, 0.0]], device=device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Y", device=device)
    identity = torch.eye(3, device=device).expand(1, 3, 3)
    torch.testing.assert_close(R @ R.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(1, device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_lookat_along_negative_up_axis(device):
    """Camera below target looking up should return a valid orthonormal frame."""
    eyes = torch.tensor([[0.0, 0.0, -5.0]], device=device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
    identity = torch.eye(3, device=device).expand(1, 3, 3)
    torch.testing.assert_close(R @ R.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(1, device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_zero_forward_returns_nan(device):
    """When eyes equal targets the result should be NaN."""
    eyes = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, eyes.clone(), up_axis="Z", device=device)
    assert torch.isnan(R).all()


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_batched_partial_failure(device):
    """Mixed batch with one degenerate row should produce NaN in that row."""
    eyes = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]], device=device)
    targets = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
    assert torch.isnan(R[0]).any()
    torch.testing.assert_close(R[1] @ R[1].T, torch.eye(3, device=device), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(R[1]), torch.tensor(1.0, device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_quat_from_matrix_unit_norm_on_valid_input(device):
    """quat_from_matrix should produce unit quaternions for valid rotations."""
    n = 100
    q_rand = math_utils.random_orientation(num=n, device=device)
    q_value = math_utils.quat_from_matrix(math_utils.matrix_from_quat(q_rand))
    torch.testing.assert_close(torch.linalg.norm(q_value, dim=-1), torch.ones(n, device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_quat_from_matrix_singular_matrix_returns_nan(device):
    """A singular matrix should produce NaN."""
    singular = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], device=device)
    assert torch.isnan(math_utils.quat_from_matrix(singular)).all()


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_standard(device):
    """Sanity test for an off-axis eye."""
    eyes = torch.tensor([[3.0, 0.0, 4.0]], device=device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
    identity = torch.eye(3, device=device).expand(1, 3, 3)
    torch.testing.assert_close(R @ R.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(1, device=device), atol=1e-5, rtol=1e-5)
    expected_z = torch.tensor([[0.6, 0.0, 0.8]], device=device)
    torch.testing.assert_close(R[:, :, 2], expected_z, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_non_finite_returns_nan(device):
    """Non-finite input should produce NaN rows."""
    eyes = torch.tensor([[float("nan"), 0.0, 0.0]], device=device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    R = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
    assert torch.isnan(R).all()


@pytest.mark.parametrize("device", test_devices())
def test_quat_from_matrix_reflection_returns_nan(device):
    """A reflection matrix should produce NaN."""
    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=device)).unsqueeze(0)
    assert torch.isnan(math_utils.quat_from_matrix(reflection)).all()


@pytest.mark.parametrize("device", test_devices())
def test_quat_from_matrix_non_orthonormal_returns_nan(device):
    """A non-orthonormal matrix should produce NaN."""
    matrix = torch.diag(torch.tensor([1.01, 1.0, 1.0], device=device)).unsqueeze(0)
    assert torch.isnan(math_utils.quat_from_matrix(matrix)).all()
