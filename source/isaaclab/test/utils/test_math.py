# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest
import scipy.spatial.transform as scipy_tf
import torch

import isaaclab.utils.math as math_utils
from isaaclab.test.utils import test_devices

pytestmark = pytest.mark.unit

DECIMAL_PRECISION = 5
"""Decimal places compared against scipy references (float32 operations are inexact)."""

EULER_ANGLE_SETS = [
    [0.0, 0.0, 0.0],
    [math.pi / 2.0, 0.0, 0.0],
    [0.0, math.pi / 2.0, 0.0],
    [0.0, 0.0, math.pi / 2.0],
    [1.5708, -2.75, 0.1],
    [0.1, math.pi, math.pi / 2],
]


def random_pose(n: int, device: str, scale: float = 1000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random translations in [-scale, scale] and unique unit quaternions."""
    t = torch.empty((n, 3), device=device).uniform_(-scale, scale)
    q = math_utils.quat_unique(math_utils.random_orientation(n, device=device))
    return t, q


def homogeneous(t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    mat = torch.eye(4, device=t.device).repeat(t.shape[0], 1, 1)
    mat[:, :3, 3] = t
    mat[:, :3, :3] = math_utils.matrix_from_quat(q)
    return mat


def scipy_rotation(q: torch.Tensor) -> scipy_tf.Rotation:
    return scipy_tf.Rotation.from_quat(q.cpu().numpy())


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((5, 4, 3), (10, 2)))
def test_scale_unscale_transform(device, size):
    inputs = torch.arange(math.prod(size), device=device, dtype=torch.float32).reshape(size)
    # (lower, upper, expected scaled output): same-shape bounds, broadcast bounds, asymmetric bounds
    cases = [
        (-2.0 * torch.ones(size, device=device), 2.0 * torch.ones(size, device=device), inputs / 2.0),
        (-3.0 * torch.ones(size[1:], device=device), 3.0 * torch.ones(size[1:], device=device), inputs / 3.0),
        (-3.0 * torch.ones(size[1:], device=device), 2.0 * torch.ones(size[1:], device=device), (inputs + 0.5) / 2.5),
    ]
    for lower, upper, expected in cases:
        scaled = math_utils.scale_transform(inputs, lower, upper)
        torch.testing.assert_close(scaled, expected)
        torch.testing.assert_close(math_utils.unscale_transform(scaled, lower, upper), inputs)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((5, 4, 3), (10, 2)))
def test_saturate(device, size):
    inputs = torch.arange(math.prod(size), device=device, dtype=torch.float32).reshape(size)
    for bounds_shape in (size, size[1:]):
        lower = -2.0 * torch.ones(bounds_shape, device=device)
        upper = 3.0 * torch.ones(bounds_shape, device=device)
        output = math_utils.saturate(inputs, lower, upper)
        torch.testing.assert_close(output, torch.clamp(inputs, -2.0, 3.0))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((5, 4, 3), (10, 2)))
def test_normalize(device, size):
    inputs = torch.arange(math.prod(size), device=device, dtype=torch.float32).reshape(size)
    norm = torch.linalg.norm(math_utils.normalize(inputs), dim=-1)
    torch.testing.assert_close(norm, torch.ones(size[:-1], device=device))


@pytest.mark.parametrize("device", test_devices())
def test_copysign(device):
    other = torch.arange(20, device=device, dtype=torch.float32).reshape(10, 2)
    other[:, 1] *= -1.0
    for mag in (2.0, -3.0):
        expected = abs(mag) * torch.ones_like(other)
        expected[:, 1] *= -1.0
        torch.testing.assert_close(math_utils.copysign(mag, other), expected)


@pytest.mark.parametrize("device", test_devices())
def test_is_identity_pose(device):
    identity_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    for shape in ((), (3,)):
        pos = torch.zeros(*shape, 3, device=device)
        rot = identity_rot.expand(*shape, 4).clone()
        assert math_utils.is_identity_pose(pos, rot) is True
        pos[..., 0] = 1.0
        rot[..., 1] = 1.0
        assert math_utils.is_identity_pose(pos, rot) is False


@pytest.mark.parametrize("device", test_devices())
def test_axis_angle_from_quat(device):
    # (2, 2, 4) quaternions with their (2, 2, 3) axis-angle counterparts
    quat = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 1.0], [0.142006, 0.0, 0.5206887, 0.8418536]],
            [[0.0, 0.0, 0.0, 1.0], [0.0995007, 0.0995007, 0.0995007, 0.9850375]],
        ],
        device=device,
    )
    expected = torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]], [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]], device=device)
    torch.testing.assert_close(math_utils.axis_angle_from_quat(quat), expected)
    torch.testing.assert_close(math_utils.axis_angle_from_quat(quat[0]), expected[0])

    # very small angles exercise the Taylor-expansion branch
    theta = 1e-7
    axis = torch.tensor([-0.302286, 0.205494, -0.930803], device=device)
    quat = torch.cat((axis * math.sin(theta / 2), torch.tensor([math.cos(theta / 2)], device=device)))
    torch.testing.assert_close(math_utils.axis_angle_from_quat(quat), theta * axis)


@pytest.mark.parametrize("device", test_devices())
def test_quat_error_magnitude(device):
    q1 = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, device=device)
    q2 = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.7071068, 0.0, 0.0, 0.7071068], [0.0, 1.0, 0.0, 0.0]], device=device)
    expected = torch.tensor([0.0, math.pi / 2, math.pi], device=device)
    torch.testing.assert_close(math_utils.quat_error_magnitude(q1, q2), expected)
    torch.testing.assert_close(math_utils.quat_error_magnitude(q1[1], q2[1]), expected[1])


@pytest.mark.parametrize("device", test_devices())
def test_quat_unique_is_sign_invariant(device):
    quats_1 = math_utils.random_orientation(num=64, device=device)
    quats_2 = math_utils.random_orientation(num=64, device=device)
    unique_1 = math_utils.quat_unique(quats_1)
    unique_2 = math_utils.quat_unique(quats_2)

    assert torch.all(unique_1[:, 3] > 0)
    negative = quats_1[:, 3] < 0
    torch.testing.assert_close(unique_1[negative], -quats_1[negative])
    torch.testing.assert_close(unique_1[~negative], quats_1[~negative])

    # products and error magnitudes do not depend on the sign of the input quaternions
    reference = math_utils.quat_unique(math_utils.quat_mul(quats_1, math_utils.quat_conjugate(quats_2)))
    for a, b in ((unique_1, unique_2), (quats_1, unique_2)):
        torch.testing.assert_close(
            math_utils.quat_unique(math_utils.quat_mul(a, math_utils.quat_conjugate(b))), reference
        )
        torch.testing.assert_close(
            math_utils.quat_error_magnitude(a, b), math_utils.quat_error_magnitude(quats_1, quats_2)
        )


@pytest.mark.parametrize("device", test_devices())
def test_convert_camera_frame_orientation_convention(device):
    quats = {
        "ros": torch.tensor([[0.33985114, 0.82047325, -0.42470819, -0.17591989]], device=device),
        "opengl": torch.tensor([[0.17591988, 0.42470818, 0.82047324, 0.33985113]], device=device),
        "world": torch.tensor([[-0.27984815, -0.1159169, 0.88047623, -0.3647052]], device=device),
    }
    for origin, quat in quats.items():
        for target, expected in quats.items():
            torch.testing.assert_close(
                math_utils.convert_camera_frame_orientation_convention(quat, origin, target), expected
            )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("size", ((10, 4), (5, 3, 4)))
def test_convert_quat(device, size):
    quat = torch.zeros(size, device=device)
    quat[..., 0] = 1.0
    expected_xyzw = torch.roll(quat, -1, dims=-1)
    expected_wxyz = torch.roll(quat, 1, dims=-1)
    torch.testing.assert_close(math_utils.convert_quat(quat), expected_xyzw)
    torch.testing.assert_close(math_utils.convert_quat(quat, to="xyzw"), expected_xyzw)
    torch.testing.assert_close(math_utils.convert_quat(quat, to="wxyz"), expected_wxyz)

    with pytest.raises(ValueError):
        math_utils.convert_quat(torch.zeros((10, 5), device=device))
    with pytest.raises(ValueError):
        math_utils.convert_quat(quat, to="xwyz")


@pytest.mark.parametrize("device", test_devices())
def test_quat_conjugate_and_inverse(device):
    quat = math_utils.random_orientation(64, device=device)
    conjugate = math_utils.quat_conjugate(quat)
    torch.testing.assert_close(conjugate[..., :3], -quat[..., :3])
    torch.testing.assert_close(conjugate[..., 3], quat[..., 3])

    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand_as(quat)
    for q in (quat, torch.randn(64, 4, device=device) * 5.0):
        q_inv = math_utils.quat_inv(q)
        torch.testing.assert_close(math_utils.quat_mul(q, q_inv), identity, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(math_utils.quat_mul(q_inv, q), identity, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", test_devices())
def test_quat_from_euler_xyz(device):
    angles = torch.tensor(EULER_ANGLE_SETS, device=device)
    quat = math_utils.quat_unique(math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2]))
    expected = torch.tensor(
        scipy_tf.Rotation.from_euler("xyz", EULER_ANGLE_SETS).as_quat(), device=device, dtype=torch.float
    )
    torch.testing.assert_close(quat, math_utils.quat_unique(expected))


@pytest.mark.parametrize("device", test_devices())
def test_wrap_to_pi(device):
    pi = math.pi
    # fmt: off
    angles = torch.tensor(
        [0.0, pi, -pi, 3 * pi, -3 * pi, 4 * pi, -4 * pi, -2 * pi, -pi - 0.1, -2.8, 3.1, pi + 0.001, pi + 1, 2 * pi + 0.1],  # noqa: E501
        device=device,
    )
    expected = torch.tensor(
        [0.0, pi, -pi, pi, -pi, 0.0, 0.0, 0.0, pi - 0.1, -2.8, 3.1, -pi + 0.001, -pi + 1, 0.1],
        device=device,
    )
    # fmt: on
    torch.testing.assert_close(math_utils.wrap_to_pi(angles), expected)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("shape", ((3,), (16, 3)))
def test_skew_symmetric_matrix(device, shape):
    vec = torch.empty(shape, device=device).uniform_(-1000.0, 1000.0)
    mat = math_utils.skew_symmetric_matrix(vec)
    v = vec.reshape(-1, 3)
    assert mat.shape == (v.shape[0], 3, 3)
    torch.testing.assert_close(mat, -mat.transpose(-1, -2))
    torch.testing.assert_close(mat[:, 0, 1], -v[:, 2])
    torch.testing.assert_close(mat[:, 0, 2], v[:, 1])
    torch.testing.assert_close(mat[:, 1, 2], -v[:, 0])


@pytest.mark.parametrize("device", test_devices())
def test_orthogonalize_perspective_depth(device):
    perspective_depth = torch.tensor([[[10.0, 0.0, 100.0], [0.0, 3000.0, 0.0], [100.0, 0.0, 100.0]]], device=device)
    intrinsics = torch.tensor([[500.0, 0.0, 5.0], [0.0, 500.0, 5.0], [0.0, 0.0, 1.0]], device=device)
    expected = torch.tensor(
        [[[9.9990, 0.0000, 99.9932], [0.0000, 2999.8079, 0.0000], [99.9932, 0.0000, 99.9964]]], device=device
    )
    torch.testing.assert_close(math_utils.orthogonalize_perspective_depth(perspective_depth, intrinsics), expected)


@pytest.mark.parametrize("device", test_devices())
def test_interpolate_poses(device):
    """interpolate_poses matches scipy Slerp for rotations and np.linspace for positions."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        pos_1, rmat1 = math_utils.unmake_pose(math_utils.generate_random_transformation_matrix())
        pos_2, rmat2 = math_utils.unmake_pose(math_utils.generate_random_transformation_matrix())
        num_steps = int(rng.integers(3, 20))
        slerp = scipy_tf.Slerp([0, 1], scipy_tf.Rotation.from_matrix(np.array([rmat1, rmat2])))
        expected_rot = slerp(np.linspace(0, 1, num_steps)).as_matrix()
        expected_pos = np.linspace(pos_1, pos_2, num_steps)

        # interpolate_poses adds the two end poses, hence num_steps - 2 intermediate steps
        poses, _ = math_utils.interpolate_poses(
            math_utils.make_pose(pos_1, rmat1), math_utils.make_pose(pos_2, rmat2), num_steps - 2
        )
        result_pos, result_rot = math_utils.unmake_pose(poses)
        np.testing.assert_array_almost_equal(result_rot, expected_rot, decimal=DECIMAL_PRECISION)
        np.testing.assert_array_almost_equal(result_pos, expected_pos, decimal=DECIMAL_PRECISION)


def test_pose_inv():
    single = math_utils.generate_random_transformation_matrix(pos_boundary=10)
    np.testing.assert_array_almost_equal(
        math_utils.pose_inv(single), np.linalg.inv(single.numpy()), decimal=DECIMAL_PRECISION
    )
    batch = torch.stack([math_utils.generate_random_transformation_matrix(pos_boundary=10) for _ in range(8)])
    np.testing.assert_array_almost_equal(
        math_utils.pose_inv(batch), np.linalg.inv(batch.numpy()), decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("device", test_devices())
def test_quat_to_and_from_angle_axis(device):
    q_rand = math_utils.quat_unique(math_utils.random_orientation(num=64, device=device))
    rot_vec = math_utils.axis_angle_from_quat(q_rand)
    expected = torch.tensor(scipy_rotation(q_rand).as_rotvec(), device=device, dtype=torch.float32)
    torch.testing.assert_close(rot_vec, expected)

    angle = torch.linalg.norm(rot_vec, dim=-1)
    q_value = math_utils.quat_unique(math_utils.quat_from_angle_axis(angle, math_utils.normalize(rot_vec)))
    torch.testing.assert_close(q_value, q_rand)


@pytest.mark.parametrize("device", test_devices())
def test_quat_box_minus(device):
    """The box-minus of two rotations about a shared axis wraps the angle difference to (-pi, pi]."""
    axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    angle_a, angle_b = math.pi - 0.1, -math.pi + 0.1
    quat_a = math_utils.quat_from_angle_axis(torch.tensor([angle_a], device=device), axis)
    quat_b = math_utils.quat_from_angle_axis(torch.tensor([angle_b], device=device), axis)
    expected = axis * math_utils.wrap_to_pi(torch.tensor(angle_a - angle_b, device=device))
    torch.testing.assert_close(math_utils.quat_box_minus(quat_a, quat_b).squeeze(0), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("n", (2, 10, 50))
def test_quat_box_plus_minus_closed_loop(device, n):
    """Integrating 2n increments of pi/n about each principal axis closes the loop; box-minus recovers them."""
    delta = math.pi / n
    delta_angle = torch.tensor(
        [[0, 0, -delta], [0, -delta, 0], [-delta, 0, 0], [0, 0, delta], [0, delta, 0], [delta, 0, 0]], device=device
    )
    trajectory = torch.zeros((len(delta_angle), 2 * n + 1, 4), device=device)
    trajectory[:, 0, 3] = 1.0
    for i in range(1, 2 * n + 1):
        trajectory[:, i] = math_utils.quat_box_plus(trajectory[:, i - 1], delta_angle)
    torch.testing.assert_close(trajectory[:, 0], trajectory[:, -1], atol=1e-4, rtol=1e-4)
    for i in range(2 * n):
        torch.testing.assert_close(
            math_utils.quat_box_minus(trajectory[:, i + 1], trajectory[:, i]), delta_angle, atol=1e-4, rtol=1e-4
        )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("with_translation", [True, False])
@pytest.mark.parametrize("with_rotation", [True, False])
def test_combine_frame_transforms(device, with_translation, with_rotation):
    """combine_frame_transforms matches a homogeneous matrix product; None inputs mean identity."""
    n = 64
    t01, q01 = random_pose(n, device)
    t12, q12 = random_pose(n, device)
    mat_12 = homogeneous(t12 if with_translation else torch.zeros_like(t12), q12)
    if not with_rotation:
        mat_12[:, :3, :3] = torch.eye(3, device=device)
    expected = homogeneous(t01, q01) @ mat_12

    t02, q02 = math_utils.combine_frame_transforms(
        t01, q01, t12 if with_translation else None, q12 if with_rotation else None
    )
    torch.testing.assert_close(t02, expected[:, :3, 3], atol=1e-3, rtol=1e-5)
    torch.testing.assert_close(
        math_utils.quat_unique(q02), math_utils.quat_unique(math_utils.quat_from_matrix(expected[:, :3, :3]))
    )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("with_translation", [True, False])
@pytest.mark.parametrize("with_rotation", [True, False])
def test_subtract_frame_transforms(device, with_translation, with_rotation):
    """subtract_frame_transforms inverts combine_frame_transforms; None inputs mean the world frame."""
    n = 64
    t01, q01 = random_pose(n, device)
    t02, q02 = random_pose(n, device)
    expected_t02 = t02 if with_translation else torch.zeros_like(t02)
    expected_q02 = q02 if with_rotation else math_utils.default_orientation(n, device=device)

    t12, q12 = math_utils.subtract_frame_transforms(
        t01, q01, t02 if with_translation else None, q02 if with_rotation else None
    )
    t02_value, q02_value = math_utils.combine_frame_transforms(t01, q01, t12, q12)
    torch.testing.assert_close(t02_value, expected_t02, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(math_utils.quat_unique(q02_value), math_utils.quat_unique(expected_q02))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("rot_error_type", ("quat", "axis_angle"))
def test_compute_pose_error(device, rot_error_type):
    n = 64
    t01, q01 = random_pose(n, device)
    t02, q02 = random_pose(n, device)
    diff_pos, diff_rot = math_utils.compute_pose_error(t01, q01, t02, q02, rot_error_type=rot_error_type)
    torch.testing.assert_close(diff_pos, t02 - t01)

    axis_angle = math_utils.quat_box_minus(q02, q01)
    if rot_error_type == "axis_angle":
        torch.testing.assert_close(diff_rot, axis_angle)
    else:
        angle = torch.linalg.norm(axis_angle, dim=-1)
        expected = math_utils.quat_from_angle_axis(angle, math_utils.normalize(axis_angle))
        torch.testing.assert_close(math_utils.quat_unique(diff_rot), math_utils.quat_unique(expected))


@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_twist_transform_round_trip(device):
    num_bodies = 16
    t_ab = torch.randn((num_bodies, 3), device=device)
    q_ab = math_utils.random_orientation(num=num_bodies, device=device)
    v_a = torch.randn((num_bodies, 3), device=device)
    w_a = torch.randn((num_bodies, 3), device=device)

    v_b, w_b = math_utils.rigid_body_twist_transform(v_a, w_a, t_ab, q_ab)
    t_ba = -math_utils.quat_rotate_inverse(q_ab, t_ab)
    v_a_back, w_a_back = math_utils.rigid_body_twist_transform(v_b, w_b, t_ba, math_utils.quat_conjugate(q_ab))
    torch.testing.assert_close(v_a_back, v_a)
    torch.testing.assert_close(w_a_back, w_a)


@pytest.mark.parametrize("device", test_devices())
def test_yaw_quat(device):
    quat = torch.tensor([0.0, 0.7071, 0.0, 0.7071], device=device)  # pure pitch has no yaw component
    result = math_utils.yaw_quat(quat)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))
    torch.testing.assert_close(quat, torch.tensor([0.0, 0.7071, 0.0, 0.7071], device=device))


@pytest.mark.parametrize("device", test_devices())
def test_quat_slerp(device):
    rng = np.random.default_rng(0)
    for _ in range(3):
        q1 = scipy_tf.Rotation.from_matrix(math_utils.generate_random_rotation()).as_quat()
        q2 = scipy_tf.Rotation.from_matrix(math_utils.generate_random_rotation()).as_quat()
        slerp = scipy_tf.Slerp([0, 1], scipy_tf.Rotation.from_quat(np.array([q1, q2])))
        for tau in rng.random(3):
            result = math_utils.quat_slerp(torch.tensor(q1, device=device), torch.tensor(q2, device=device), tau)
            np.testing.assert_array_almost_equal(result.cpu(), slerp(tau).as_quat(), decimal=DECIMAL_PRECISION)


def test_interpolate_rotations():
    """interpolate_rotations matches scipy Slerp via both the quaternion and the axis-angle path."""
    rng = np.random.default_rng(0)
    for _ in range(3):
        rmat1 = math_utils.generate_random_rotation()
        rmat2 = math_utils.generate_random_rotation()
        num_steps = int(rng.integers(2, 20))
        slerp = scipy_tf.Slerp([0, 1], scipy_tf.Rotation.from_matrix(np.array([rmat1, rmat2])))
        expected = slerp(np.linspace(0, 1, num_steps)).as_matrix()
        # interpolate_rotations returns one extra rotation, hence num_steps - 1
        for axis_angle in (False, True):
            result = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1, axis_angle=axis_angle)
            np.testing.assert_array_almost_equal(result.cpu(), expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("device", test_devices())
def test_matrix_from_quat_round_trip(device):
    q_rand = math_utils.quat_unique(math_utils.random_orientation(num=64, device=device))
    rot_mat = math_utils.matrix_from_quat(q_rand)
    expected = torch.tensor(scipy_rotation(q_rand).as_matrix(), device=device, dtype=torch.float32)
    torch.testing.assert_close(rot_mat, expected)

    q_value = math_utils.quat_from_matrix(rot_mat)
    torch.testing.assert_close(torch.linalg.norm(q_value, dim=-1), torch.ones(64, device=device))
    torch.testing.assert_close(math_utils.quat_unique(q_value), q_rand)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "convention", ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "ZYZ", "YZY", "XYX", "XZX", "ZXZ", "YXY")
)
def test_matrix_from_euler(device, convention):
    angles = torch.tensor(EULER_ANGLE_SETS, device=device)
    expected = torch.tensor(
        scipy_tf.Rotation.from_euler(convention, EULER_ANGLE_SETS).as_matrix(), device=device, dtype=torch.float
    )
    torch.testing.assert_close(math_utils.matrix_from_euler(angles, convention=convention), expected)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("inverse", [False, True])
def test_quat_apply(device, inverse):
    """quat_apply(_inverse) matches scipy and accepts arbitrary leading batch dimensions."""
    fn = math_utils.quat_apply_inverse if inverse else math_utils.quat_apply
    q_rand = math_utils.random_orientation(num=64, device=device)
    v_rand = math_utils.sample_uniform(-1000, 1000, (64, 3), device=device)
    expected = scipy_rotation(q_rand).apply(v_rand.cpu().numpy(), inverse=inverse)
    result = fn(q_rand, v_rand)
    torch.testing.assert_close(result, torch.tensor(expected, device=device, dtype=torch.float), atol=2e-4, rtol=2e-4)

    batched = fn(q_rand.reshape(4, 2, 8, 4), v_rand.reshape(4, 2, 8, 3))
    torch.testing.assert_close(batched.reshape(64, 3), result)


def test_euler_xyz_from_quat():
    """Single-axis rotations recover their angles in (-pi, pi] and, when requested, in [0, 2pi)."""
    quat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.3826834, 0.0, 0.0, 0.9238795],
            [0.0, -0.3826834, 0.0, 0.9238795],
            [0.0, 0.0, -0.3826834, 0.9238795],
            [-0.7071068, 0.0, 0.0, 0.7071068],
            [0.0, 0.0, -0.7071068, 0.7071068],
            [-0.9238795, 0.0, 0.0, 0.3826834],
            [0.0, 0.0, -0.9238795, 0.3826834],
        ]
    )
    pi = torch.pi
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [pi / 4, 0.0, 0.0],
            [0.0, -pi / 4, 0.0],
            [0.0, 0.0, -pi / 4],
            [-pi / 2, 0.0, 0.0],
            [0.0, 0.0, -pi / 2],
            [-3 * pi / 4, 0.0, 0.0],
            [0.0, 0.0, -3 * pi / 4],
        ]
    )
    torch.testing.assert_close(torch.stack(math_utils.euler_xyz_from_quat(quat), dim=-1), expected)
    wrapped = torch.stack(math_utils.euler_xyz_from_quat(quat, wrap_to_2pi=True), dim=-1)
    torch.testing.assert_close(wrapped, expected % (2 * pi))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    ("eye", "up_axis", "expected_z"),
    [
        ((0.0, 0.0, 5.0), "Z", None),  # looking straight down the up axis
        ((0.0, 5.0, 0.0), "Y", None),
        ((0.0, 0.0, -5.0), "Z", None),  # looking straight up along the up axis
        ((3.0, 0.0, 4.0), "Z", (0.6, 0.0, 0.8)),  # OpenGL: z-axis points from target back to eye
    ],
)
def test_create_rotation_matrix_from_view(device, eye, up_axis, expected_z):
    eyes = torch.tensor([eye], device=device)
    targets = torch.zeros(1, 3, device=device)
    rot = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis=up_axis, device=device)
    identity = torch.eye(3, device=device).expand(1, 3, 3)
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(rot), torch.ones(1, device=device), atol=1e-5, rtol=1e-5)
    if expected_z is not None:
        torch.testing.assert_close(rot[:, :, 2], torch.tensor([expected_z], device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
def test_create_rotation_matrix_from_view_degenerate_rows_are_nan(device):
    """Undefined (eye == target) or non-finite rows produce NaN without corrupting valid rows of the batch."""
    eyes = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 5.0]], device=device)
    targets = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
    rot = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device=device)
    assert torch.isnan(rot[:2]).all()
    torch.testing.assert_close(rot[2] @ rot[2].T, torch.eye(3, device=device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "matrix",
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # singular
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],  # reflection, det = -1
        [[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # non-orthonormal
    ],
    ids=["singular", "reflection", "non_orthonormal"],
)
def test_quat_from_matrix_invalid_rotation_returns_nan(device, matrix):
    quat = math_utils.quat_from_matrix(torch.tensor([matrix], device=device))
    assert torch.isnan(quat).all()
