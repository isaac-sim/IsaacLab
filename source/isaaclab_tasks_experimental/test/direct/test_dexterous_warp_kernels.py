# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for the dexterous-task Warp kernels.

Each test asserts kernel behavior on hand-computed cases (known rotations,
distances, and layouts) so quaternion-layout or formula regressions fail
loudly without re-implementing the replaced math as a reference.
"""

import math

import pytest
import torch
import warp as wp
from isaaclab_tasks_experimental.direct.handover.handover_kernels import (
    handover_reward_kernel,
    handover_success_kernel,
    object_goal_kernel,
)
from isaaclab_tasks_experimental.direct.reorient.reorient_kernels import (
    compute_cube_keypoints,
    reorient_success_kernel,
    rotation_distance_kernel,
)

# wp.from_torch below does not auto-initialize warp
wp.init()


def _cube_keypoints(pose: torch.Tensor, size: tuple[float, float, float] = (0.06, 0.06, 0.06)) -> torch.Tensor:
    """Torch convenience wrapper over the Warp-signature keypoints launcher."""
    out = torch.empty(pose.shape[0], 8, 3, dtype=torch.float32, device=pose.device)
    compute_cube_keypoints(
        wp.from_torch(pose.contiguous(), dtype=wp.float32),
        wp.from_torch(out, dtype=wp.vec3),
        size=size,
    )
    return out


_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

_SIN45 = math.sin(math.pi / 4)
_COS45 = math.cos(math.pi / 4)
# (x, y, z, w) storage everywhere
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_ROT90_X = (_SIN45, 0.0, 0.0, _COS45)
_ROT180_Z = (0.0, 0.0, 1.0, 0.0)


def _quats(device, *quats):
    return wp.from_torch(torch.tensor(quats, dtype=torch.float32, device=device), dtype=wp.quatf)


def _vecs(device, *vecs):
    return wp.from_torch(torch.tensor(vecs, dtype=torch.float32, device=device), dtype=wp.vec3f)


@pytest.mark.parametrize("device", _DEVICES)
def test_rotation_distance_recovers_known_angles(device):
    distance = torch.empty(3, device=device)
    wp.launch(
        rotation_distance_kernel,
        dim=3,
        inputs=[_quats(device, _IDENTITY, _ROT90_X, _ROT180_Z), _quats(device, _IDENTITY, _IDENTITY, _IDENTITY)],
        outputs=[wp.from_torch(distance)],
        device=wp.device_from_torch(distance.device),
    )
    torch.testing.assert_close(distance, torch.tensor([0.0, math.pi / 2, math.pi], device=device), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_reorient_success_thresholds_on_rotation_distance(device):
    success = torch.empty(2, dtype=torch.bool, device=device)
    distance = torch.empty(2, device=device)
    wp.launch(
        reorient_success_kernel,
        dim=2,
        inputs=[_quats(device, _IDENTITY, _ROT90_X), _quats(device, _IDENTITY, _IDENTITY), 0.4],
        outputs=[wp.from_torch(success), wp.from_torch(distance)],
        device=wp.device_from_torch(distance.device),
    )
    assert success.tolist() == [True, False]
    torch.testing.assert_close(distance, torch.tensor([0.0, math.pi / 2], device=device), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_handover_success_measures_env_frame_distance(device):
    success = torch.empty(2, dtype=torch.bool, device=device)
    distance = torch.empty(2, device=device)
    wp.launch(
        handover_success_kernel,
        dim=2,
        inputs=[
            _vecs(device, (1.0, 2.0, 3.0), (5.0, 5.0, 5.0)),  # object, world frame
            _vecs(device, (1.0, 2.0, 0.0), (5.0, 5.0, 5.0)),  # environment origins
            _vecs(device, (0.0, 0.0, 3.05), (1.0, 0.0, 0.0)),  # goals, environment frame
            0.1,
        ],
        outputs=[wp.from_torch(success), wp.from_torch(distance)],
        device=wp.device_from_torch(distance.device),
    )
    torch.testing.assert_close(distance, torch.tensor([0.05, 1.0], device=device), atol=1e-6, rtol=0.0)
    assert success.tolist() == [True, False]


@pytest.mark.parametrize("device", _DEVICES)
def test_handover_reward_falls_off_exponentially(device):
    scale = 20.0
    distance = torch.tensor([0.0, math.log(2.0) / scale], device=device)
    reward = torch.empty(2, device=device)
    wp.launch(
        handover_reward_kernel,
        dim=2,
        inputs=[wp.from_torch(distance), scale],
        outputs=[wp.from_torch(reward)],
        device=wp.device_from_torch(reward.device),
    )
    # 2 * exp(-scale * d): d = 0 -> 2.0; d = ln(2)/scale -> 1.0
    torch.testing.assert_close(reward, torch.tensor([2.0, 1.0], device=device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_object_goal_block_layout(device):
    out = torch.empty(1, 24, dtype=torch.float32, device=device)
    wp.launch(
        object_goal_kernel,
        dim=1,
        inputs=[
            _vecs(device, (1.0, 2.0, 3.0)),  # object position, world
            _vecs(device, (1.0, 0.0, 0.0)),  # environment origin
            _quats(device, _ROT90_X),  # object rotation
            _vecs(device, (0.5, 0.0, 0.0)),  # linear velocity
            _vecs(device, (0.0, 10.0, 0.0)),  # angular velocity
            _vecs(device, (0.0, -0.6, 0.5)),  # goal position, environment frame
            _quats(device, _IDENTITY),  # goal rotation
            0.2,  # angular-velocity scale
        ],
        outputs=[wp.from_torch(out)],
        device=wp.device_from_torch(out.device),
    )
    expected = torch.tensor(
        [[0.0, 2.0, 3.0, *_ROT90_X, 0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -0.6, 0.5, *_IDENTITY, *_ROT90_X]],
        device=device,
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_cube_keypoints_identity_pose_gives_half_side_corners(device):
    pose = torch.zeros(1, 7, device=device)
    pose[0, 6] = 1.0  # identity orientation (x, y, z, w)

    keypoints = _cube_keypoints(pose, size=(0.4, 0.6, 0.8))

    corners = {tuple(round(c, 3) for c in corner) for corner in keypoints[0].tolist()}
    expected = {(sx * 0.2, sy * 0.3, sz * 0.4) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)}
    assert corners == expected


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("with_forces", [1, 0, -1])
def test_full_obs_kernel_force_block_modes(device, with_forces):
    """The wrench segment is written (1), omitted (0), or zero-filled without reads (-1)."""
    from isaaclab_tasks_experimental.direct.reorient.reorient_kernels import full_obs_kernel

    num_joints, num_fingers = 2, 1
    # layout: 2J | 13 object | 11 goal | 13F fingertip | (6F forces unless 0) | actions
    force_offset = 2 * num_joints + 13 + 11 + 13 * num_fingers
    obs_dim = force_offset + (6 * num_fingers if with_forces != 0 else 0) + 2
    zeros2 = torch.zeros(1, num_joints, device=device)
    f32 = {"dtype": torch.float32, "device": device}

    out = torch.full((1, obs_dim), torch.nan, **f32)
    actions = torch.tensor([[0.25, -0.75]], **f32)
    # with_forces == -1 models the pre-init sensor: a (1, 1) dummy array whose payload
    # must never be read; the sentinel 7.0 makes an accidental read fail the zero assert
    force = torch.full((1, 1) if with_forces == -1 else (1, 2), 7.0, device=device).unsqueeze(-1).repeat(1, 1, 3)
    torque = 2.0 * force
    wrench_ids = torch.tensor([0 if with_forces == -1 else 1], dtype=torch.int32, device=device)
    wp.launch(
        full_obs_kernel,
        dim=(1, obs_dim),
        inputs=[
            wp.from_torch(zeros2),  # joint_pos
            wp.from_torch(zeros2),  # joint_vel
            wp.from_torch(zeros2 - 1.0),  # lower
            wp.from_torch(zeros2 + 1.0),  # upper
            _vecs(device, (0.0, 0.0, 0.0)),  # object_pos_w
            _vecs(device, (0.0, 0.0, 0.0)),  # env_origins
            _quats(device, _IDENTITY),  # object_quat
            _vecs(device, (0.0, 0.0, 0.0)),  # object_lin_vel
            _vecs(device, (0.0, 0.0, 0.0)),  # object_ang_vel
            _vecs(device, (0.0, 0.0, 0.0)),  # in_hand_pos_e
            _quats(device, _IDENTITY),  # goal_quat
            wp.from_torch(torch.zeros(1, 2, 3, **f32), dtype=wp.vec3f),  # body_pos_w
            wp.from_torch(torch.zeros(1, 2, 4, **f32), dtype=wp.quatf),  # body_quat_w
            wp.from_torch(torch.zeros(1, 2, 6, **f32), dtype=wp.spatial_vectorf),  # body_vel_w
            wp.from_torch(torch.tensor([1], dtype=torch.int32, device=device)),  # finger_ids
            wp.from_torch(force, dtype=wp.vec3f),
            wp.from_torch(torque, dtype=wp.vec3f),
            wp.from_torch(wrench_ids),
            wp.from_torch(actions),
            1.0,  # vel_scale
            0.5,  # force_scale
            with_forces,
        ],
        outputs=[wp.from_torch(out)],
        device=wp.device_from_torch(out.device),
    )
    if with_forces == 1:
        expected_wrench = 0.5 * torch.tensor([7.0, 7.0, 7.0, 14.0, 14.0, 14.0], device=device)
        torch.testing.assert_close(out[0, force_offset : force_offset + 6], expected_wrench)
    elif with_forces == -1:
        torch.testing.assert_close(out[0, force_offset : force_offset + 6], torch.zeros(6, device=device))
    torch.testing.assert_close(out[0, -2:], actions[0])
    assert not out.isnan().any()
