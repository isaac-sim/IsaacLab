# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for the dexterous-task math helpers.

Each test asserts behavior on hand-computed cases (known rotations, distances,
and layouts) so quaternion-layout or formula regressions fail loudly without
re-implementing the tested math as a reference.
"""

import math

import pytest
import torch

import isaaclab.utils.math as math_utils

from isaaclab_tasks.core.handover.mdp.rewards import evaluate_handover_success, handover_reward
from isaaclab_tasks.core.reorient.mdp.observations import compute_cube_keypoints, cube_keypoints_from_quat
from isaaclab_tasks.core.reorient.mdp.rewards import evaluate_reorient_success

_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

_SIN45 = math.sin(math.pi / 4)
_COS45 = math.cos(math.pi / 4)
# (x, y, z, w) storage everywhere
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_ROT90_X = (_SIN45, 0.0, 0.0, _COS45)
_ROT180_Z = (0.0, 0.0, 1.0, 0.0)


def _quats(device, *quats):
    return torch.tensor(quats, dtype=torch.float32, device=device)


@pytest.mark.parametrize("device", _DEVICES)
def test_rotation_distance_recovers_known_angles(device):
    _, distance = evaluate_reorient_success(
        _quats(device, _IDENTITY, _ROT90_X, _ROT180_Z), _quats(device, _IDENTITY, _IDENTITY, _IDENTITY), 0.4
    )
    torch.testing.assert_close(distance, torch.tensor([0.0, math.pi / 2, math.pi], device=device), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_reorient_success_thresholds_on_rotation_distance(device):
    success, distance = evaluate_reorient_success(
        _quats(device, _IDENTITY, _ROT90_X), _quats(device, _IDENTITY, _IDENTITY), 0.4
    )
    assert success.tolist() == [True, False]
    torch.testing.assert_close(distance, torch.tensor([0.0, math.pi / 2], device=device), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_goal_quat_error_composes_the_conjugate_product(device):
    # error(q, identity) = q; error(q, q) = identity; the 180-degree case exercises w = 0.
    # This is the composition the goal_quat_diff and object_goal observation terms rely on.
    asset = _quats(device, _ROT90_X, _ROT90_X, _ROT180_Z)
    goal = _quats(device, _IDENTITY, _ROT90_X, _IDENTITY)
    out = math_utils.quat_mul(asset, math_utils.quat_conjugate(goal))
    torch.testing.assert_close(out[0], torch.tensor(_ROT90_X, device=device), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[1], torch.tensor(_IDENTITY, device=device), atol=1e-6, rtol=0.0)
    # 180-degree rotations keep w = 0, so quat_unique must leave them unchanged
    unique = math_utils.quat_unique(out)
    torch.testing.assert_close(unique[2].abs(), torch.tensor(_ROT180_Z, device=device).abs(), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_goal_quat_error_flips_sign_for_negative_real_part(device):
    # error(270-degree X rotation, identity) has w = -cos45 < 0, exercising the sign flip
    rot270_x = (_SIN45, 0.0, 0.0, -_COS45)
    out = math_utils.quat_mul(_quats(device, rot270_x), math_utils.quat_conjugate(_quats(device, _IDENTITY)))
    torch.testing.assert_close(out[0], torch.tensor(rot270_x, device=device), atol=1e-6, rtol=0.0)
    unique = math_utils.quat_unique(out)
    expected = (-_SIN45, 0.0, 0.0, _COS45)
    torch.testing.assert_close(unique[0], torch.tensor(expected, device=device), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_handover_success_measures_env_frame_distance(device):
    object_pos = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]], device=device)
    goal_pos = torch.tensor([[0.0, 0.0, 3.05], [1.0, 0.0, 0.0]], device=device)
    success, distance = evaluate_handover_success(object_pos, goal_pos, 0.1)
    torch.testing.assert_close(distance, torch.tensor([0.05, 1.0], device=device), atol=1e-6, rtol=0.0)
    assert success.tolist() == [True, False]


@pytest.mark.parametrize("device", _DEVICES)
def test_handover_reward_falls_off_exponentially(device):
    scale = 20.0
    distance = torch.tensor([0.0, math.log(2.0) / scale], device=device)
    reward = handover_reward(distance, scale)
    # 2 * exp(-scale * d): d = 0 -> 2.0; d = ln(2)/scale -> 1.0
    torch.testing.assert_close(reward, torch.tensor([2.0, 1.0], device=device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_cube_keypoints_identity_pose_gives_half_side_corners(device):
    pose = torch.zeros(1, 7, device=device)
    pose[0, 6] = 1.0  # identity orientation (x, y, z, w)

    keypoints = compute_cube_keypoints(pose, size=(0.4, 0.6, 0.8))

    corners = {tuple(round(c, 3) for c in corner) for corner in keypoints[0].tolist()}
    expected = {(sx * 0.2, sy * 0.3, sz * 0.4) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)}
    assert corners == expected


@pytest.mark.parametrize("device", _DEVICES)
def test_cube_keypoints_write_into_optional_out_buffer(device):
    pose = torch.zeros(2, 7, device=device)
    pose[:, 6] = 1.0
    out = torch.full((2, 8, 3), torch.nan, dtype=torch.float32, device=device)
    result = compute_cube_keypoints(pose, out=out)
    assert result is out
    assert not out.isnan().any()
    torch.testing.assert_close(out, compute_cube_keypoints(pose))


@pytest.mark.parametrize("device", _DEVICES)
def test_goal_keypoints_are_rotation_only_offsets(device):
    quat = _quats(device, _IDENTITY)
    flattened = cube_keypoints_from_quat(quat, half_size=(0.2, 0.3, 0.4))
    corners = {tuple(round(c, 3) for c in corner) for corner in flattened.view(8, 3).tolist()}
    expected = {(sx * 0.2, sy * 0.3, sz * 0.4) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)}
    assert corners == expected
