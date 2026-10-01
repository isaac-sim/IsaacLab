# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for the dexterous-task math helpers.

Each test asserts behavior on hand-computed cases (known distances and layouts)
so quaternion-layout or formula regressions fail loudly without re-implementing
the tested math as a reference.
"""

import math

import pytest
import torch

from isaaclab_tasks.core.handover.mdp.rewards import evaluate_handover_success, handover_reward
from isaaclab_tasks.core.reorient.mdp.observations import compute_cube_keypoints, cube_keypoints_from_quat

# Elementwise torch helpers with no device-specific branch, so CPU suffices.
_DEVICES = ["cpu"]

# (x, y, z, w) storage everywhere
_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _quats(device, *quats):
    return torch.tensor(quats, dtype=torch.float32, device=device)


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
