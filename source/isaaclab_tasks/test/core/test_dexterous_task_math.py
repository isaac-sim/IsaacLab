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
