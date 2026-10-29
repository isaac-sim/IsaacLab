# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic checks for per-body gravity observations."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.envs.mdp.observations import body_projected_gravity_b
from isaaclab.managers import SceneEntityCfg
from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import ProxyArray


@pytest.mark.unit
@pytest.mark.parametrize("device", test_devices())
def test_body_projected_gravity_b_stacks_every_selected_body(device):
    """Each selected body receives its own environment's gravity, including integer selections."""
    num_envs = 2
    angles = (0.0, 0.5 * math.pi, math.pi)
    body_quat = torch.tensor([[math.sin(0.5 * a), 0.0, 0.0, math.cos(0.5 * a)] for a in angles], device=device).repeat(
        num_envs, 1, 1
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_quat_w=ProxyArray(wp.from_torch(body_quat, dtype=wp.quat)),
            GRAVITY_VEC_W=ProxyArray(wp.array([[0.0, 0.0, -9.81], [0.0, 9.81, 0.0]], dtype=wp.vec3, device=device)),
        )
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs)
    # R_x(a)^T applied to -Z in the first environment and +Y in the second.
    expected_z = [[0.0, -math.sin(a), -math.cos(a)] for a in angles]
    expected_y = [[0.0, math.cos(a), -math.sin(a)] for a in angles]
    expected = torch.tensor([expected_z, expected_y], device=device).reshape(num_envs, -1)
    torch.testing.assert_close(body_projected_gravity_b(env, SceneEntityCfg("robot")), expected)
    for body_ids in ([1], 1):
        asset_cfg = SceneEntityCfg("robot", body_ids=body_ids)
        torch.testing.assert_close(body_projected_gravity_b(env, asset_cfg), expected[:, 3:6])
