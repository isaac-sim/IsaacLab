# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for MDP observation terms."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.observations import body_projected_gravity_b
from isaaclab.managers import SceneEntityCfg

pytestmark = pytest.mark.unit


def test_body_projected_gravity_b_stacks_every_selected_body():
    """Gravity is projected into each selected body frame, including the default all-body selection."""
    num_envs = 2
    # Bodies rotated about x by 0, 90 and 180 degrees, as (x, y, z, w) quaternions.
    angles = (0.0, 0.5 * math.pi, math.pi)
    body_quat = torch.tensor([[math.sin(0.5 * a), 0.0, 0.0, math.cos(0.5 * a)] for a in angles]).repeat(num_envs, 1, 1)
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_quat_w=SimpleNamespace(torch=body_quat),
            GRAVITY_VEC_W=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -9.81]]).repeat(num_envs, 1)),
        )
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs)

    # R_x(a)^T @ (0, 0, -1) = (0, -sin(a), -cos(a)).
    expected = torch.tensor([[0.0, -math.sin(a), -math.cos(a)] for a in angles]).reshape(1, -1).repeat(num_envs, 1)
    torch.testing.assert_close(body_projected_gravity_b(env, SceneEntityCfg("robot")), expected)

    # A single body selected as a list or as an integer index.
    for body_ids in ([1], 1):
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.body_ids = body_ids
        torch.testing.assert_close(body_projected_gravity_b(env, asset_cfg), expected[:, 3:6])
