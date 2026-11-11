# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MDP event terms."""

import re
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import reset_joints_within_limits_range
from isaaclab.managers import EventTermCfg, SceneEntityCfg

JOINT_NAMES = ["j0", "j1", "j2", "j3"]
# Joints with distinct limits, so applying one joint's limits to another moves its value.
SOFT_JOINT_POS_LIMITS = torch.tensor([[-1.0, 1.0], [-2.0, 2.0], [10.0, 20.0], [-30.0, -20.0]])
SOFT_JOINT_VEL_LIMITS = torch.tensor([1.0, 2.0, 3.0, 4.0])
DEFAULT_JOINT_POS = torch.tensor([0.0, 0.5, 15.0, -25.0])
DEFAULT_JOINT_VEL = torch.tensor([0.5, 1.5, 2.5, 3.5])
# Listed out of joint order, with a range wider than every soft limit.
WIDE_RANGE = {"j3": (-100.0, 100.0), "j1": (-100.0, 100.0)}


@pytest.mark.parametrize(
    "position_range, velocity_range", [(WIDE_RANGE, {}), ({}, WIDE_RANGE)], ids=["position", "velocity"]
)
def test_reset_joints_within_limits_range_clamps_only_listed_joints(position_range, velocity_range) -> None:
    """Listed joints are clamped to their own soft limits, and the other joints keep their default state."""
    num_envs = 64
    torch.manual_seed(0)
    data = SimpleNamespace(
        default_joint_pos=SimpleNamespace(torch=DEFAULT_JOINT_POS.repeat(num_envs, 1)),
        default_joint_vel=SimpleNamespace(torch=DEFAULT_JOINT_VEL.repeat(num_envs, 1)),
        soft_joint_pos_limits=SimpleNamespace(torch=SOFT_JOINT_POS_LIMITS.repeat(num_envs, 1, 1)),
        soft_joint_vel_limits=SimpleNamespace(torch=SOFT_JOINT_VEL_LIMITS.repeat(num_envs, 1)),
    )
    written = {}
    robot = SimpleNamespace(
        device="cpu",
        data=data,
        find_joints=lambda keys: ([i for i, name in enumerate(JOINT_NAMES) if re.fullmatch(keys, name)], None),
        write_joint_position_to_sim_index=lambda position, env_ids: written.update(pos=position),
        write_joint_velocity_to_sim_index=lambda velocity, env_ids: written.update(vel=velocity),
    )
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot})
    params = {
        "position_range": position_range,
        "velocity_range": velocity_range,
        "use_default_offset": False,
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "abs",
    }
    term = reset_joints_within_limits_range(EventTermCfg(func=reset_joints_within_limits_range, params=params), env)

    term(env, torch.arange(num_envs), **params)

    sampled = [JOINT_NAMES.index(name) for name in WIDE_RANGE]
    other = [i for i in range(len(JOINT_NAMES)) if i not in sampled]
    if position_range:
        # the wide range is clamped to each listed joint's own limits, so the samples reach both of them
        lower, upper = SOFT_JOINT_POS_LIMITS[sampled].T
        torch.testing.assert_close(written["pos"][:, sampled].aminmax(dim=0), (lower, upper))
        torch.testing.assert_close(written["pos"][:, other], data.default_joint_pos.torch[:, other])
        torch.testing.assert_close(written["vel"], data.default_joint_vel.torch)
    else:
        limit = SOFT_JOINT_VEL_LIMITS[sampled]
        torch.testing.assert_close(written["vel"][:, sampled].aminmax(dim=0), (-limit, limit))
        torch.testing.assert_close(written["vel"][:, other], data.default_joint_vel.torch[:, other])
        torch.testing.assert_close(written["pos"], data.default_joint_pos.torch)
