# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for joint-state event terms that sample from per-joint range dictionaries."""

import re
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import reset_joints_within_limits_range
from isaaclab.managers import EventTermCfg, SceneEntityCfg

pytestmark = pytest.mark.unit

JOINT_NAMES = ["j0", "j1", "j2", "j3"]
# Joints with distinct limits, so applying one joint's limits to another moves its value.
SOFT_JOINT_POS_LIMITS = [[-1.0, 1.0], [-2.0, 2.0], [10.0, 20.0], [-30.0, -20.0]]
DEFAULT_JOINT_POS = [0.0, 0.5, 15.0, -25.0]
SOFT_JOINT_VEL_LIMITS = [1.0, 2.0, 3.0, 4.0]
DEFAULT_JOINT_VEL = [0.5, 1.5, 2.5, 3.5]


def _make_env(device: str, num_envs: int = 64) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Create an environment double whose robot records the written joint state."""

    def _per_env(values: list) -> SimpleNamespace:
        tensor = torch.tensor(values, device=device)
        return SimpleNamespace(torch=tensor.expand(num_envs, *tensor.shape).clone())

    asset = SimpleNamespace(
        device=device,
        data=SimpleNamespace(
            default_joint_pos=_per_env(DEFAULT_JOINT_POS),
            default_joint_vel=_per_env(DEFAULT_JOINT_VEL),
            soft_joint_pos_limits=_per_env(SOFT_JOINT_POS_LIMITS),
            soft_joint_vel_limits=_per_env(SOFT_JOINT_VEL_LIMITS),
        ),
        written={},
    )

    def find_joints(name_keys: str):
        joint_ids = [i for i, name in enumerate(JOINT_NAMES) if re.fullmatch(name_keys, name)]
        return joint_ids, [JOINT_NAMES[i] for i in joint_ids]

    asset.find_joints = find_joints
    asset.write_joint_position_to_sim_index = lambda position, env_ids: asset.written.update(pos=position)
    asset.write_joint_velocity_to_sim_index = lambda velocity, env_ids: asset.written.update(vel=velocity)
    env = SimpleNamespace(num_envs=num_envs, device=device, scene={"robot": asset})
    return env, asset


@pytest.mark.parametrize(
    "position_range, velocity_range, sampled_pos_joints, sampled_vel_joints",
    [
        pytest.param({"j1": (-5.0, 5.0)}, {"j2": (-10.0, 10.0)}, [1], [2], id="one_joint"),
        pytest.param({"j0|j1": (-5.0, 5.0)}, {"j0|j1": (-10.0, 10.0)}, [0, 1], [0, 1], id="joint_subset"),
        pytest.param(
            {"j2|j3": (None, None), "j0|j1": (None, None)},
            {"j2|j3": (None, None), "j0|j1": (None, None)},
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            id="all_joints_out_of_order",
        ),
        pytest.param({}, {"j3": (-10.0, 10.0)}, [], [3], id="empty_position_range"),
        pytest.param({"j0": (-5.0, 5.0)}, {}, [0], [], id="empty_velocity_range"),
    ],
)
def test_reset_joints_within_limits_range_respects_each_joint(
    position_range, velocity_range, sampled_pos_joints, sampled_vel_joints
):
    """Sampled joints stay in their own soft limits and joints without a range keep their default state."""
    device = "cpu"
    torch.manual_seed(0)
    env, asset = _make_env(device)
    params = {
        "position_range": position_range,
        "velocity_range": velocity_range,
        "use_default_offset": False,
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "abs",
    }
    term = reset_joints_within_limits_range(EventTermCfg(func=reset_joints_within_limits_range, params=params), env)

    term(env, torch.arange(env.num_envs, device=device), **params)

    pos_limits = torch.tensor(SOFT_JOINT_POS_LIMITS, device=device)
    vel_limits = torch.tensor(SOFT_JOINT_VEL_LIMITS, device=device)
    for joint_id in range(len(JOINT_NAMES)):
        joint_pos = asset.written["pos"][:, joint_id]
        joint_vel = asset.written["vel"][:, joint_id]
        if joint_id in sampled_pos_joints:
            assert ((joint_pos >= pos_limits[joint_id, 0]) & (joint_pos <= pos_limits[joint_id, 1])).all()
            assert joint_pos.unique().numel() > 1
        else:
            torch.testing.assert_close(joint_pos, asset.data.default_joint_pos.torch[:, joint_id])
        if joint_id in sampled_vel_joints:
            assert (joint_vel.abs() <= vel_limits[joint_id]).all()
            assert joint_vel.unique().numel() > 1
        else:
            torch.testing.assert_close(joint_vel, asset.data.default_joint_vel.torch[:, joint_id])
