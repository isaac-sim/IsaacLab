# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Asimov-1 velocity environment that need Isaac Sim for physics."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.velocity.config.asimov_1 import mdp
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Velocity-Rough-Asimov1"


@pytest.mark.integration
def test_joint_zero_offsets_are_consistent_across_actions_observations_and_resets():
    """A zero action targets the pose observed as the default pose, and resets keep the sampled offsets."""
    cfg = parse_env_cfg(TASK, device="cuda", num_envs=2)
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        joint_ids = robot.find_joints(cfg.actions.joint_pos.joint_names, preserve_order=True)[0]
        zero_offset = term.joint_zero_offset.clone()
        low, high = cfg.actions.joint_pos.zero_offset_range
        assert ((zero_offset[:, joint_ids] >= low) & (zero_offset[:, joint_ids] <= high)).all()
        assert zero_offset[:, joint_ids].abs().max() > 0.0

        env.step(torch.zeros(2, term.action_dim, device=env.device))
        target = robot.data.joint_pos_target.torch[:, joint_ids]
        default = robot.data.default_joint_pos.torch[:, joint_ids]
        torch.testing.assert_close(target, default + zero_offset[:, joint_ids])

        measured = mdp.joint_pos_rel_measured(env, "joint_pos", SceneEntityCfg("robot"))[:, joint_ids]
        torch.testing.assert_close(measured, robot.data.joint_pos.torch[:, joint_ids] - target)

        env.reset()
        torch.testing.assert_close(term.joint_zero_offset, zero_offset)
    finally:
        env.close()
