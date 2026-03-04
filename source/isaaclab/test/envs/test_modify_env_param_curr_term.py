# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test texture randomization in the cartpole scene using pytest."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import pytest
import torch
import warp as wp

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


def replace_value(env, env_id, data, value, num_steps):
    if env.common_step_counter > num_steps and data != value:
        return value
    # use the sentinel to indicate “no change”
    return mdp.modify_env_param.NO_CHANGE


@configclass
class CurriculumsCfg:
    modify_observation_joint_pos = CurrTerm(
        # test writing a term's func.
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_pos_rel.func",
            "modify_fn": replace_value,
            "modify_params": {"value": mdp.joint_pos, "num_steps": 1},
        },
    )

    # test writing a term's param that involves dictionary.
    modify_reset_joint_pos = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.reset_cart_position.params.position_range",
            "modify_fn": replace_value,
            "modify_params": {"value": (-0.0, 0.0), "num_steps": 1},
        },
    )

    # test writing a non_term env parameter using modify_env_param.
    modify_episode_max_length = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "cfg.episode_length_s",
            "modify_fn": replace_value,
            "modify_params": {"value": 20, "num_steps": 1},
        },
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_curriculum_modify_env_param(device):
    """Ensure curriculum terms apply correctly after the fallback and replacement."""
    # new USD stage
    sim_utils.create_new_stage()

    # configure the cartpole env
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = 16
    env_cfg.curriculum = CurriculumsCfg()
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot: Articulation = env.scene["robot"]

    # run a few steps under inference mode
    with torch.inference_mode():
        for count in range(3):
            env.reset()
            actions = torch.randn_like(env.action_manager.action)

            if count == 0:
                # test before curriculum kicks in, value agrees with default configuration
                joint_ids = env.event_manager.cfg.reset_cart_position.params["asset_cfg"].joint_ids
                assert env.observation_manager.cfg.policy.joint_pos_rel.func == mdp.joint_pos_rel
                assert torch.any(wp.to_torch(robot.data.joint_pos)[:, joint_ids] != 0.0)
                assert env.max_episode_length_s == env_cfg.episode_length_s

            if count == 2:
                # test after curriculum makes effect, value agrees with new values
                assert env.observation_manager.cfg.policy.joint_pos_rel.func == mdp.joint_pos
                joint_ids = env.event_manager.cfg.reset_cart_position.params["asset_cfg"].joint_ids
                assert torch.all(wp.to_torch(robot.data.joint_pos)[:, joint_ids] == 0.0)
                assert env.max_episode_length_s == 20

            env.step(actions)

    env.close()
