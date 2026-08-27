# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test curriculum-based environment parameter modification."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.test.env_cfgs import EmptyManagerCfg
from isaaclab.test.integration_scene_cfgs import CartpoleTestSceneCfg
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


def replace_value(env, env_id, data, value, num_steps):
    if env.common_step_counter > num_steps and data != value:
        return value
    # use the sentinel to indicate “no change”
    return mdp.modify_env_param.NO_CHANGE


@configclass
class ActionsCfg:
    """Action specifications for the curriculum test environment."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the curriculum test environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset event specifications for the curriculum test environment."""

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class CurriculumsCfg:
    """Curriculum specifications under test."""

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


@configclass
class CurriculumTestEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal cart-pole environment configuration for curriculum tests."""

    scene: CartpoleTestSceneCfg = CartpoleTestSceneCfg(num_envs=16, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: EmptyManagerCfg = EmptyManagerCfg()
    terminations: EmptyManagerCfg = EmptyManagerCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self) -> None:
        """Set the simulation timing used by the test."""
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_curriculum_modify_env_param(device):
    """Ensure curriculum terms apply correctly after the fallback and replacement."""
    # new USD stage
    sim_utils.create_new_stage()

    # configure the cartpole env
    env_cfg = CurriculumTestEnvCfg()
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
                assert torch.any(robot.data.joint_pos.torch[:, joint_ids] != 0.0)
                assert env.max_episode_length_s == env_cfg.episode_length_s

            if count == 2:
                # test after curriculum makes effect, value agrees with new values
                assert env.observation_manager.cfg.policy.joint_pos_rel.func == mdp.joint_pos
                joint_ids = env.event_manager.cfg.reset_cart_position.params["asset_cfg"].joint_ids
                assert torch.all(robot.data.joint_pos.torch[:, joint_ids] == 0.0)
                assert env.max_episode_length_s == 20

            env.step(actions)

    env.close()
