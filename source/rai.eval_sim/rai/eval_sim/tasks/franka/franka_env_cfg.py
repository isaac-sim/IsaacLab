# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp.observations as obs
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    ActionsCfg,
    EventCfg,
    ReachSceneCfg,
)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=obs.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=obs.joint_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_effort = ObsTerm(func=obs.joint_effort, noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg(
        arm_action=JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint[1-7]"], scale=1.0),
        gripper_action=BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 1.0},
            close_command_expr={"panda_finger_joint.*": 0.0},
        ),
    )
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = 16  # render step every 16 sim steps: 0.002*16 = 0.032 -> 32 Hz
        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # disable observation concatenation REQUIRED for observation based publishers
        self.observations.policy.concatenate_terms = False
