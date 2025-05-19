# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp.observations as obs
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    EventCfg,
    MySceneCfg,
)

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=obs.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=obs.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        joint_effort = ObsTerm(func=obs.joint_effort, noise=Unoise(n_min=-1.5, n_max=1.5))
        base_link_pose = ObsTerm(func=obs.link_pose, params={"link_name": "base"})

        def __post_init__(self):
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class AnymalCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    event: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [1.2, 0.0, 1.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]


@configclass
class AnymalDEnvCfg(AnymalCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scanner sensor
        self.scene.height_scanner = None
        # make a smaller scene for play
        self.scene.env_spacing = 2.5
        # disable observation corruptions
        self.observations.policy.enable_corruption = True
        # disable observation concatenation REQUIRED for observation based publishers
        self.observations.policy.concatenate_terms = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # disable events for play
        self.events.push_robot = None
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # physics sim rate (sec)
        self.sim.dt = 0.002  # 0.002 sec -> 500 Hz
        # control decimation
        self.decimation = 4  # env step every 4 sim steps: 0.002*4 = 0.008 -> 125 Hz
        # rendering substeps
        self.sim.render_interval = 16  # render step every 16 sim steps: 0.002*16 = 0.032 -> 32 Hz
