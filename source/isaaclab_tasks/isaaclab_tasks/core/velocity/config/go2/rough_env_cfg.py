# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# UNITREE_GO2_CFG.init_state.pos is (0, 0, 0.4) -- the asset's own authored standing height.
_GO2_STANDING_HEIGHT = 0.4


@configclass
class UnitreeGo2Rewards(RewardsCfg):
    """Adds a base-height term on top of the shared velocity rewards.

    The stock velocity task has no term constraining base height. Go2 carries colliders on its
    legs (not just the feet), so a policy can rest its weight on the legs and collect the full
    episode-length alive bonus from a permanent crouch -- torso-contact termination never fires
    and flat-orientation reward can't see it, since the torso stays level while the robot sinks.
    """

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={"target_height": _GO2_STANDING_HEIGHT},
    )


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeGo2Rewards = UnitreeGo2Rewards()

    def __post_init__(self):
        super().__post_init__()

        # simulation
        # execute the DC motor actuators through the backend-native path
        self.sim.use_newton_actuators = True
        # scene
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["boxes"].grid_height_range = (0.025, 0.1)
        terrains["random_rough"].noise_range = (0.01, 0.06)
        terrains["random_rough"].noise_step = 0.01
        terrains["pyramid_stairs"].step_height_range = (0.025, 0.12)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.12)
        # actions
        self.actions.joint_pos.scale = 0.25
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # base_height_l2: rough terrain, adjust target height using the height-scanner rays.
        self.rewards.base_height_l2.weight = -30.0
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        # action_rate_l2: halved from the shared default of -0.01. The stronger penalty locked
        # one hind foot into a low-amplitude, dragging gait on flat terrain with Newton.
        self.rewards.action_rate_l2.weight = -0.005
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
