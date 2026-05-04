# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab_tasks.utils import preset

##
# Pre-defined configs
##
from isaaclab_assets.robots.cassie import CASSIE_CFG  # isort: skip


@configclass
class CassieRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe"),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_abduction_.*", "hip_rotation_.*"])},
    )
    joint_deviation_toes = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["toe_joint_.*"])},
    )
    # penalize toe joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="toe_joint_.*")},
    )


@configclass
class CassieRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Cassie rough environment configuration."""

    rewards: CassieRewardsCfg = CassieRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # biped yaw control is harder than quadruped — relax the per-episode-mean yaw
        # threshold to 0.8 rad/s (defaults work for quadrupeds).
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        # scene
        self.scene.robot = CASSIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Cassie Newton-only armature for biped stability on rough terrain; PhysX unchanged
        self.scene.robot.actuators["legs"].armature = preset(default=0.0, newton=0.02)

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"

        # Cassie uses "pelvis" as base body. Override the shared symmetric
        # (1/1.25, 1.25) log-uniform scale with asymmetric (1.0, 1.25) —
        # lighter-than-nominal pelvis destabilizes Cassie's closed-loop
        # Achilles coupling + hip PD response, so only heavier perturbations
        # are safe. Symmetric ±25% regressed reward 40% vs disabled;
        # (1.0, 1.25) recovers to 90% of baseline.
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.25)
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ".*pelvis"
        # Cassie has precise initial pose — don't scale joint defaults randomly on reset
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # actions
        self.actions.joint_pos.scale = 0.5

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*pelvis"]

        # rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight *= 1.5
        self.rewards.dof_acc_l2.weight *= 1.5


@configclass
class CassieRoughEnvCfg_PLAY(CassieRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
