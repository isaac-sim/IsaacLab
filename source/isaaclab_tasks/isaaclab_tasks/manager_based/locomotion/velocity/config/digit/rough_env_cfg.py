# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_assets.robots.agility import ARM_JOINT_NAMES, DIGIT_V4_CFG, LEG_JOINT_NAMES

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class DigitRewards:
    termination_penalty = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-100.0,
    )
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "threshold": 0.8,
            "command_name": "base_velocity",
        },
    )
    feet_slide = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_leg_toe_roll"),
        },
    )
    dof_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
    )
    dof_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
    )
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.008,
    )
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-2.5,
    )
    stand_still = RewardTermCfg(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.1,
    )
    no_jumps = RewardTermCfg(
        func=mdp.desired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_toe_roll"])},
    )
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_toe_roll", ".*_leg_toe_pitch", ".*_tarsus"])},
    )
    joint_deviation_hip_roll = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_roll")},
    )
    joint_deviation_hip_yaw = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_yaw")},
    )
    joint_deviation_knee = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_tarsus")},
    )
    joint_deviation_feet = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_a", ".*_toe_b"])},
    )
    joint_deviation_arms = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_arm_.*"),
        },
    )

    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_rod", ".*_tarsus"]),
            "threshold": 1.0,
        },
    )


@configclass
class DigitObservations:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObservationTermCfg(func=mdp.last_action)
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups:
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_base"]),
            "threshold": 1.0,
        },
    )
    base_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class DigitRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitRewards = DigitRewards()
    observations: DigitObservations = DigitObservations()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.dt = 0.005

        # Scene
        self.scene.robot = DIGIT_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_base"
        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Events:
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_base")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_base")
        # Don't randomize the initial joint positions because we have closed loops.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # remove COM randomization
        self.events.base_com = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)


@configclass
class DigitRoughEnvCfg_PLAY(DigitRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Spawn the robot randomly in the grid (instead of their terrain levels).
        self.scene.terrain.max_init_terrain_level = None
        # Reduce the number of terrains to save memory.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
