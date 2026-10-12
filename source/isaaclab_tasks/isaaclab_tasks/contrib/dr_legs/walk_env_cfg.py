# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking walk environment for the Disney DR Legs closed-loop biped.

Extends the hold-pose task with a base-velocity command, a gait-phase clock, a foot
contact sensor, and gait / contact / foot-clearance rewards.
"""

import math

from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.dr_legs.mdp as mdp
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.dr_legs import DR_LEGS_ACTUATED_JOINTS

from .hold_pose_env_cfg import ActionsCfg, DrLegsHoldPoseEnvCfg, HoldPoseSceneCfg, ObservationsCfg, _dr_legs_robot_cfg

##
# Shared constants
##

# Full left+right gait cycle [s]. Shared by the gait-phase observation and the rewards.
_NEWTON_GAIT_PERIOD = 0.6
_PHYSX_GAIT_PERIOD = 1.0

# The 12 servo-driven joints (used by the implicit-PD effort proxy).
_ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=DR_LEGS_ACTUATED_JOINTS, preserve_order=True)

# Foot bodies in canonical [left, right] order for the kinematics and contact rewards.
_FOOT_ASSET_CFG = SceneEntityCfg("robot", body_names=["foot_l", "foot_r"], preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=["foot_l", "foot_r"], preserve_order=True)


@configclass
class DrLegsContactSensorCfg(PresetCfg):
    """Backend-specific foot contact sensor configuration."""

    default = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/foot_[^/]*",
        history_length=3,
        track_air_time=True,
    )
    newton_kamino = default
    physx = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/foot_[^/]*",
        history_length=3,
        track_air_time=True,
    )
    isaacsim_physx = physx


@configclass
class WalkSceneCfg(HoldPoseSceneCfg):
    robot = _dr_legs_robot_cfg()
    robot.actuators["driven_joints"].stiffness = preset(default=15.0, newton_kamino=15.0, physx=5.0, isaacsim_physx=5.0)
    robot.actuators["driven_joints"].damping = preset(default=0.6, newton_kamino=0.6, physx=0.2, isaacsim_physx=0.2)
    robot.actuators["driven_joints"].armature = preset(
        default=0.01, newton_kamino=0.01, physx=None, isaacsim_physx=None
    )
    contact_forces = DrLegsContactSensorCfg()


def _newton_walk_actions_cfg() -> ActionsCfg:
    """Return the Newton walking action profile."""
    cfg = ActionsCfg()
    cfg.joint_pos.scale = 0.4
    return cfg


def _physx_walk_actions_cfg() -> ActionsCfg:
    """Return the PhysX walking action profile."""
    cfg = ActionsCfg()
    cfg.joint_pos.scale = 0.1
    return cfg


@configclass
class WalkActionsCfg(PresetCfg):
    """Backend-specific DR Legs walking action configurations."""

    default: ActionsCfg = _newton_walk_actions_cfg()
    newton_kamino: ActionsCfg = default
    physx: ActionsCfg = _physx_walk_actions_cfg()
    isaacsim_physx: ActionsCfg = physx


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
        ),
    )


def _newton_commands_cfg() -> CommandsCfg:
    """Return commands without heading conversion for Newton walking."""
    cfg = CommandsCfg()
    cfg.base_velocity.heading_command = False
    cfg.base_velocity.ranges.heading = None
    return cfg


@configclass
class DrLegsCommandsCfg(PresetCfg):
    """Backend-specific DR Legs walking command configurations."""

    default: CommandsCfg = _newton_commands_cfg()
    newton_kamino: CommandsCfg = default
    physx: CommandsCfg = CommandsCfg()
    isaacsim_physx: CommandsCfg = physx


@configclass
class WalkObservationsCfg(ObservationsCfg):
    @configclass
    class WalkPolicyCfg(ObservationsCfg.PolicyCfg):
        base_lin_vel: ObsTerm | None = None
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": _PHYSX_GAIT_PERIOD})
        joint_vel: ObsTerm | None = None

    policy: WalkPolicyCfg = WalkPolicyCfg()


def _newton_walk_observations_cfg() -> WalkObservationsCfg:
    """Return Newton walking observations with velocity feedback and a richer gait clock."""
    cfg = WalkObservationsCfg()
    cfg.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    cfg.policy.gait_phase.params = {"period": _NEWTON_GAIT_PERIOD, "num_harmonics": 2}
    cfg.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ACTUATED_JOINT_CFG}, scale=0.2)
    return cfg


@configclass
class DrLegsWalkObservationsCfg(PresetCfg):
    """Backend-specific DR Legs walking observation configurations."""

    default: WalkObservationsCfg = _newton_walk_observations_cfg()
    newton_kamino: WalkObservationsCfg = default
    physx: WalkObservationsCfg = WalkObservationsCfg()
    isaacsim_physx: WalkObservationsCfg = physx


@configclass
class WalkRewardsCfg:
    """Gait / contact / feet reward suite for velocity-tracking walking."""

    # -- positive (task / gait)
    termination_penalty: RewTerm | None = None
    survival = RewTerm(func=mdp.is_alive, weight=10.0)
    contact_matching = RewTerm(
        func=mdp.contact_matching,
        weight=10.0,
        params={"sensor_cfg": _FOOT_SENSOR_CFG, "threshold": 1.0, "gait_period": _PHYSX_GAIT_PERIOD},
    )
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.125)},
    )
    root_orientation = RewTerm(
        func=mdp.root_orientation_exp,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.2},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=5.0,
        params={
            "asset_cfg": _FOOT_ASSET_CFG,
            "target_height": 0.065,
            "sigma": 0.01,
            "gait_period": _PHYSX_GAIT_PERIOD,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    feet_parallel = RewTerm(
        func=mdp.feet_parallel,
        weight=2.0,
        params={"asset_cfg": _FOOT_ASSET_CFG, "sigma": 0.2},
    )
    feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=2.0,
        params={"asset_cfg": _FOOT_ASSET_CFG, "sigma": 0.2},
    )

    # -- negative (penalties)
    feet_touchdown_vel = RewTerm(
        func=mdp.feet_touchdown_vel,
        weight=-2.0,
        params={"asset_cfg": _FOOT_ASSET_CFG, "sensor_cfg": _FOOT_SENSOR_CFG, "threshold": 1.0},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    torques = RewTerm(
        func=mdp.joint_pd_command_l2,
        weight=-0.001,
        params={"asset_cfg": _ACTUATED_JOINT_CFG, "stiffness": 5.0, "damping": 0.2},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_2nd_derivative = RewTerm(func=mdp.ActionRate2L2, weight=-5.0e-5)

    # -- success metric (velocity tracking + gait-contact match)
    success_rate = RewTerm(
        func=mdp.walk_success_rate,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "gait_period": _PHYSX_GAIT_PERIOD,
            "contact_threshold": 1.0,
            "vel_xy_threshold": 0.15,
            "vel_yaw_threshold": 0.4,
            "contact_match_threshold": 0.7,
        },
    )


def _newton_walk_rewards_cfg() -> WalkRewardsCfg:
    """Return the Newton reward profile for stable gait and velocity tracking."""
    cfg = WalkRewardsCfg()
    cfg.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    cfg.survival.weight = 0.0
    cfg.contact_matching.params["gait_period"] = _NEWTON_GAIT_PERIOD
    cfg.track_lin_vel_xy.weight = 20.0
    cfg.track_lin_vel_xy.params["std"] = 0.2
    cfg.root_orientation.weight = 1.0
    cfg.foot_clearance.weight = 2.0
    cfg.foot_clearance.params["gait_period"] = _NEWTON_GAIT_PERIOD
    cfg.track_ang_vel_z.weight = 10.0
    cfg.track_ang_vel_z.params["std"] = 0.5
    cfg.feet_parallel.weight = 0.5
    cfg.feet_flat.weight = 0.5
    cfg.success_rate.params["gait_period"] = _NEWTON_GAIT_PERIOD
    return cfg


@configclass
class DrLegsWalkRewardsCfg(PresetCfg):
    """Backend-specific DR Legs walking reward configurations."""

    default: WalkRewardsCfg = _newton_walk_rewards_cfg()
    newton_kamino: WalkRewardsCfg = default
    physx: WalkRewardsCfg = WalkRewardsCfg()
    isaacsim_physx: WalkRewardsCfg = physx


@configclass
class DrLegsWalkEnvCfg(DrLegsHoldPoseEnvCfg):
    """DR Legs velocity-tracking walk environment."""

    scene: WalkSceneCfg = WalkSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: DrLegsWalkObservationsCfg = DrLegsWalkObservationsCfg()
    actions: WalkActionsCfg = WalkActionsCfg()
    commands: DrLegsCommandsCfg = DrLegsCommandsCfg()
    rewards: DrLegsWalkRewardsCfg = DrLegsWalkRewardsCfg()
