# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 29-DoF G1 on rough terrain with WBC-AGILE's reward and termination set.

WBC-AGILE trains the asset Unitree ships today -- collision set untouched, all 24 collider links --
and it walks. The stock Isaac Lab velocity task does not: its only termination is torso contact,
which never fires, so any robot with knee and pelvis colliders learns a crouch that survives the
whole episode. This config keeps the shipped asset exactly as it is and swaps in WBC-AGILE's
rewards and terminations instead, to see how much of their result is the reward set rather than the
robot.

Two of their terms are the ones our own single-variable runs point at:

* ``illegal_contacts`` terminates on ground contact at the torso, pelvis, hips or knees. That is
  what turns the extra colliders from a free crutch into a cost, and is the most likely reason the
  same asset trains here and not in the stock task.
* ``hip_pos_pen`` penalises hip roll and yaw deviation with **L2 at weight -1.0**, against the
  stock task's L1 at -0.1. Our foot-plate run reached ``success_rate`` 0.906 while its
  ``joint_deviation_hip`` grew to -0.228, twelve times the old asset's -0.018: with nothing else
  constraining stance width, the policy buys yaw authority by splaying its legs. This term prices
  that properly.

**What is ported and what is not.** Every reward and termination term is carried over at
WBC-AGILE's own weights. Their command term is not: ``UniformVelocityBaseHeightCommand`` appends a
commanded pelvis height to the velocity command, and six of their rewards read it. Porting it is
unnecessary here because *in their own configuration that command is a constant* --
``random_height_during_walking=False`` and ``base_height=(0.72, 0.72)`` -- so ``target_height`` is
always ``DEFAULT_PELVIS_HEIGHT``. The two height rewards below therefore measure against a constant
0.72 through the height scanner this task already has, which is equivalent rather than approximate,
and the four ``*_if_null_cmd`` terms only ever ask whether the velocity command is zero, which the
stock ``UniformVelocityCommand`` answers through ``rel_standing_envs``.

One consequence worth knowing before reading the numbers: this task's ``rel_standing_envs`` is
0.02 against WBC-AGILE's 0.20, so the four null-command terms shape a tenth as many environments
here as they do there.
"""

from __future__ import annotations

import math
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .rough_29dof_env_cfg import G129DofRoughEnvCfg

##
# Constants
##

_LEG_JOINT_NAMES = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
"""WBC-AGILE's ``unitree_g1.LEG_JOINT_NAMES``. Their regularisation penalties cover the legs only."""

_TARGET_PELVIS_HEIGHT = 0.72
"""WBC-AGILE's ``DEFAULT_PELVIS_HEIGHT`` [m], and the constant value of their height command."""


def _is_null_command(env, command_name: str) -> torch.Tensor:
    """True where the velocity command is exactly zero, i.e. the environment is told to stand."""
    return (env.command_manager.get_term(command_name).command[:, :3] == 0).all(dim=1)


def _terrain_relative_base_height(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root height above the ground beneath it [m].

    WBC-AGILE's command term derives this from its own height sensor; on rough terrain the world-frame
    root height is not the same quantity, because the ground moves under the robot. The median of the
    ray hits rejects the outliers a grid scanner picks up at terrain edges, and ``nan_to_num`` covers
    rays that miss entirely.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    return asset.data.root_pos_w.torch[:, 2] - ground


##
# WBC-AGILE reward terms not present in Isaac Lab's mdp
##


def track_base_height_exp_smooth(
    env,
    std: float,
    target_height: float = _TARGET_PELVIS_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Reward holding the pelvis at the commanded height."""
    error = torch.square(_terrain_relative_base_height(env, asset_cfg, sensor_cfg) - target_height)
    return torch.exp(-error / std**2)


def base_height_in_threshold(
    env,
    threshold: float,
    target_height: float = _TARGET_PELVIS_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Reward being within ``threshold`` [m] of the commanded height."""
    error = torch.abs(_terrain_relative_base_height(env, asset_cfg, sensor_cfg) - target_height)
    return (error < threshold).float()


def stand_with_both_feet_if_null_cmd(
    env,
    threshold: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: ARG001
) -> torch.Tensor:
    """Under a null command, reward even weight on both feet and punish standing on one."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    feet_z_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].abs()
    mean_force = feet_z_forces.mean(dim=1)
    reward = 1.0 - torch.abs(mean_force.unsqueeze(1) - feet_z_forces).mean(dim=1) / (mean_force + 1e-6)
    reward[~(feet_z_forces > threshold).all(dim=1)] = -1.0
    reward[~_is_null_command(env, command_name)] = 0.0
    return reward


def equal_foot_force_if_null_cmd(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Under a null command, reward an even vertical force split between the feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    feet_z_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].abs()
    total_force = feet_z_forces.sum(dim=1, keepdim=True)
    distribution = feet_z_forces / (total_force + 1e-6)
    ideal = 1.0 / feet_z_forces.shape[1]
    reward = 1.0 - torch.abs(distribution - ideal).mean(dim=1) / (2 * ideal)
    return reward * _is_null_command(env, command_name).float()


def flat_orientation_if_null_cmd(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Under a null command, penalise a non-level base."""
    asset = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)
    return torch.where(_is_null_command(env, command_name), error, 0.0)


def relax_if_null_cmd(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Under a null command, penalise squared applied torque."""
    asset = env.scene[asset_cfg.name]
    torques = torch.sum(torch.square(asset.data.applied_torque.torch[:, asset_cfg.joint_ids]), dim=1)
    return torch.where(_is_null_command(env, command_name), torques, 0.0)


def no_undesired_base_velocity_exp(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Reward the absence of vertical and roll/pitch base motion, which is never commanded."""
    asset = env.scene[asset_cfg.name]
    lin_vel_z = torch.square(asset.data.root_lin_vel_b.torch[:, 2])
    ang_vel_xy = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)
    return torch.exp(-(lin_vel_z + ang_vel_xy) / std**2)


def jumping(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise having no foot on the ground."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=2)
    return (feet_forces < threshold).all(dim=1).float()


def impact_velocity(
    env,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise how fast a body is moving at the moment it makes contact.

    WBC-AGILE caches the sensor-body-to-articulation-body mapping in a ``ManagerTermBase``. Here the
    two selections name the same bodies (``.*ankle_roll_link``) and Isaac Lab resolves both against
    the same articulation, so the ids line up without a mapping step.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history.torch
    in_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > force_threshold
    body_velocities = torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids], dim=-1)
    return torch.where(in_contact, body_velocities, 0.0).sum(dim=1)


def joint_deviation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared deviation of the selected joints from their default angles."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


##
# WBC-AGILE termination terms not present in Isaac Lab's mdp
##


def illegal_ground_contact(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Terminate on a hard contact at a body that should never touch the ground.

    The height gate is what makes this usable on a robot that carries knee and hand colliders: a
    contact only counts as a fall when the reference body is also low. Without it, brushing a
    terrain edge at full height would end the episode.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history.torch
    asset_height = torch.min(env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids, 2], dim=1)[0]
    in_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )
    return in_contact & (asset_height < min_height)


##
# Configuration
##


@configclass
class G129DofRoughWbcRewardsCfg:
    """WBC-AGILE's reward set, term for term and weight for weight."""

    # -- task
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=10.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_base_height_exp_smooth = RewTerm(func=track_base_height_exp_smooth, weight=2.0, params={"std": 0.05})
    base_height_in_threshold = RewTerm(func=base_height_in_threshold, weight=0.1, params={"threshold": 0.005})
    stand_with_both_feet_if_null_cmd = RewTerm(
        func=stand_with_both_feet_if_null_cmd,
        weight=10.0,
        params={
            "threshold": 1.0,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )
    equal_foot_force_if_null_cmd = RewTerm(
        func=equal_foot_force_if_null_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )

    # -- penalties
    dof_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    dof_vel_limits = RewTerm(
        func=base_mdp.joint_vel_limits,
        weight=-0.1,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    dof_vel_l2 = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    torque_limits = RewTerm(
        func=base_mdp.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    no_undesired_base_velocity_exp = RewTerm(
        func=no_undesired_base_velocity_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1},
    )
    dof_torques_l2 = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    dof_acc_l2 = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    jumping_penalty = RewTerm(
        func=jumping,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"), "threshold": 0.1},
    )
    impact_velocity = RewTerm(
        func=impact_velocity,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "force_threshold": 10.0,
        },
    )
    flat_orientation = RewTerm(
        func=base_mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    flat_orientation_if_null_cmd = RewTerm(
        func=flat_orientation_if_null_cmd,
        weight=-4.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    relax_if_null_cmd = RewTerm(
        func=relax_if_null_cmd,
        weight=-1.0e-3,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)},
    )
    # The term our foot-plate run says the stock task is missing: L2 at -1.0, against -0.1 L1.
    hip_pos_pen = RewTerm(
        func=joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )


@configclass
class G129DofRoughWbcTerminationsCfg:
    """WBC-AGILE's termination set."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_orientation = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "limit_angle": math.radians(40.0)},
    )
    # The term that turns the shipped asset's extra colliders from a free crutch into a cost.
    illegal_contacts = DoneTerm(
        func=illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso_link", "pelvis", ".*_hip_.*_link", ".*_knee_link"],
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "threshold": 50.0,
            "min_height": 0.4,
        },
    )


@configclass
class G129DofRoughWbcEnvCfg(G129DofRoughEnvCfg):
    """The shipped 29-DoF asset, collision set untouched, under WBC-AGILE's rewards and terminations."""

    def __post_init__(self):
        # The parent chain edits the stock reward set in place -- zeroing ``lin_vel_z_l2``,
        # retargeting joint names, installing the base-height termination. Those terms do not exist
        # in WBC-AGILE's set, so the swap has to happen after that chain has run, not before it.
        super().__post_init__()
        self.rewards = G129DofRoughWbcRewardsCfg()
        self.terminations = G129DofRoughWbcTerminationsCfg()
