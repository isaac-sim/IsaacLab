# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for the G1 29-DoF granular locomotion task.

The proprioceptive groups match the soft-contact task, and so does the privileged group: the
contact terms are served by the reaction wrench that the coupler feeds back from the MPM entry
to the proxy feet, and the terrain-material term keeps the soft-contact layout as a placeholder.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_contrib.mdp import mirror_identity, mirror_quat, mirror_vec3

import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as g1_mdp
from .. import mdp as mpm_mdp
from ..mdp import symmetry
from .action_cfg import ACTIVE_JOINT


@configclass
class PolicyCfg(ObsGroup):
    """Observations for the actor."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        mirror=mirror_vec3,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Noise-free proprioception for the critic."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        scale=0.25,
    )
    base_quat = ObsTerm(
        func=mdp.root_quat_w,
        mirror=mirror_quat,
        params={"make_quat_unique": True},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PolicyHistoryCfg(PolicyCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class CriticHistoryCfg(CriticCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class CommandCfg(ObsGroup):
    """Command observations for the actor and critic."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        mirror=symmetry.mirror_velocity_heading,
        params={"command_name": "base_velocity"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedObsCfg(ObsGroup):
    """Granular-terrain information available to the critic only."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        mirror=symmetry.mirror_foot_scalars,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_contact = ObsTerm(func=mpm_mdp.foot_contact, mirror=symmetry.mirror_foot_scalars)
    foot_contact_force = ObsTerm(
        func=mpm_mdp.foot_contact_force,
        mirror=symmetry.mirror_foot_forces,
        params={"force_filter_threshold": 5.0},
    )
    foot_air_time = ObsTerm(func=mpm_mdp.foot_air_time, mirror=symmetry.mirror_foot_scalars)
    terrain_material_parameters = ObsTerm(func=mpm_mdp.terrain_material_parameters, mirror=mirror_identity)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedHistoryCfg(PrivilegedObsCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class LoggingObsCfg(ObsGroup):
    """Quantities recorded for analysis; not consumed by the learning algorithm."""

    base_pos = ObsTerm(func=mdp.root_pos_w, mirror=mirror_vec3)
    base_quat = ObsTerm(func=mdp.root_quat_w, mirror=mirror_quat)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True})
    commands = ObsTerm(
        func=mdp.generated_commands,
        mirror=symmetry.mirror_velocity_heading,
        params={"command_name": "base_velocity"},
    )
    contact_forces = ObsTerm(
        func=mpm_mdp.foot_contact_force_raw,
        mirror=symmetry.mirror_foot_forces,
        params={"force_filter_threshold": 5.0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    command: CommandCfg = CommandCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()
    # logging: LoggingObsCfg = LoggingObsCfg()
