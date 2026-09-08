# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Allegro Hand Direct reorientation task."""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    ALLEGRO_HAND_ROBOT_CFG,
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
)
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import ReorientManagerEnvBaseCfg, ReorientSceneBaseCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


@configclass
class AllegroHandManagerSceneCfg(ReorientSceneBaseCfg):
    """The shared scene, holding the Allegro hand and its in-hand cube."""

    robot: ArticulationCfg = ALLEGRO_HAND_ROBOT_CFG
    object: RigidObjectCfg = CUBE_CFG


@configclass
class AllegroHandResetEventCfg:
    """Only the per-episode state reset, with no domain randomization."""

    reset_object = EventTerm(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            # the Direct task jitters the drop position and samples a random orientation
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},  # [m]
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_hand = EventTerm(
        func=mdp.reset_reorient_hand,
        mode="reset",
        params={
            "joint_position_noise": 0.2,  # [rad]
            "joint_velocity_noise": 0.0,  # [rad/s]
        },
    )


@configclass
class AllegroHandRandomizationEventCfg(AllegroHandResetEventCfg):
    """Randomization terms plus the Direct task's reset distribution."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
            "damping_distribution_params": (0.75, 1.5),  # default: 0.1
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.4, 1.6),
            "operation": "scale",
        },
    )


@configclass
class AllegroHandEventPresetCfg(PresetCfg):
    """``presets=randomized`` adds the domain-randomization terms to the episode reset."""

    randomized = AllegroHandRandomizationEventCfg()
    reset_only = AllegroHandResetEventCfg()
    default = reset_only


@configclass
class AllegroHandManagerEnvCfg(ReorientManagerEnvBaseCfg):
    """Manager-based Allegro Hand task with Direct-compatible semantics."""

    fingertip_body_names = ALLEGRO_FINGERTIP_BODY_NAMES
    actuated_joint_names = ALLEGRO_ACTUATED_JOINT_NAMES
    goal_orientation_threshold = 0.2
    goal_marker_cfg = GOAL_OBJECT_CFG
    decimation = 4

    scene: AllegroHandManagerSceneCfg = AllegroHandManagerSceneCfg()
    # ``presets=randomized`` adds the domain-randomization terms
    events: AllegroHandEventPresetCfg = AllegroHandEventPresetCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
