# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Allegro Hand Direct reorientation task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    ALLEGRO_HAND_ROBOT_CFG,
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    PhysicsCfg,
)
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import ManagerEnvCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


@configclass
class AllegroHandManagerSceneCfg(InteractiveSceneCfg):
    """Shared reorientation scene with the Allegro hand and a ground plane."""

    num_envs = 8192
    env_spacing = 0.75

    robot: ArticulationCfg = ALLEGRO_HAND_ROBOT_CFG
    object: RigidObjectCfg = CUBE_CFG
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ResetEventCfg:
    """Only the per-episode state reset, with no domain randomization."""

    reset_state = EventTerm(
        func=mdp.reset_reorient_state,
        mode="reset",
        params={
            "position_noise": 0.01,  # [m]
            "joint_position_noise": 0.2,  # [rad]
            "joint_velocity_noise": 0.0,  # [rad/s]
        },
    )


@configclass
class EventCfg(ResetEventCfg):
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
class EventPresetCfg(PresetCfg):
    """``presets=randomized`` adds the domain-randomization terms to the episode reset."""

    randomized = EventCfg()
    default = ResetEventCfg()


@configclass
class AllegroHandManagerEnvCfg(ManagerEnvCfg):
    """Manager-based Allegro Hand task with Direct-compatible semantics."""

    fingertip_body_names = ALLEGRO_FINGERTIP_BODY_NAMES
    actuated_joint_names = ALLEGRO_ACTUATED_JOINT_NAMES
    goal_orientation_threshold = 0.2
    goal_marker_cfg = GOAL_OBJECT_CFG
    decimation = 4

    scene: AllegroHandManagerSceneCfg = AllegroHandManagerSceneCfg()
    # ``presets=randomized`` adds the domain-randomization terms
    events: EventPresetCfg = EventPresetCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
