# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand identity shared by the Direct and manager-based reorientation tasks.

Asset and marker configurations, joint/body name lists, backend physics and
domain-randomization presets, and the sim mixins. No task tunables: reward
scales and thresholds live inline in the workflow configuration files.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as reorient_mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import (
    SHADOW_HAND_CFG,
    SHADOW_HAND_NEWTON_CFG,
)


@configclass
class ShadowHandRandomizationEventCfg:
    """Randomization of the hand and the object, applied on every physics backend."""

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": False,
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )


@configclass
class ShadowHandManagerResetEventCfg:
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
        func=reorient_mdp.reset_reorient_hand,
        mode="reset",
        params={
            "joint_position_noise": 0.2,  # [rad]
            "joint_velocity_noise": 0.0,  # [rad/s]
        },
    )


@configclass
class ShadowHandManagerEventCfg(ShadowHandRandomizationEventCfg, ShadowHandManagerResetEventCfg):
    """Randomization plus the state reset the manager tasks apply on every episode."""


@configclass
class ShadowHandManagerEventPresetCfg(PresetCfg):
    """``presets=randomized`` adds the domain-randomization terms to the episode reset."""

    randomized = ShadowHandManagerEventCfg()
    default = ShadowHandManagerResetEventCfg()


@configclass
class ShadowHandRobotCfg(PresetCfg):
    physx = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        )
    )
    isaacsim_physx = physx
    # Newton robot lives in the asset (see isaaclab_assets.robots.shadow_hand); reorient
    # uses its default gains. The handover task consumes the same asset cfg and overrides
    # only the finger gains.
    newton_mjwarp = SHADOW_HAND_NEWTON_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    ovphysx = SHADOW_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # OVPhysX does not expose the fixed-tendon runtime API, so spawn without tendon overrides.
        spawn=SHADOW_HAND_CFG.spawn.replace(fixed_tendons_props=None),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
    )
    default = newton_mjwarp


CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            mesh_collision_property=sim_utils.MeshCollisionPropertiesCfg(mesh_approximation_name="convexHull")
        ),
        semantic_tags=[("class", "cube")],
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(0.0, 0.0, 0.0, 1.0)),
)
"""In-hand cube for the Shadow Hand reorientation tasks."""


@configclass
class PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            njmax=200,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
        ),
        num_substeps=2,
    )
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = newton_mjwarp


GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        )
    },
)
