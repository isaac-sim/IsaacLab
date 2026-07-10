# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand identity shared by the Direct and manager-based reorientation tasks.

Asset and marker configurations, joint/body name lists, backend physics and
domain-randomization presets, and the sim mixins. No task tunables: reward
scales and thresholds live inline in the workflow configuration files.
"""

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import MENAGERIE_SHADOW_FINGERS_ACTUATOR_BASE, SHADOW_HAND_CFG


@configclass
class NewtonEventCfg:
    """Event randomization config for the Newton physics backend.

    Includes joint-parameter, mass, and gravity randomization.
    Material and tendon randomization are omitted: Newton does not expose
    per-body friction-material buckets or fixed-tendon APIs.
    """

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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


@configclass
class PhysxEventCfg:
    # -- robot
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

    # -- object
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
class ShadowHandEventCfg(PresetCfg):
    physx = PhysxEventCfg()
    newton_mjwarp = NewtonEventCfg()
    ovphysx = physx  # OvPhysX is PhysX-based; reuse the PhysX randomization terms
    default = newton_mjwarp
    newton_kamino = newton_mjwarp


@configclass
class ShadowHandRobotCfg(PresetCfg):
    physx = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        )
    )
    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # newton requires implicitactuators be specified in usd and there's a bug with physx tendons
            usd_path=f"{MUJOCO_MENAGERIE_DIR}/shadow_hand/right_hand/right_hand.usda",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force", ensure_drives_exist=True),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(damping=0.1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # WARNING(Octi): Newton's import_usd.py bakes the USD body xformOp rotation into
            # joint_X_p for the root fixed joint, which cancels with the matching localPose1
            # rotation in joint_X_c during FK (joint_X_p * inv(joint_X_c) ≈ identity). This
            # discards the root body's native USD orientation, so we must re-apply it here as a
            # spawn rotation. PhysX or USD does not have this issue. Remove once Newton fixes root joint
            # transform handling in import_usd.py.
            rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": MENAGERIE_SHADOW_FINGERS_ACTUATOR_BASE.replace(friction=1e-2, armature=2e-3),
        },
        soft_joint_pos_limit_factor=1.0,
    )
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
    newton_kamino = newton_mjwarp


@configclass
class ObjectCfg(PresetCfg):
    physx = RigidObjectCfg(
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
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            semantic_tags=[("class", "cube")],
            scale=(0.9, 0.9, 0.9),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.36, 0.535), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
        articulation_root_prim_path="",
    )
    ovphysx = physx  # OvPhysX is PhysX-based; use the rigid-body cube, not Newton's articulation
    default = newton_mjwarp
    newton_kamino = newton_mjwarp


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            njmax=400,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.02),
    )
    ovphysx = OvPhysxCfg()
    default = newton_mjwarp
    newton_kamino = NewtonCfg(solver_cfg=KaminoSolverCfg(max_contacts_per_world=128))


# Scene pieces shared verbatim by the manager-based variants.
ROBOT_CFG = ShadowHandRobotCfg()
OBJECT_CFG = ObjectCfg()
GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
        )
    },
)


# Per-step gaussian noise + reset-sampled bias, shared verbatim by the manager-based variant.
OPENAI_ACTION_NOISE_CFG = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
)
OPENAI_OBSERVATION_NOISE_CFG = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
)
