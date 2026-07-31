# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka surface deformable lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .franka_soft_env_cfg import (
    FRANKA_CAMERA_CFG,
    FrankaCameraObservationsCfg,
    FrankaSoftEnvCfg,
    _FrankaSoftSceneCfg,
)
from .franka_soft_env_cfg import (
    EventCfg as FrankaSoftEventCfg,
)

##
# Scene definition
##

ROBOT_SHAPE_MATERIAL_MU = 100.0
"""Franka collision-shape friction coefficient [dimensionless] used for Newton cloth contact."""

ROBOT_SHAPE_MATERIAL_BODY_NAMES = ".*"
"""Franka body-name regex receiving :data:`ROBOT_SHAPE_MATERIAL_MU`."""


@configclass
class PhysicsCfg(PresetCfg):
    # Newton physics: MJWarp rigid + VBD soft, coupled through lagged proxies
    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                    ),
                    # the cube is a rigid body, so it must be owned by the rigid entry
                    bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Cube"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=1024),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_.*/Robot/panda_hand",
                        r"/World/envs/env_.*/Robot/panda_(left|right)finger",
                        r"/World/envs/env_.*/Cube",
                    ],
                    # detect contact every substep so the gripper stops at the cloth surface
                    collide_interval=1,
                )
            ],
            iterations=1,
        ),
        num_substeps=2,
    )

    ovphysx: OvPhysxCfg = OvPhysxCfg()

    default = newton_mjwarp_vbd_proxy


@configclass
class DeformableCfg(PresetCfg):
    """Preset config for the deformable object, matching the Newton example."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(30, 30),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=50.0,
                particle_radius=0.005,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=2.0,
                edge_kd=1e-3,
            ),
        ),
    )

    ovphysx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(30, 30),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
                density=50.0,
                youngs_modulus=2000.0,
                poissons_ratio=0.25,
                surface_thickness=0.005,
                surface_stretch_stiffness=0.8,
                surface_shear_stiffness=0.7,
                surface_bend_stiffness=0.6,
                elasticity_damping=0.03,
                bend_damping=0.04,
            ),
        ),
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka surface deformable environment."""

    deformable: DeformableCfg = DeformableCfg()

    # Collidable cube the cloth drapes onto (sits on the table top at z = 0). Kinematic so the
    # reset event can move it under the randomized cloth without it being simulated.
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.01, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # increase franka gripper stiffness
        self.robot.actuators["panda_hand"].effort_limit_sim = 500.0
        self.robot.actuators["panda_hand"].stiffness = 2000.0
        self.robot.actuators["panda_hand"].damping = 100.0


@configclass
class FrankaClothScenePresetCfg(PresetCfg):
    """Preset config for the Franka surface deformable scene."""

    newton_mjwarp_vbd_proxy: FrankaClothSceneCfg = FrankaClothSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )

    ovphysx: FrankaClothSceneCfg = FrankaClothSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class CurriculumCfg:
    """Ramp the action-rate penalty once the policy has learned to lift (matches rigid recipe)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 50000}
    )

    # Since we use 24 steps per env, 20000 steps correspond to 20000/24 = 833.33 learning iterations
    gravity = CurrTerm(
        func=mdp.modify_gravity_linear,
        params={"start_gravity_z": -1.0, "end_gravity_z": -9.81, "start_step": 0, "end_step": 20000},
    )


@configclass
class FrankaClothEventCfg(FrankaSoftEventCfg):
    """Reset and startup events for the Franka cloth environment."""

    # Replaces the base term so the cube follows the randomized cloth position.
    reset_deformable = EventTerm(
        func=mdp.reset_deformable_over_support,
        mode="reset",
        params={
            "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "support_offset_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            "asset_cfg": SceneEntityCfg("deformable"),
            "support_cfg": SceneEntityCfg("cube"),
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_SHAPE_MATERIAL_BODY_NAMES),
            "static_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
            "dynamic_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


def _make_ovphysx_event_cfg() -> FrankaClothEventCfg:
    """Create cloth events that select all robot shapes on OvPhysX."""
    cfg = FrankaClothEventCfg()
    cfg.robot_physics_material.params["asset_cfg"] = SceneEntityCfg("robot")
    return cfg


@configclass
class EventPresetCfg(PresetCfg):
    """Preset config for Franka cloth startup and reset events."""

    newton_mjwarp_vbd_proxy: FrankaClothEventCfg = FrankaClothEventCfg()
    ovphysx: FrankaClothEventCfg = _make_ovphysx_event_cfg()

    default = newton_mjwarp_vbd_proxy


##
# Environment configuration
##


@configclass
class FrankaClothEnvCfg(FrankaSoftEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a surface deformable."""

    scene: FrankaClothScenePresetCfg = FrankaClothScenePresetCfg()
    events: EventPresetCfg = EventPresetCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # override the soft-beam physics with the cloth presets
        self.sim.physics = PhysicsCfg()


@configclass
class FrankaClothCameraEnvCfg(FrankaClothEnvCfg):
    """Visual Franka surface-deformable lifting environment."""

    scene: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
