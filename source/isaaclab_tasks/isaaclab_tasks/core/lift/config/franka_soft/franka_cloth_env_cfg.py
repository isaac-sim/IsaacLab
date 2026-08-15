# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka surface deformable lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from .franka_soft_env_cfg import (
    FRANKA_CAMERA_CFG,
    FrankaCameraObservationsCfg,
    FrankaSoftEnvCfg,
    _FrankaSoftSceneCfg,
)
from .franka_soft_env_cfg import (
    EventCfg as FrankaSoftEventCfg,
)
from .franka_soft_env_cfg import (
    RewardsCfg as FrankaSoftRewardsCfg,
)

##
# Scene definition
##


@configclass
class PhysicsCfg(PresetCfg):
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
                    bodies=[r"/World/envs/env_[^/]+/Robot", r"/World/envs/env_[^/]+/Support(Neg|Pos)Y"],
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
                        r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
                        r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
                        r"/World/envs/env_[^/]+/Support(Neg|Pos)Y",
                    ],
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                )
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=8.0e3,
            soft_contact_kd=1.0e-2,
            soft_contact_mu=10.0,
        ),
        num_substeps=2,
    )

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_found_lost_pairs_capacity=2**22)

    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)

    default = newton_mjwarp_vbd_proxy


SUPPORT_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.1, 0.02, 0.15),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.01, dynamic_friction=0.01),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
)


@configclass
class DeformableCfg(PresetCfg):
    """Preset configurations for the cloth."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.102), rot=(0.70710678, 0.0, 0.0, 0.70710678)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(8, 8),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.002,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=0.5,
                edge_kd=1e-3,
            ),
        ),
    )

    physx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.102), rot=(0.70710678, 0.0, 0.0, 0.70710678)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(8, 8),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            collision_props=[PhysxCollisionCfg(rest_offset=0.002, contact_offset=0.01)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
                density=1000.0,
                surface_thickness=0.001,
                poissons_ratio=0.25,
                youngs_modulus=1e6,
                surface_bend_stiffness=1e6,
                elasticity_damping=1e-1,
                bend_damping=1e-1,
                static_friction=10.0,
                dynamic_friction=10.0,
            ),
        ),
    )
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka surface deformable environment."""

    deformable: DeformableCfg = DeformableCfg()

    support_neg_y: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupportNegY",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.02, 0.075)),
        spawn=SUPPORT_SPAWN_CFG,
    )
    support_pos_y: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupportPosY",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.02, 0.075)),
        spawn=SUPPORT_SPAWN_CFG,
    )


@configclass
class FrankaClothScenePresetCfg(PresetCfg):
    """Preset config for the Franka surface deformable scene."""

    newton_mjwarp_vbd_proxy: FrankaClothSceneCfg = FrankaClothSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )

    # PhysX does not support replicating physics for deformable objects
    physx: FrankaClothSceneCfg = FrankaClothSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=False)
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class FrankaClothCameraScenePresetCfg(PresetCfg):
    """Scene presets for visual Franka cloth lifting."""

    newton_mjwarp_vbd_proxy: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )
    physx: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=False)
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothEventCfg(FrankaSoftEventCfg):
    """Reset and startup events for the Franka cloth environment."""

    reset_deformable = EventTerm(
        func=mdp.reset_deformable_over_support,
        mode="reset",
        params={
            "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "clear_gap_range": (0.01, 0.03),
            "asset_cfg": SceneEntityCfg("deformable"),
            "support_cfg": (SceneEntityCfg("support_neg_y"), SceneEntityCfg("support_pos_y")),
        },
    )


##
# Environment configuration
##


@configclass
class FrankaClothRewardsCfg(FrankaSoftRewardsCfg):
    """Rewards for the Franka cloth environment."""

    reaching_deformable = RewTerm(
        func=mdp.deformable_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    lifting_deformable = RewTerm(
        func=mdp.deformable_lifting,
        params={"std": 0.1, "minimal_height": 0.11, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )


@configclass
class FrankaClothEnvCfg(FrankaSoftEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a surface deformable."""

    scene: FrankaClothScenePresetCfg = FrankaClothScenePresetCfg()
    events: FrankaClothEventCfg = FrankaClothEventCfg()
    rewards: FrankaClothRewardsCfg = FrankaClothRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # override the soft-beam physics with the cloth presets
        self.sim.physics = PhysicsCfg()
        # Fully close the gripper on the thin cloth; the shared beam default only closes to 0.01 m.
        self.actions.ik.gripper_action.close_command_expr = {"panda_finger_joint1": 0.0}


@configclass
class FrankaClothCameraEnvCfg(FrankaClothEnvCfg):
    """Visual Franka surface-deformable lifting environment."""

    scene: FrankaClothCameraScenePresetCfg = FrankaClothCameraScenePresetCfg()
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
