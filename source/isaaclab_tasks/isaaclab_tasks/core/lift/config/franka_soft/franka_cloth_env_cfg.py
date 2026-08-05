# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka surface deformable lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

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
                        r"/World/envs/env_.*/Robot/Geometry/.*panda_hand",
                        r"/World/envs/env_.*/Robot/Geometry/.*panda_(left|right)finger",
                        r"/World/envs/env_.*/Cube",
                    ],
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                )
            ],
            iterations=1,
            model_cfg=NewtonModelCfg(soft_contact_mu=10.0),
        ),
        num_substeps=2,
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class DeformableCfg(PresetCfg):
    """Preset config for the deformable object, matching the Newton example."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(8, 8),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=10.0,
                particle_radius=0.001,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=0.5,
                edge_kd=1e-3,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.05, 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.03, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # low friction so the cloth slides off the cube easily when lifted
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
        ),
    )


@configclass
class FrankaClothScenePresetCfg(PresetCfg):
    """Preset config for the Franka surface deformable scene."""

    newton_mjwarp_vbd_proxy: FrankaClothSceneCfg = FrankaClothSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


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
        params={"std": 0.1, "minimal_height": 0.08, "asset_cfg": SceneEntityCfg("deformable")},
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


@configclass
class FrankaClothCameraEnvCfg(FrankaClothEnvCfg):
    """Visual Franka surface-deformable lifting environment."""

    scene: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
