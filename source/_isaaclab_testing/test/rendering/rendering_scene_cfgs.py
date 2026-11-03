# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-independent composition for shared and specialized rendering scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonCollisionCfg, NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    CableObjectCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
    VisualMaterialCfg,
)
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers.output_contract import RenderBufferKind
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG
from isaaclab_assets.robots.shadow_hand import (
    SHADOW_HAND_FRICTION_REPLACEMENTS,
    SHADOW_HAND_NEWTON_CFG,
    SHADOW_HAND_PHYSX_CFG,
)
from isaaclab_assets.robots.universal_robots import UR10_CFG


@configclass
class _RenderingMaterialCfg(sim_utils.MdlFileCfg):
    """Cross-renderer OmniPBR material for simple task-authored colors."""

    mdl_path: str = "OmniPBR.mdl"
    diffuse_color_constant: tuple[float, float, float] = (0.2, 0.2, 0.2)
    reflection_roughness_constant: float = 0.5
    metallic_constant: float = 0.0


_CARTPOLE_ROBOT = CARTPOLE_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=CARTPOLE_CFG.spawn.replace(semantic_tags=[("class", "cartpole")]),
)
_RENDERING_CARTPOLE_CFG = _CARTPOLE_ROBOT.replace(
    spawn=_CARTPOLE_ROBOT.spawn.replace(semantic_tags=[("class", "robot")]),
    init_state=ArticulationCfg.InitialStateCfg(pos=(-0.9, 0.35, 2.0), joint_pos={".*": 0.0}),
)


@configclass
class RenderingSceneCfg(InteractiveSceneCfg):
    """Ground, camera slot, and lighting shared by rendering scenes."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(6.0, 6.0), color=None, semantic_tags=[("class", "ground")]),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=256,
        height=256,
        offset=CameraCfg.OffsetCfg(
            pos=(3.8, -4.2, 3.4),
            rot=tuple(
                quat_from_matrix(
                    create_rotation_matrix_from_view(
                        torch.tensor([[3.8, -4.2, 3.4]]), torch.tensor([[0.05, 0.05, 1.25]])
                    )
                )[0].tolist()
            ),
            convention="opengl",
        ),
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=5.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0)
        ),
        # Pin the material used by constant-diffuse mode, independently of native reader defaults.
        render_settings={"omni:rtx:minimal:constantColor": ("Color3f", (0.5, 0.5, 0.5))},
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2400.0, color=(1.0, 0.91, 0.78), angle=0.7),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3826834, -0.2209424, 0.0, 0.8968727)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.58, 0.68, 1.0)),
    )


_CARTPOLE_LIGHT = AssetBaseCfg(
    prim_path="/World/DistantLight",
    init_state=AssetBaseCfg.InitialStateCfg(
        rot=(-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
    ),
    spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
)
_CARTPOLE_CAMERA = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    ),
    width=96,
    height=96,
)


@configclass
class CartpoleRenderingSceneCfg(RenderingSceneCfg):
    """Four-view Cartpole scene preserving the task's partitioning-sensitive composition."""

    ground = None
    camera: CameraCfg = _CARTPOLE_CAMERA.copy()
    key_light = _CARTPOLE_LIGHT.copy()
    fill_light = None
    robot: ArticulationCfg = _CARTPOLE_ROBOT.copy()


@configclass
class RenderingTestSceneCfg(RenderingSceneCfg):
    """Small deterministic scene covering renderer outputs and textured materials."""

    robot: ArticulationCfg = _RENDERING_CARTPOLE_CFG.copy()
    moving_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MovingCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.35),
            rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=_RenderingMaterialCfg(
                diffuse_color_constant=(0.12, 0.55, 0.95),
                metallic_constant=0.15,
                reflection_roughness_constant=0.25,
            ),
            visual_material_path="/World/Looks/RenderingMovingCube",
            semantic_tags=[("class", "moving_cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.15, -0.85, 0.55),
            rot=(0.0, 0.0, 0.1305262, 0.9914449),
            lin_vel=(0.6, 0.25, 0.0),
            ang_vel=(0.0, 0.0, 1.0),
        ),
    )
    # Referenced instanceable assets apply xform scale inconsistently across scene-reader backends. Keep
    # structural geometry explicit; the grid ground retains textured-asset coverage.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.15, 0.8, 0.12),
            visual_material=_RenderingMaterialCfg(
                diffuse_color_constant=(0.55, 0.25, 0.08), reflection_roughness_constant=0.55
            ),
            visual_material_path="/World/Looks/RenderingTable",
            semantic_tags=[("class", "table")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.95, 0.55, 0.46)),
    )
    cylinder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.16,
            height=0.42,
            visual_material=_RenderingMaterialCfg(
                diffuse_color_constant=(0.95, 0.22, 0.18),
                metallic_constant=0.4,
                reflection_roughness_constant=0.2,
            ),
            visual_material_path="/World/Looks/RenderingCylinder",
            semantic_tags=[("class", "cylinder")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.47, 0.73)),
    )
    sphere = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            visual_material=_RenderingMaterialCfg(
                diffuse_color_constant=(0.28, 0.85, 0.32),
                reflection_roughness_constant=0.35,
            ),
            visual_material_path="/World/Looks/RenderingSphere",
            semantic_tags=[("class", "sphere")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.18, 0.66, 0.72)),
    )


_FRANKA_ROBOT = FRANKA_PANDA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=FRANKA_PANDA_CFG.spawn.replace(rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props.replace(disable_gravity=True)),
    actuators={
        "panda_arm": FRANKA_PANDA_CFG.actuators["panda_arm"].replace(
            joint_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
            stiffness={
                "panda_joint[1-4]": 600.0,
                "panda_joint5": 250.0,
                "panda_joint6": 150.0,
                "panda_joint7": 50.0,
            },
            damping={
                "panda_joint[1-4]": 50.0,
                "panda_joint5": 30.0,
                "panda_joint6": 25.0,
                "panda_joint7": 15.0,
            },
            armature={
                "panda_joint[1-2]": 0.6057,
                "panda_joint[3-4]": 0.4625,
                "panda_joint[5-7]": 0.2055,
            },
        ),
        "panda_hand": FRANKA_PANDA_CFG.actuators["panda_hand"].replace(
            joint_effort_limit=500.0,
            actuator_velocity_limit=0.2,
            joint_velocity_limit=2.0,
            stiffness=1000.0,
            damping=100.0,
            armature=0.1,
        ),
        "panda_finger2_passive": FRANKA_PANDA_CFG.actuators["panda_finger2_passive"].replace(
            joint_effort_limit=1.0,
            actuator_velocity_limit=0.2,
            joint_velocity_limit=2.0,
            armature=0.1,
        ),
    },
)
_FRANKA_TABLE_SPAWN = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)),
)
_FRANKA_SOFT_TABLE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=_FRANKA_TABLE_SPAWN,
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
)
_FRANKA_CLOTH_SUPPORT = sim_utils.CuboidCfg(
    size=(0.1, 0.02, 0.15),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.01, dynamic_friction=0.01),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
)
_FRANKA_CLOTH_SUPPORT_NEG_Y = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SupportNegY",
    spawn=_FRANKA_CLOTH_SUPPORT.copy(),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.02, 0.075)),
)
_FRANKA_CLOTH_SUPPORT_POS_Y = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SupportPosY",
    spawn=_FRANKA_CLOTH_SUPPORT.copy(),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.02, 0.075)),
)
_FRANKA_GROUND = AssetBaseCfg(
    prim_path="/World/GroundPlane",
    spawn=sim_utils.GroundPlaneCfg(),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
)
_TASK_SKY_LIGHT = AssetBaseCfg(
    prim_path="/World/skyLight",
    spawn=sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    ),
)
_FRANKA_CAMERA = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(pos=(0.85, -0.55, 0.42), rot=(0.5080, 0.2114, 0.318, 0.7720), convention="opengl"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),
    width=128,
    height=128,
)

_YOUNGS_MODULUS = 2.0e5
_POISSONS_RATIO = 0.3
_SOFT_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Deformable",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.04, 0.04),
        edge_refinement=8.0,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=1000.0,
            k_mu=_YOUNGS_MODULUS / (2.0 * (1.0 + _POISSONS_RATIO)),
            k_lambda=_YOUNGS_MODULUS * _POISSONS_RATIO / ((1.0 + _POISSONS_RATIO) * (1.0 - 2.0 * _POISSONS_RATIO)),
            particle_radius=0.0025,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
)
_CLOTH_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Deformable",
    spawn=sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        edge_refinement=8.0,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
            density=1.0,
            particle_radius=0.002,
            tri_ke=5.0e2,
            tri_ka=5.0e2,
            tri_kd=1.0e-3,
            edge_ke=0.5,
            edge_kd=1.0e-3,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.102), rot=(0.70710678, 0.0, 0.0, 0.70710678)),
)
_FRANKA_HAND_PROXY_BODIES = [
    r"/World/envs/env_.*/Robot/Geometry/.*panda_hand",
    r"/World/envs/env_.*/Robot/Geometry/.*panda_(left|right)finger",
]


def _make_franka_newton_physics(
    rigid_bodies: list[str], proxy_bodies: list[str], contact_buffer_size: int
) -> NewtonCfg:
    """Build the shared rigid/VBD task solver with scene-specific ownership."""
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=50, integrator="implicitfast"),
                    bodies=rigid_bodies,
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=contact_buffer_size),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=proxy_bodies,
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(enable_rigid_soft_full_surface_contact=True),
                )
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=8.0e3, soft_contact_kd=1.0e-2, soft_contact_mu=10.0),
        num_substeps=2,
    )


_SOFT_NEWTON_PHYSICS = _make_franka_newton_physics([r"/World/envs/env_.*/Robot"], _FRANKA_HAND_PROXY_BODIES.copy(), 256)
_CLOTH_NEWTON_PHYSICS = _make_franka_newton_physics(
    [r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Support(Neg|Pos)Y"],
    [*_FRANKA_HAND_PROXY_BODIES, r"/World/envs/env_.*/Support(Neg|Pos)Y"],
    1024,
)


@configclass
class FrankaSoftRenderingSceneCfg(RenderingSceneCfg):
    """Task-scale soft beam and default Franka on the table."""

    ground = _FRANKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _FRANKA_CAMERA.copy()
    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_SOFT_TABLE.copy()
    deformable: DeformableObjectCfg = _SOFT_NEWTON.copy()


@configclass
class FrankaClothRenderingSceneCfg(RenderingSceneCfg):
    """Task-scale cloth and supports beside a default Franka."""

    ground = _FRANKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _FRANKA_CAMERA.copy()
    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_SOFT_TABLE.copy()
    support_neg_y: RigidObjectCfg = _FRANKA_CLOTH_SUPPORT_NEG_Y.copy()
    support_pos_y: RigidObjectCfg = _FRANKA_CLOTH_SUPPORT_POS_Y.copy()
    deformable: DeformableObjectCfg = _CLOTH_NEWTON.copy()


_KUKA_ROBOT = KUKA_ALLEGRO_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=KUKA_ALLEGRO_CFG.spawn.copy(),
)
_KUKA_SHAPE_PHYSICS = {
    "physics_material": sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
    "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
}
_KUKA_OBJECT = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiAssetSpawnerCfg(
        # One exact member of each topology in the task's larger shape family.
        assets_cfg=[
            sim_utils.MeshCuboidCfg(size=(0.025, 0.1, 0.1), **_KUKA_SHAPE_PHYSICS),
            sim_utils.MeshSphereCfg(radius=0.05, **_KUKA_SHAPE_PHYSICS),
            sim_utils.MeshCapsuleCfg(radius=0.04, height=0.1, **_KUKA_SHAPE_PHYSICS),
            sim_utils.MeshConeCfg(radius=0.05, height=0.1, **_KUKA_SHAPE_PHYSICS),
        ],
        random_choice=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=0, disable_gravity=False
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            mesh_collision_property=sim_utils.MeshCollisionPropertiesCfg(mesh_approximation_name="convexHull")
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
)
_KUKA_TABLE = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/table",
    spawn=sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235)),
)
_KUKA_GROUND = AssetBaseCfg(
    prim_path="/World/GroundPlane",
    spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
    collision_group=-1,
)
_KUKA_CAMERA = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(pos=(0.57, -0.8, 0.5), rot=(0.6124, 0.3536, 0.3536, 0.6124), convention="opengl"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
    width=64,
    height=64,
)


@configclass
class KukaHeterogeneousRenderingSceneCfg(RenderingSceneCfg):
    """Four deterministic clones drawn from the task's heterogeneous object family."""

    ground = _KUKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _KUKA_CAMERA.copy()
    robot: ArticulationCfg = _KUKA_ROBOT.copy()
    object: RigidObjectCfg = _KUKA_OBJECT.copy()
    table: RigidObjectCfg = _KUKA_TABLE.copy()


_SHADOW_HAND_PHYSX = SHADOW_HAND_PHYSX_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SHADOW_HAND_PHYSX_CFG.spawn.copy(),
)
_SHADOW_HAND_NEWTON = SHADOW_HAND_NEWTON_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SHADOW_HAND_NEWTON_CFG.spawn.copy(),
)
_SHADOW_OBJECT = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
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
_SHADOW_LIGHT = AssetBaseCfg(
    prim_path="/World/Light",
    spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
)
_SHADOW_CAMERA = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(pos=(0.0, -0.35, 1.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="world"),
    width=120,
    height=120,
    update_period=0.0,
    update_latest_camera_pose=True,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    ),
)


@configclass
class ShadowHandRenderingSceneCfg(RenderingSceneCfg):
    """Open default Shadow Hand beside the task's labeled DexCube."""

    ground = None
    key_light = None
    fill_light = _SHADOW_LIGHT.copy()
    camera: CameraCfg = _SHADOW_CAMERA.copy()
    robot: ArticulationCfg = _SHADOW_HAND_NEWTON.copy()
    object: RigidObjectCfg = _SHADOW_OBJECT.copy()


@configclass
class FrankaCableRenderingSceneCfg(FrankaSoftRenderingSceneCfg):
    """Simulated cable curves beside the rigid arm, without task reset/command machinery."""

    deformable = None
    cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=[(0.03 * index, 0.0, 0.0) for index in range(13)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=0.01,
                density=1000.0,
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e5,
            ),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.32, 0.0, 0.011)),
    )


def spawn_particle_push_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author the robot and welded paddle in one clone prototype."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    robot = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root in sim_utils.find_matching_prims(prim_path, stage=robot.GetStage()):
        stage = root.GetStage()
        paddle_path = f"{root.GetPath()}/ee_link/Paddle"
        paddle = sim_utils.CuboidCfg(
            size=(0.34, 0.24, 0.04),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=0.01, contact_gap=0.002),
            ],
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.7),
        )
        paddle.func(paddle_path, paddle, translation=(0.19, 0.0, 0.0))
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{paddle_path}/geometry/mesh")).MakeInvisible()
        visual = sim_utils.CuboidCfg(
            size=(0.34, 0.24, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.45, 0.82), metallic=0.25, roughness=0.35
            ),
        )
        visual.func(f"{paddle_path}/PaddleVisual", visual)
        joint = UsdPhysics.FixedJoint.Define(stage, f"{paddle_path}/FixedJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{root.GetPath()}/ee_link")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(paddle_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.19, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    return robot


@configclass
class ParticleRenderingSceneCfg(InteractiveSceneCfg):
    """Original UR10 workcell and tan particle pile, without task reset or IK managers."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.70710678, 0.70710678)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
        ),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0)
    )
    robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UR10_CFG.init_state.replace(
            joint_pos={
                "shoulder_pan_joint": -2.6687801,
                "shoulder_lift_joint": -0.9745972,
                "elbow_joint": 3.8264365,
                "wrist_1_joint": -1.2810427,
                "wrist_2_joint": 1.5707960,
                "wrist_3_joint": -1.0979838,
            }
        ),
        spawn=UR10_CFG.spawn.replace(
            func=spawn_particle_push_robot, joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)]
        ),
        actuators={"arm": UR10_CFG.actuators["arm"].replace(stiffness=2400.0, damping=70.0)},
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=128,
        height=128,
        offset=CameraCfg.OffsetCfg(
            pos=(1.15, -0.80, 0.55), rot=(0.541755, 0.163176, 0.237799, 0.789510), convention="opengl"
        ),
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),
    )
    media = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        init_state=MPMObjectCfg.InitialStateCfg(pos=(0.67, 0.0, 0.0)),
        spawn=MPMGridCfg(
            lower=(-0.274999 / 2, -0.349999 / 2, 0.0125 - 0.099999 / 16),
            upper=(0.274999 / 2, 0.349999 / 2, 0.0125 - 0.099999 / 16 + 0.099999),
            voxel_size=0.025,
            particles_per_cell=2.0,
            particle_placement="cell_center",
            jitter=0.0,
            radius=0.5 * math.prod((0.274999 / 22, 0.349999 / 28, 0.099999 / 8)) ** (1.0 / 3.0),
            material=MPMParticleMaterialCfg(density=300.0, friction=0.7, yield_pressure=1.0e12),
            visual_color=(0.83, 0.60, 0.22),
        ),
    )


@configclass
class AnymalRenderingSceneCfg(RenderingSceneCfg):
    """Standing ANYmal used to compare viewport and tiled presentation."""

    ground = None
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = ANYMAL_D_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", init_state=ANYMAL_D_CFG.init_state.replace(pos=(0.0, 0.0, 0.65))
    )


@dataclass(frozen=True)
class RenderingSceneSpec:
    """Scene-owned inputs and golden-image tolerances consumed by the shared runner."""

    cfg: InteractiveSceneCfg
    expected_instances: dict[str, str] = field(default_factory=dict)
    min_semantic_pixels: dict[str, int] = field(default_factory=dict)
    physics_cfg: PhysicsCfg | None = None
    image_max_diff_pct: float = 3.0
    min_ssim: float = 0.985
    dt: float = 1.0 / 60.0
    steps_before_capture: int = 0
    image_tolerance_overrides: dict[tuple[str, RenderBufferKind], tuple[float, float]] = field(default_factory=dict)

    def image_tolerance(self, renderer: str, aov: RenderBufferKind) -> tuple[float, float]:
        threshold = self.image_max_diff_pct
        if renderer == "ovrtx" and aov not in (
            RenderBufferKind.DEPTH,
            RenderBufferKind.DISTANCE_TO_CAMERA,
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        ):
            threshold = min(threshold, 3.0)
        default = (threshold, self.min_ssim)
        return self.image_tolerance_overrides.get((renderer, aov), default)


def make_rendering_scene_spec(scene: str, physics: str) -> RenderingSceneSpec:
    """Resolve scene-owned configuration while leaving lifecycle behavior to the runner."""
    if scene == "franka_cable":
        physics_cfg = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=50, integrator="implicitfast"),
                        bodies=[r"/World/envs/env_.*/Robot"],
                    ),
                    CouplerEntryCfg(
                        name="cable",
                        solver_cfg=VBDSolverCfg(iterations=10),
                        bodies=[r"/World/envs/env_.*/Cable"],
                        include_static_shapes=True,
                    ),
                ],
                proxies=[CouplerProxyMappingCfg(source="rigid", destination="cable", bodies=_FRANKA_HAND_PROXY_BODIES)],
                iterations=1,
            ),
            default_shape_cfg=NewtonShapeCfg(ke=2.5e3, kd=100.0, mu=10.0),
            num_substeps=4,
        )
        cfg = FrankaCableRenderingSceneCfg(num_envs=4, env_spacing=3.0)
        cfg.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=0.01, dynamic_friction=0.01)
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        cfg.robot.spawn.variants["Physics"] = "mujoco"
        return RenderingSceneSpec(
            cfg,
            physics_cfg=physics_cfg,
            image_max_diff_pct=8.0,
            dt=1.0 / 120.0,
            steps_before_capture=4,
            image_tolerance_overrides={("ovrtx", RenderBufferKind.RGB): (8.0, 0.98)},
        )
    if scene == "mpm":
        cfg = ParticleRenderingSceneCfg(num_envs=4, env_spacing=3.0)
        boxes = (
            ("WorkSurface", (1.28, 0.91, 0.04), (0.3939, 0.0, -0.02)),
            ("BinFloor", (0.36, 0.56, 0.04), (1.2139, 0.0, -0.20)),
            ("BinFront", (0.05, 0.66, 0.20), (1.0089, 0.0, -0.10)),
            ("BinBack", (0.05, 0.66, 0.32), (1.4189, 0.0, -0.04)),
            ("BinLeft", (0.41, 0.05, 0.32), (1.2139, 0.305, -0.04)),
            ("BinRight", (0.41, 0.05, 0.32), (1.2139, -0.305, -0.04)),
        )
        for name, size, position in boxes:
            vars(cfg)[f"mpm_{name}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/MPM{name}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visible=name != "WorkSurface",
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    collision_props=[
                        sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                        NewtonCollisionCfg(contact_margin=0.0125, contact_gap=0.0),
                    ],
                    physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.7),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.22, 0.34), roughness=0.65),
                ),
            )
            if name != "WorkSurface":
                vars(cfg)[f"rigid_{name}"] = AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Rigid{name}",
                    init_state=AssetBaseCfg.InitialStateCfg(pos=position),
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        visible=False,
                        collision_props=[
                            sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                            NewtonCollisionCfg(contact_margin=0.004, contact_gap=0.002),
                        ],
                        physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.7),
                    ),
                )
        physics_cfg = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="robot",
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False, integrator="implicitfast", njmax=256, nconmax=512
                        ),
                        bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Table"],
                        include_body_shapes=False,
                        substeps=3,
                    ),
                    CouplerEntryCfg(
                        name="media",
                        solver_cfg=MPMSolverCfg(
                            voxel_size=0.025,
                            grid_type="sparse",
                            grid_padding=0,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=24,
                            tolerance=1.0e-4,
                            warmstart_mode="auto",
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            collider_velocity_mode="forward",
                            solver="auto",
                            separate_worlds=True,
                            project_outside_colliders=False,
                            max_active_cell_count=6144,
                            max_leaf_node_count=192,
                            max_lower_node_count=32,
                            max_upper_node_count=32,
                        ),
                        bodies=[rf"/World/envs/env_.*/MPM{name}" for name, _, _ in boxes],
                        shape_label_patterns=[r".*/GroundPlane/CollisionPlane$"],
                        all_particles=True,
                        include_static_shapes=False,
                        include_child_joints=False,
                        substeps=1,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="robot",
                        destination="media",
                        bodies=[r"/World/envs/env_.*/Robot/ee_link/Paddle"],
                        mode="lagged",
                        mass_scale=10.0,
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=1,
            use_cuda_graph=True,
        )
        return RenderingSceneSpec(
            cfg,
            physics_cfg=physics_cfg,
            dt=1.0 / 120.0,
            steps_before_capture=60,
            image_max_diff_pct=8.0,
            min_ssim=0.97,
        )
    if scene == "kuka_materials":
        cfg = KukaHeterogeneousRenderingSceneCfg(num_envs=4, env_spacing=3.0)
        cfg.ground = cfg.object = cfg.table = None
        cfg.fill_light = AssetBaseCfg(
            prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=750.0)
        )
        cfg.camera.width = cfg.camera.height = 128
        cfg.camera.spawn.clipping_range = (0.01, 3.0)
        cfg.camera.offset = CameraCfg.OffsetCfg(
            pos=(1.2, -1.2, 1.1),
            rot=(0.5149, 0.2650, 0.3730, 0.7249),
            convention="opengl",
        )
        links = [
            *(f"iiwa7_link_{i}/visuals" for i in range(8)),
            "ee_link/palm_link/visuals",
            *(
                f"ee_link/{finger}_link_{i}/visuals"
                for finger in ("index", "middle", "ring", "thumb")
                for i in range(4)
            ),
        ]
        bindings = {}
        for index, link in enumerate(links):
            name = f"link_{index}"
            vars(cfg)[name] = VisualMaterialCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{name}", spawn=sim_utils.PbrMdlCfg())
            bindings[link] = f"./{name}"
        spawn = cfg.robot.spawn.replace(activate_contact_sensors=False, visual_material_bindings=bindings)
        cfg.robot.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=[spawn, spawn.copy()], random_choice=False)
        return RenderingSceneSpec(cfg, image_max_diff_pct=4.0, min_ssim=0.99, dt=0.005, steps_before_capture=1)
    if scene == "rendering_scene":
        return RenderingSceneSpec(
            cfg=RenderingTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=True),
            expected_instances={
                "/World/Ground": "ground",
                "/World/envs/env_0/Robot": "robot",
                "/World/envs/env_0/MovingCube": "moving_cube",
                "/World/envs/env_0/Table": "table",
                "/World/envs/env_0/Cylinder": "cylinder",
                "/World/envs/env_0/Sphere": "sphere",
            },
            min_semantic_pixels={"table": 500},
            image_tolerance_overrides={("isaac_rtx", RenderBufferKind.RGB): (8.0, 0.975)},
        )
    if scene == "cartpole":
        return RenderingSceneSpec(
            cfg=CartpoleRenderingSceneCfg(num_envs=4, env_spacing=20.0, lazy_sensor_update=True),
            expected_instances={f"/World/envs/env_{env_id}/Robot": "cartpole" for env_id in range(4)},
            image_max_diff_pct=1.5,
            dt=1.0 / 120.0,
        )
    if scene in {"franka_soft", "franka_cloth"}:
        scene_cfg_type, deformable, physics_cfg, max_diff_pct = {
            "franka_soft": (FrankaSoftRenderingSceneCfg, _SOFT_NEWTON, _SOFT_NEWTON_PHYSICS, 8.0),
            "franka_cloth": (
                FrankaClothRenderingSceneCfg,
                _CLOTH_NEWTON,
                _CLOTH_NEWTON_PHYSICS,
                8.0,
            ),
        }[scene]
        # Match the former golden harness override so neighboring task markers stay outside every camera view.
        cfg = scene_cfg_type(num_envs=4, env_spacing=3.0, lazy_sensor_update=True)
        cfg.deformable = deformable.copy()
        cfg.robot.spawn.variants["Physics"] = "physx" if physics == "physx" else "mujoco"
        if scene == "franka_cloth":
            cfg.robot.actuators["panda_hand"].stiffness = 2000.0
        if physics == "physx":
            cfg.deformable.cloning_contexts = ("isaaclab.cloner:UsdReplicateContext",)
            cfg.deformable.spawn.deformable_props = PhysxDeformableBodyPropertiesCfg()
            cfg.deformable.spawn.collision_props = PhysxCollisionCfg(rest_offset=0.002, contact_offset=0.01)
            cfg.deformable.spawn.physics_material = PhysxSurfaceDeformableBodyMaterialCfg(
                density=1000.0,
                surface_thickness=0.001,
                poissons_ratio=0.25,
                youngs_modulus=1e6,
                surface_bend_stiffness=1e6,
                elasticity_damping=0.1,
                bend_damping=0.1,
                static_friction=10.0,
                dynamic_friction=10.0,
            )
            physics_cfg = PhysxCfg(gpu_found_lost_pairs_capacity=2**22)
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        return RenderingSceneSpec(
            cfg=cfg,
            physics_cfg=physics_cfg,
            image_max_diff_pct=max_diff_pct,
            dt=1.0 / 120.0,
            steps_before_capture=4 if scene == "franka_cloth" else 0,
            image_tolerance_overrides=(
                {("isaac_rtx", RenderBufferKind.RGB): (20.0, 0.96)}
                if scene == "franka_cloth" and physics == "newton"
                else {}
            ),
        )
    if scene == "kuka_heterogeneous":
        cfg = KukaHeterogeneousRenderingSceneCfg(num_envs=4, env_spacing=3.0, lazy_sensor_update=True)
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        return RenderingSceneSpec(
            cfg=cfg,
            image_max_diff_pct=8.0,
            min_ssim=0.95,
        )
    if scene == "shadow_hand":
        cfg = ShadowHandRenderingSceneCfg(num_envs=4, env_spacing=2.0, lazy_sensor_update=True)
        cfg.robot = {"physx": _SHADOW_HAND_PHYSX, "ovphysx": _SHADOW_HAND_PHYSX, "newton": _SHADOW_HAND_NEWTON}[
            physics
        ].copy()
        cfg.object = _SHADOW_OBJECT.copy()
        return RenderingSceneSpec(
            cfg=cfg,
            expected_instances={f"/World/envs/env_{env_id}/object": "cube" for env_id in range(4)},
            physics_cfg=(
                OvPhysxCfg(legacy_joint_friction_replacements=SHADOW_HAND_FRICTION_REPLACEMENTS)
                if physics == "ovphysx"
                else None
            ),
            image_max_diff_pct=5.0,
            dt=1.0 / 120.0,
            image_tolerance_overrides={
                ("isaac_rtx", RenderBufferKind.RGB): (7.0 if physics == "newton" else 5.0, 0.985)
            },
        )
    raise ValueError(f"Unknown rendering scene: {scene!r}")
