# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-independent scene composition for specialized renderer probes."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import (
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.test.integration_scene_cfgs import RenderingSceneCfg, RenderingTestSceneCfg
from isaaclab.test.utils.rendering import CAMERA_EYE, CAMERA_TARGET
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG, SHADOW_HAND_NEWTON_CFG

_FRANKA_ROBOT = FRANKA_PANDA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=FRANKA_PANDA_CFG.spawn.replace(
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props.replace(disable_gravity=True),
        semantic_tags=[("class", "robot")],
    ),
)
_FRANKA_TABLE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        semantic_tags=[("class", "table")],
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
)
_FRANKA_CLOTH_CUBE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.03, 0.01, 0.08),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
        semantic_tags=[("class", "cube")],
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.04)),
)
_FRANKA_GROUND = AssetBaseCfg(
    prim_path="/World/GroundPlane",
    spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]),
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

_YOUNGS_MODULUS = 8.0e4
_POISSONS_RATIO = 0.25
_SOFT_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Deformable",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.05, 0.05),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=300.0,
            k_mu=_YOUNGS_MODULUS / (2.0 * (1.0 + _POISSONS_RATIO)),
            k_lambda=_YOUNGS_MODULUS * _POISSONS_RATIO / ((1.0 + _POISSONS_RATIO) * (1.0 - 2.0 * _POISSONS_RATIO)),
            particle_radius=0.01,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        semantic_tags=[("class", "soft")],
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
)
_SOFT_PHYSX = _SOFT_NEWTON.replace(
    spawn=_SOFT_NEWTON.spawn.replace(
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        physics_material=PhysxDeformableBodyMaterialCfg(
            density=300.0,
            youngs_modulus=_YOUNGS_MODULUS,
            poissons_ratio=_POISSONS_RATIO,
            static_friction=10.0,
            dynamic_friction=5.0,
        ),
    )
)
_CLOTH_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Deformable",
    spawn=sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        # The task uses 30x30; 12x12 preserves its silhouette at 128px without the solver cost.
        resolution=(12, 12),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
            density=50.0,
            particle_radius=0.005,
            tri_ke=5.0e2,
            tri_ka=5.0e2,
            tri_kd=1.0e-3,
            edge_ke=2.0,
            edge_kd=1.0e-3,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        semantic_tags=[("class", "cloth")],
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
)
_CLOTH_PHYSX = _CLOTH_NEWTON.replace(
    spawn=_CLOTH_NEWTON.spawn.replace(
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
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
    )
)
_DEFORMABLE_NEWTON_PHYSICS = NewtonCfg(
    solver_cfg=CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=MJWarpSolverCfg(ls_iterations=10, integrator="implicitfast"),
                bodies=[r"/World/envs/env_.*/Robot"],
            ),
            CouplerEntryCfg(
                name="soft", solver_cfg=VBDSolverCfg(iterations=5), all_particles=True, include_static_shapes=True
            ),
        ],
        model_cfg=NewtonModelCfg(soft_contact_ke=1.0e3, soft_contact_kd=1.0e-5, soft_contact_mu=0.5),
    ),
    num_substeps=2,
    use_cuda_graph=True,
)


@configclass
class FrankaSoftRenderingSceneCfg(RenderingSceneCfg):
    """Task-scale volume deformable beside a default Franka on its Seattle table."""

    ground = _FRANKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _FRANKA_CAMERA.copy()
    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_TABLE.copy()
    deformable: DeformableObjectCfg = _SOFT_NEWTON.copy()


@configclass
class FrankaClothRenderingSceneCfg(RenderingSceneCfg):
    """Task-scale cloth above its tiny table obstacle and default Franka."""

    ground = _FRANKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _FRANKA_CAMERA.copy()
    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_TABLE.copy()
    cube: AssetBaseCfg = _FRANKA_CLOTH_CUBE.copy()
    deformable: DeformableObjectCfg = _CLOTH_NEWTON.copy()


_KUKA_ROBOT = KUKA_ALLEGRO_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=KUKA_ALLEGRO_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
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
            sim_utils.MeshCuboidCfg(size=(0.05, 0.1, 0.1), semantic_tags=[("class", "cube")], **_KUKA_SHAPE_PHYSICS),
            sim_utils.MeshSphereCfg(radius=0.05, semantic_tags=[("class", "sphere")], **_KUKA_SHAPE_PHYSICS),
            sim_utils.MeshCapsuleCfg(
                radius=0.04, height=0.1, semantic_tags=[("class", "capsule")], **_KUKA_SHAPE_PHYSICS
            ),
            sim_utils.MeshConeCfg(radius=0.05, height=0.1, semantic_tags=[("class", "cone")], **_KUKA_SHAPE_PHYSICS),
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
# The task colors this footprint through success/failure markers; author its default failure state directly.
_KUKA_TABLE = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/table",
    spawn=sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15)),
        semantic_tags=[("class", "table")],
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235)),
)
_KUKA_GROUND = AssetBaseCfg(
    prim_path="/World/GroundPlane",
    spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0), semantic_tags=[("class", "ground")]),
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


_SHADOW_HAND_PHYSX = SHADOW_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SHADOW_HAND_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={".*": 0.0}),
)
_SHADOW_HAND_OVPHYSX = _SHADOW_HAND_PHYSX.replace(spawn=_SHADOW_HAND_PHYSX.spawn.replace(fixed_tendons_props=None))
_SHADOW_HAND_NEWTON = SHADOW_HAND_NEWTON_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SHADOW_HAND_NEWTON_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
)
_SHADOW_OBJECT_PHYSX = RigidObjectCfg(
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
        mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        semantic_tags=[("class", "cube")],
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6)),
)
_SHADOW_OBJECT_NEWTON = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        semantic_tags=[("class", "cube")],
        scale=(0.9, 0.9, 0.9),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.36, 0.535), joint_pos={}, joint_vel={}),
    actuators={},
    articulation_root_prim_path="",
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
    object: ArticulationCfg | RigidObjectCfg = _SHADOW_OBJECT_NEWTON.copy()


def make_rendering_scene_cfg(
    scene: str, physics: str
) -> tuple[
    InteractiveSceneCfg,
    tuple[float, float, float],
    tuple[float, float, float],
    frozenset[str],
    PhysicsCfg | None,
    frozenset[str],
]:
    """Resolve scene-owned configuration while leaving lifecycle behavior to the runner."""
    if scene == "rendering_scene":
        return (
            RenderingTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=True),
            CAMERA_EYE,
            CAMERA_TARGET,
            frozenset({"robot"}),
            None,
            frozenset(),
        )
    if scene in {"franka_soft", "franka_cloth"}:
        scene_cfg_type, newton_deformable, physx_deformable = {
            "franka_soft": (FrankaSoftRenderingSceneCfg, _SOFT_NEWTON, _SOFT_PHYSX),
            "franka_cloth": (FrankaClothRenderingSceneCfg, _CLOTH_NEWTON, _CLOTH_PHYSX),
        }[scene]
        cfg = scene_cfg_type(num_envs=4, env_spacing=3.0, lazy_sensor_update=True)
        cfg.deformable = (newton_deformable if physics == "newton" else physx_deformable).copy()
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        hand_actuator = cfg.robot.actuators["panda_hand"]
        hand_actuator.effort_limit_sim = 500.0
        hand_actuator.stiffness = 2000.0 if scene == "franka_cloth" else 1000.0
        hand_actuator.damping = 100.0
        physics_cfg = _DEFORMABLE_NEWTON_PHYSICS if physics == "newton" else None
        return cfg, (0.85, -0.55, 0.42), (0.20051, 0.099902, 0.025508), frozenset(), physics_cfg, frozenset()
    if scene == "kuka_heterogeneous":
        cfg = KukaHeterogeneousRenderingSceneCfg(num_envs=4, env_spacing=3.0, lazy_sensor_update=True)
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        return cfg, (0.57, -0.8, 0.5), (-0.296179, -0.299998, 0.500133), frozenset(), None, frozenset()
    if scene == "shadow_hand":
        cfg = ShadowHandRenderingSceneCfg(num_envs=4, env_spacing=2.0, lazy_sensor_update=True)
        cfg.robot = {"physx": _SHADOW_HAND_PHYSX, "ovphysx": _SHADOW_HAND_OVPHYSX, "newton": _SHADOW_HAND_NEWTON}[
            physics
        ].copy()
        cfg.object = (_SHADOW_OBJECT_NEWTON if physics == "newton" else _SHADOW_OBJECT_PHYSX).copy()
        return cfg, (0.0, -0.35, 1.0), (0.0, -0.35, 0.0), frozenset({"cube"}), None, frozenset({"robot"})
    raise ValueError(f"Unknown rendering scene: {scene!r}")
