# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-independent composition for shared and specialized rendering scenes."""

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from rendering_runtime import CAMERA_EYE, CAMERA_TARGET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers.output_contract import RenderBufferKind
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG, SHADOW_HAND_NEWTON_CFG

_RENDERING_CARTPOLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.9, 0.35, 2.0),
        joint_pos={"slider_to_cart": -0.25, "cart_to_pole": 0.45},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class RenderingSceneCfg(InteractiveSceneCfg):
    """Ground, camera slot, and lighting shared by rendering scenes."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.1),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.22, 0.28), roughness=0.8),
            semantic_tags=[("class", "ground")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    camera: CameraCfg | None = None
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2400.0, color=(1.0, 0.91, 0.78), angle=0.7),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3826834, -0.2209424, 0.0, 0.8968727)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.58, 0.68, 1.0)),
    )


@configclass
class RenderingTestSceneCfg(RenderingSceneCfg):
    """Small deterministic scene shared by renderer and visualizer integration tests."""

    robot: ArticulationCfg = _RENDERING_CARTPOLE_CFG.copy()
    moving_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MovingCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.35),
            rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.55, 0.95),
                metallic=0.15,
                roughness=0.25,
            ),
            semantic_tags=[("class", "moving_cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.15, -0.85, 0.55),
            rot=(0.0, 0.0, 0.1305262, 0.9914449),
            lin_vel=(0.6, 0.25, 0.0),
            ang_vel=(0.0, 0.0, 1.0),
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.15, 0.8, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.25, 0.08),
                roughness=0.55,
            ),
            semantic_tags=[("class", "table")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.95, 0.55, 0.46)),
    )
    cylinder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.16,
            height=0.42,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.22, 0.18),
                metallic=0.4,
                roughness=0.2,
            ),
            semantic_tags=[("class", "cylinder")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.47, 0.73)),
    )
    sphere = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.85, 0.32),
                roughness=0.35,
            ),
            semantic_tags=[("class", "sphere")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.18, 0.66, 0.72)),
    )


_FRANKA_ROBOT = FRANKA_PANDA_MENAGERIE_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=FRANKA_PANDA_MENAGERIE_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
    actuators={
        "panda_arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit_sim={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
            velocity_limit_sim={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
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
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1"],
            effort_limit_sim=70.0,
            velocity_limit=0.2,
            velocity_limit_sim=2.0,
            stiffness=350.0,
            damping=175.0,
            armature=0.1,
        ),
        "panda_finger2_passive": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint2"],
            effort_limit_sim=1.0,
            velocity_limit=0.2,
            velocity_limit_sim=2.0,
            stiffness=0.0,
            damping=0.0,
            armature=0.1,
        ),
    },
)
_FRANKA_FAILURE_TABLE_SPAWN = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    # The task always draws its otherwise-hidden table through this failure-state command marker. Authoring the
    # marker directly also makes it part of OVRTX's initial scene capture instead of a late manager-owned addition.
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)),
    semantic_tags=[("class", "table")],
)
_FRANKA_SOFT_TABLE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=_FRANKA_FAILURE_TABLE_SPAWN,
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
)
_FRANKA_CLOTH_TABLE = _FRANKA_SOFT_TABLE.replace(
    spawn=_FRANKA_FAILURE_TABLE_SPAWN.replace(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1)
    ),
)
_FRANKA_CLOTH_SUPPORT = sim_utils.CuboidCfg(
    size=(0.1, 0.02, 0.15),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
    semantic_tags=[("class", "support")],
)
_FRANKA_CLOTH_SUPPORT_NEG_Y = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SupportNegY",
    spawn=_FRANKA_CLOTH_SUPPORT,
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.02, 0.075)),
)
_FRANKA_CLOTH_SUPPORT_POS_Y = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SupportPosY",
    spawn=_FRANKA_CLOTH_SUPPORT,
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.02, 0.075)),
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

_YOUNGS_MODULUS = 2.0e5
_POISSONS_RATIO = 0.3
_SOFT_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Deformable",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.04, 0.04),
        edge_refinement=3.0,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=1000.0,
            k_mu=_YOUNGS_MODULUS / (2.0 * (1.0 + _POISSONS_RATIO)),
            k_lambda=_YOUNGS_MODULUS * _POISSONS_RATIO / ((1.0 + _POISSONS_RATIO) * (1.0 - 2.0 * _POISSONS_RATIO)),
            particle_radius=0.0025,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
        semantic_tags=[("class", "soft")],
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
)
_CLOTH_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Deformable",
    spawn=sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        resolution=(8, 8),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
            density=10.0,
            particle_radius=0.002,
            tri_ke=5.0e2,
            tri_ka=5.0e2,
            tri_kd=1.0e-3,
            edge_ke=0.5,
            edge_kd=1.0e-3,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        semantic_tags=[("class", "cloth")],
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
                    solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=20, integrator="implicitfast"),
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
            model_cfg=NewtonModelCfg(soft_contact_ke=8.0e3, soft_contact_mu=10.0),
        ),
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
    """Task-scale soft beam and default Menagerie Franka on the failure-state table."""

    ground = _FRANKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _FRANKA_CAMERA.copy()
    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_SOFT_TABLE.copy()
    deformable: DeformableObjectCfg = _SOFT_NEWTON.copy()


@configclass
class FrankaClothRenderingSceneCfg(RenderingSceneCfg):
    """Task-scale cloth and supports beside a default Menagerie Franka."""

    ground = _FRANKA_GROUND.copy()
    key_light = None
    fill_light = _TASK_SKY_LIGHT.copy()
    camera: CameraCfg = _FRANKA_CAMERA.copy()
    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_CLOTH_TABLE.copy()
    support_neg_y: RigidObjectCfg = _FRANKA_CLOTH_SUPPORT_NEG_Y.copy()
    support_pos_y: RigidObjectCfg = _FRANKA_CLOTH_SUPPORT_POS_Y.copy()
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


@dataclass(frozen=True)
class RenderingSceneSpec:
    """Scene-owned inputs and golden-image tolerances consumed by the shared runner."""

    cfg: InteractiveSceneCfg
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    expected_instances: dict[str, str] = field(default_factory=dict)
    physics_cfg: PhysicsCfg | None = None
    preserve_fixed_articulation_roots: frozenset[str] = frozenset()
    image_max_diff_pct: float = 3.0
    min_ssim: float = 0.98
    image_tolerance_overrides: dict[tuple[str, RenderBufferKind], tuple[float, float]] = field(default_factory=dict)

    def image_tolerance(self, renderer: str, aov: RenderBufferKind) -> tuple[float, float]:
        return self.image_tolerance_overrides.get((renderer, aov), (self.image_max_diff_pct, self.min_ssim))


def make_rendering_scene_spec(scene: str, physics: str) -> RenderingSceneSpec:
    """Resolve scene-owned configuration while leaving lifecycle behavior to the runner."""
    if scene == "rendering_scene":
        return RenderingSceneSpec(
            cfg=RenderingTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=True),
            camera_eye=CAMERA_EYE,
            camera_target=CAMERA_TARGET,
            expected_instances={
                "/World/Ground": "ground",
                "/World/envs/env_0/Robot": "robot",
                "/World/envs/env_0/MovingCube": "moving_cube",
                "/World/envs/env_0/Table": "table",
                "/World/envs/env_0/Cylinder": "cylinder",
                "/World/envs/env_0/Sphere": "sphere",
            },
            image_tolerance_overrides={("isaac_rtx", RenderBufferKind.RGB): (8.0, 0.975)},
        )
    if scene in {"franka_soft", "franka_cloth"}:
        if physics != "newton":
            raise ValueError(f"{scene} rendering requires its shared Newton task composition.")
        scene_cfg_type, deformable, physics_cfg, max_diff_pct = {
            "franka_soft": (FrankaSoftRenderingSceneCfg, _SOFT_NEWTON, _SOFT_NEWTON_PHYSICS, 12.0),
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
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        return RenderingSceneSpec(
            cfg=cfg,
            camera_eye=(0.85, -0.55, 0.42),
            camera_target=(0.20051, 0.099902, 0.025508),
            physics_cfg=physics_cfg,
            image_max_diff_pct=max_diff_pct,
        )
    if scene == "kuka_heterogeneous":
        cfg = KukaHeterogeneousRenderingSceneCfg(num_envs=4, env_spacing=3.0, lazy_sensor_update=True)
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
        return RenderingSceneSpec(
            cfg=cfg,
            camera_eye=(0.57, -0.8, 0.5),
            camera_target=(-0.296179, -0.299998, 0.500133),
            expected_instances={
                "/World/GroundPlane": "ground",
                **{f"/World/envs/env_{env_id}/Robot": "robot" for env_id in range(4)},
                **{f"/World/envs/env_{env_id}/table": "table" for env_id in range(4)},
                **{
                    f"/World/envs/env_{env_id}/Object": label
                    for env_id, label in enumerate(("cube", "sphere", "capsule", "cone"))
                },
            },
            image_max_diff_pct=15.0,
            min_ssim=0.95,
        )
    if scene == "shadow_hand":
        cfg = ShadowHandRenderingSceneCfg(num_envs=4, env_spacing=2.0, lazy_sensor_update=True)
        cfg.robot = {"physx": _SHADOW_HAND_PHYSX, "ovphysx": _SHADOW_HAND_OVPHYSX, "newton": _SHADOW_HAND_NEWTON}[
            physics
        ].copy()
        cfg.object = (_SHADOW_OBJECT_NEWTON if physics == "newton" else _SHADOW_OBJECT_PHYSX).copy()
        return RenderingSceneSpec(
            cfg=cfg,
            camera_eye=(0.0, -0.35, 1.0),
            camera_target=(0.0, -0.35, 0.0),
            expected_instances={
                f"/World/envs/env_{env_id}/{prim}": label
                for env_id in range(4)
                for prim, label in (("Robot", "robot"), ("object", "cube"))
            },
            preserve_fixed_articulation_roots=frozenset({"robot"}),
            image_max_diff_pct=10.0,
        )
    raise ValueError(f"Unknown rendering scene: {scene!r}")
