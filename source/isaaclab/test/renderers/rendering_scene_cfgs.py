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
from isaaclab.test.integration_scene_cfgs import RenderingSceneCfg, RenderingTestSceneCfg
from isaaclab.test.utils.rendering import CAMERA_EYE, CAMERA_TARGET
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
    init_state=FRANKA_PANDA_CFG.init_state.replace(pos=(-0.7, -0.25, 0.0)),
)
_FRANKA_TABLE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.CuboidCfg(
        size=(1.0, 0.8, 0.12),
        collision_props=sim_utils.CollisionBaseCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.22, 0.08), roughness=0.55),
        semantic_tags=[("class", "table")],
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.1, 0.4)),
)
_SUPPORT = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Support",
    spawn=sim_utils.CuboidCfg(
        size=(0.28, 0.28, 0.25),
        collision_props=sim_utils.CollisionBaseCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.24, 0.3), roughness=0.7),
        semantic_tags=[("class", "support")],
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.1, 0.585)),
)

_YOUNGS_MODULUS = 8.0e4
_POISSONS_RATIO = 0.25
_SOFT_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soft",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.32, 0.18, 0.16),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=300.0,
            k_mu=_YOUNGS_MODULUS / (2.0 * (1.0 + _POISSONS_RATIO)),
            k_lambda=_YOUNGS_MODULUS * _POISSONS_RATIO / ((1.0 + _POISSONS_RATIO) * (1.0 - 2.0 * _POISSONS_RATIO)),
            particle_radius=0.015,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.62), roughness=0.3),
        semantic_tags=[("class", "soft")],
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.45, 0.1, 0.88)),
)
_SOFT_PHYSX = _SOFT_NEWTON.replace(
    spawn=_SOFT_NEWTON.spawn.replace(
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        physics_material=PhysxDeformableBodyMaterialCfg(
            density=300.0,
            youngs_modulus=_YOUNGS_MODULUS,
            poissons_ratio=_POISSONS_RATIO,
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
    )
)
_CLOTH_NEWTON = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cloth",
    spawn=sim_utils.MeshRectangleCfg(
        size=(0.55, 0.55),
        resolution=(12, 12),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
            density=50.0,
            particle_radius=0.012,
            tri_ke=5.0e2,
            tri_ka=5.0e2,
            tri_kd=1.0e-3,
            edge_ke=2.0,
            edge_kd=1.0e-3,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.78, 0.08), roughness=0.5),
        semantic_tags=[("class", "cloth")],
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.45, 0.1, 0.9)),
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
    """Franka and one volume deformable, deliberately separated in frame."""

    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_TABLE.copy()
    soft: DeformableObjectCfg = _SOFT_NEWTON.copy()


@configclass
class FrankaClothRenderingSceneCfg(RenderingSceneCfg):
    """Franka and one low-resolution cloth over a visible support."""

    robot: ArticulationCfg = _FRANKA_ROBOT.copy()
    table: AssetBaseCfg = _FRANKA_TABLE.copy()
    support: AssetBaseCfg = _SUPPORT.copy()
    cloth: DeformableObjectCfg = _CLOTH_NEWTON.copy()


_KUKA_ROBOT = KUKA_ALLEGRO_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=KUKA_ALLEGRO_CFG.spawn.replace(
        rigid_props=KUKA_ALLEGRO_CFG.spawn.rigid_props.replace(disable_gravity=True),
        semantic_tags=[("class", "robot")],
    ),
    init_state=KUKA_ALLEGRO_CFG.init_state.replace(pos=(-0.65, -0.2, 0.0)),
)
_KUKA_OBJECT = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.CuboidCfg(
                size=(0.28, 0.22, 0.22),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.5, 0.95)),
                semantic_tags=[("class", "cube")],
            ),
            sim_utils.SphereCfg(
                radius=0.16,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.2)),
                semantic_tags=[("class", "sphere")],
            ),
            sim_utils.CapsuleCfg(
                radius=0.11,
                height=0.34,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.35)),
                semantic_tags=[("class", "capsule")],
            ),
        ],
        random_choice=False,
        rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
        collision_props=sim_utils.CollisionBaseCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.1, 0.68)),
)
_KUKA_TABLE = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.CuboidCfg(
        size=(0.9, 0.9, 0.12),
        collision_props=sim_utils.CollisionBaseCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.22, 0.09), roughness=0.6),
        semantic_tags=[("class", "table")],
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.52, 0.1, 0.46)),
)


@configclass
class KukaHeterogeneousRenderingSceneCfg(RenderingSceneCfg):
    """Three deterministic clone variants around a configured Kuka Allegro."""

    robot: ArticulationCfg = _KUKA_ROBOT.copy()
    object: RigidObjectCfg = _KUKA_OBJECT.copy()
    table: AssetBaseCfg = _KUKA_TABLE.copy()


_SHADOW_HAND_JOINT_POS = {
    "robot0_WR.*": 0.0,
    "robot0_(FF|MF|RF)J3": 0.2,
    "robot0_(FF|MF|RF)J(2|1)": 0.55,
    "robot0_LFJ(4|3)": 0.2,
    "robot0_LFJ(2|1)": 0.5,
    "robot0_THJ(4|3)": 0.35,
    "robot0_THJ2": 0.15,
    "robot0_THJ1": 0.2,
    "robot0_THJ0": 0.0,
}
_SHADOW_HAND_PHYSX = SHADOW_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SHADOW_HAND_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
    init_state=SHADOW_HAND_CFG.init_state.replace(pos=(-0.15, 0.05, 0.47), joint_pos=_SHADOW_HAND_JOINT_POS),
)
_SHADOW_HAND_OVPHYSX = _SHADOW_HAND_PHYSX.replace(spawn=_SHADOW_HAND_PHYSX.spawn.replace(fixed_tendons_props=None))
_SHADOW_HAND_NEWTON = SHADOW_HAND_NEWTON_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SHADOW_HAND_NEWTON_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
    init_state=SHADOW_HAND_NEWTON_CFG.init_state.replace(pos=(-0.15, 0.05, 0.47), joint_pos=_SHADOW_HAND_JOINT_POS),
)


@configclass
class ShadowHandRenderingSceneCfg(RenderingSceneCfg):
    """Configured high-DOF hand with a labeled, visibly offset cube."""

    robot: ArticulationCfg = _SHADOW_HAND_NEWTON.copy()
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.22, 0.88), roughness=0.25),
            semantic_tags=[("class", "cube")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.1, 0.55), rot=(0.0, 0.0, 0.258819, 0.965926)),
    )
    support = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=(0.85, 0.75, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.21, 0.27), roughness=0.75),
            semantic_tags=[("class", "support")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.02, -0.18, 0.38)),
    )


def make_rendering_scene_cfg(
    scene: str, physics: str
) -> tuple[
    InteractiveSceneCfg,
    tuple[float, float, float],
    tuple[float, float, float],
    frozenset[str],
    PhysicsCfg | None,
]:
    """Resolve scene-owned configuration while leaving lifecycle behavior to the runner."""
    if scene == "rendering_scene":
        return (
            RenderingTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=True),
            CAMERA_EYE,
            CAMERA_TARGET,
            frozenset({"robot"}),
            None,
        )
    if scene == "franka_soft":
        cfg = FrankaSoftRenderingSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=True)
        cfg.soft = (_SOFT_NEWTON if physics == "newton" else _SOFT_PHYSX).copy()
        physics_cfg = _DEFORMABLE_NEWTON_PHYSICS if physics == "newton" else None
        return cfg, (2.7, -3.2, 2.25), (0.05, 0.0, 0.72), frozenset(), physics_cfg
    if scene == "franka_cloth":
        cfg = FrankaClothRenderingSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=True)
        cfg.cloth = (_CLOTH_NEWTON if physics == "newton" else _CLOTH_PHYSX).copy()
        physics_cfg = _DEFORMABLE_NEWTON_PHYSICS if physics == "newton" else None
        return cfg, (2.7, -3.2, 2.25), (0.05, 0.0, 0.72), frozenset(), physics_cfg
    if scene == "kuka_heterogeneous":
        cfg = KukaHeterogeneousRenderingSceneCfg(num_envs=3, env_spacing=4.25, lazy_sensor_update=True)
        return cfg, (2.8, -4.0, 2.65), (0.0, 0.0, 0.82), frozenset(), None
    if scene == "shadow_hand":
        cfg = ShadowHandRenderingSceneCfg(num_envs=1, env_spacing=3.0, lazy_sensor_update=True)
        cfg.robot = {"physx": _SHADOW_HAND_PHYSX, "ovphysx": _SHADOW_HAND_OVPHYSX, "newton": _SHADOW_HAND_NEWTON}[
            physics
        ].copy()
        return cfg, (-1.15, -1.1, 1.05), (-0.05, 0.0, 0.62), frozenset({"cube"}), None
    raise ValueError(f"Unknown rendering scene: {scene!r}")
