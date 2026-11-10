# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare contact-driven bimanual tablecloth pulls at three speeds.

InteractiveScene owns fixed-base H2 Plus robots with SHARPA hands, cloths, and tableware.
Newton VBD simulates the articulations, cloth, and tableware together; Newton IK and a Warp
state machine run inside the physics CUDA graph. This is scripted manipulation, not an RL task.

.. code-block:: bash

    uv run isaaclab example tablecloth --device cuda:0
    uv run --extra isaacsim isaaclab example tablecloth --device cuda:0 --visualizer kit
"""

from __future__ import annotations

import argparse
import logging
import math
from functools import partial

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

logger = logging.getLogger("isaaclab.examples.tablecloth")

parser = argparse.ArgumentParser(description="Three side-by-side GPU humanoid tablecloth pulls.")
parser.add_argument(
    "--speeds",
    nargs="+",
    type=float,
    default=[4.0, 1.5, 0.5],
    help="Peak pull speeds, ordered left to right from fastest to slowest [m/s].",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Frame limit; zero completes all pulls and a final hold, negative runs forever.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if not args_cli.device.startswith("cuda"):
    parser.error("This graph-captured example requires a CUDA device.")
if any(not math.isfinite(speed) or speed <= 0 for speed in args_cli.speeds):
    parser.error("--speeds must contain positive finite values.")

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import (
    NewtonArticulationCfg,
    NewtonCollisionCfg,
    NewtonDeformableBodyPropertiesCfg,
    NewtonSDFCollisionCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg, NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import clone, configclass, instantiate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    import newton
    from newton import Mesh, ModelBuilder

    from pxr import Usd

    from isaaclab.assets import Articulation


# Scene configuration and native contact preparation.

RIGID_GAP = 0.001
CLOTH_FRICTION = 0.20
GRIP_FRICTION = 8.0
# Pin the manufacturer's articulated H2 Plus / SHARPA model; do not vendor its meshes.
H2_PLUS_USD = (
    "https://media.githubusercontent.com/media/unitreerobotics/unitree_model/"
    "b6a8942b0803b6c137e58cef12beb4b03e4a2fa7/H2_Plus/H2_with_sharpa.usd"
)

TABLE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "ArchVis/Residential/Furniture/DiningSets/EastRural/EastRural_Table.usd"
)
BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Dishware/Bowl_G01/sm_kitchenware_bowl_g01_01.usd"
WINE_GLASS_USD = (
    f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Furnishing/Kitchen/Kitchenware/Dishware/Glass_Wine_A01/"
    "sm_dishware_glass_wine_a01_01.usd"
)
FORK_USD = f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Utensils/Fork_K01/sm_kitchenware_fork_k01_01.usd"
CANTINA_USD = f"{ISAACLAB_NUCLEUS_DIR}/Environments/TableClothPull/tablecloth_cantina.usdc"

# Table coordinates are authored in centimeters; USD references do not convert stage units.
TABLE_ASSET_SIZE = (91.44, 152.7302, 76.20)
TABLE_ASSET_TOP_THICKNESS = 2.54
# Keep the calibrated front edge at x = -0.20 m when enlarging the dining table.
TABLE_SIZE = (0.48, 0.84)
TABLE_CENTER_X = 0.04
# Dimensions measured from the SimReady tableware assets [m].
BOWL_LOCAL_Z_MIN = 0.000001783
BOWL_LOCAL_RADIUS = 0.052765541
BOWL_LOCAL_HEIGHT = 0.043569469
WINE_GLASS_LOCAL_Z_MIN = 0.000358
FORK_LOCAL_Z_MIN = 0.000127
BOWL_SCALE = 1.35
FORK_SCALE = 0.85
FORK_ROTATION = (0.0, 0.0, -0.70710678, 0.70710678)
FORK_CENTER_OF_MASS = (0.0, 0.0092, 0.0089)
FORK_DIAGONAL_INERTIA = (1.86e-4, 6.2e-6, 1.90e-4)
# A base-only collision proxy would otherwise give the tall glass a flat-disk inertia.
WINE_GLASS_CENTER_OF_MASS = (0.0, 0.0, 0.04)
WINE_GLASS_DIAGONAL_INERTIA = (3.0e-3, 3.0e-3, 5.0e-4)


def _material(
    *, friction: float, contact_stiffness: float = 1.0e4, contact_damping: float = 1.0e1
) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Return solver-common friction and Newton compliant-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        NewtonMaterialCfg(contact_stiffness=contact_stiffness, contact_damping=contact_damping),
    ]


def _collision(
    *, contact_margin: float = 0.002, contact_gap: float = RIGID_GAP
) -> list[UsdPhysicsCollisionCfg | NewtonCollisionCfg]:
    """Return collision properties shared by cloth-contacting rigid shapes."""
    return [
        UsdPhysicsCollisionCfg(collision_enabled=True),
        NewtonCollisionCfg(contact_margin=contact_margin, contact_gap=contact_gap),
    ]


def _table_collider(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    """Create a low-cost collision proxy for a visual-only table."""
    return sim_utils.CuboidCfg(
        size=size,
        visible=False,
        # Inflated rigid offsets must not support dishes above the cloth.
        collision_props=_collision(contact_margin=0.0002, contact_gap=0.0),
        # Newton's geometric-mean mixing gives 0.3 cloth-table friction.
        physics_material=_material(friction=0.45),
    )


def _configure_visuals(
    prim_path: str,
    color: tuple[float, float, float] | None = None,
    opacity: float | None = None,
    roughness: float | None = None,
) -> None:
    """Remove inherited physics bindings and normalize cross-viewer materials."""
    from pxr import Gf, UsdShade  # noqa: PLC0415

    for child in sim_utils.get_all_matching_child_prims(prim_path):
        binding = child.GetRelationship("material:binding:physics")
        if binding:
            binding.SetTargets([])
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        shader_id = shader.GetIdAttr().Get()
        if shader_id == "UsdPreviewSurface":
            for name in ("metallic", "roughness"):
                shader.GetInput(name).DisconnectSource()
            names = ("diffuseColor", "opacity", "roughness")
        elif shader_id == "mdl:OmniGlass":
            names = ("glass_color", "cutout_opacity", "frosting_roughness")
        else:
            continue
        for name, value in zip(names, (color, opacity, roughness), strict=True):
            if value is not None:
                shader.GetInput(name).Set(Gf.Vec3f(*value) if name == names[0] else value)


def _spawn_visual_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn the table or tableware, with primitive proxies for the tableware."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    from pxr import Gf, UsdPhysics  # noqa: PLC0415

    for root in sim_utils.find_matching_prims(prim_path):
        root_path = root.GetPath().pathString
        if not isinstance(cfg, _TablewareCfg):
            _configure_visuals(root_path)
            continue
        _configure_visuals(root_path, cfg.visual_color, cfg.visual_opacity, cfg.visual_roughness)

        mesh_colliders = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda child: child.HasAPI(UsdPhysics.CollisionAPI),
        )
        if len(mesh_colliders) != 1:
            raise RuntimeError(f"Expected one SimReady tableware mesh collider, found {len(mesh_colliders)}")
        UsdPhysics.CollisionAPI(mesh_colliders[0]).CreateCollisionEnabledAttr(False)

        bodies = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda child: child.HasAPI(UsdPhysics.RigidBodyAPI),
        )
        if len(bodies) != 1:
            raise RuntimeError(f"Expected one SimReady tableware rigid body, found {len(bodies)}")
        body = bodies[0]
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(cfg.mass)
        mass_api.CreateDensityAttr(0.0)
        if cfg.center_of_mass is not None:
            mass_api.CreateCenterOfMassAttr(cfg.center_of_mass)
        if cfg.diagonal_inertia is not None:
            mass_api.CreateDiagonalInertiaAttr(cfg.diagonal_inertia)
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

        for index, (proxy_cfg, proxy_position) in enumerate(
            zip(cfg.collision_proxy_cfgs, cfg.collision_proxy_positions, strict=True)
        ):
            suffix = "" if index == 0 else str(index)
            proxy_cfg.func(
                f"{body.GetPath()}/CollisionProxy{suffix}",
                proxy_cfg,
                translation=proxy_position,
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
    return prim


@configclass
class _TablewareCfg(sim_utils.UsdFileCfg):
    """Visuals with body-local collision proxies [m], mass [kg] and inertia [kg m²]."""

    func: Callable | str = _spawn_visual_usd
    make_uninstanceable: bool = True
    rigid_props = {"(/.*)?": [UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False)]}
    mass: float = MISSING
    collision_proxy_cfgs: tuple[sim_utils.ShapeCfg, ...] = MISSING
    collision_proxy_positions: tuple[tuple[float, float, float], ...] = MISSING
    center_of_mass: tuple[float, float, float] | None = None
    diagonal_inertia: tuple[float, float, float] | None = None
    visual_color: tuple[float, float, float] | None = None
    visual_opacity: float | None = None
    visual_roughness: float | None = None


FPS = 60
SUBSTEPS = 16
# Table height calibrated for the fixed-pelvis H2 stance and native fingertip pinch.
TABLE_TOP_Z = 1.0352
TABLEWARE_CLEARANCE = 0.011
# Start above the table contact skin and the cloth's 4 mm particle radius.
CLOTH_Z = TABLE_TOP_Z + 0.007

TABLE_SCALE = (
    TABLE_SIZE[0] / TABLE_ASSET_SIZE[0],
    TABLE_SIZE[1] / TABLE_ASSET_SIZE[1],
    TABLE_TOP_Z / TABLE_ASSET_SIZE[2],
)
TABLETOP_THICKNESS = TABLE_ASSET_TOP_THICKNESS * TABLE_SCALE[2]

H2_HAND_OFFSETS = (
    (0.08611124818088065, -0.005675509582575613, 0.11567062942496234),
    (0.08611175437468235, 0.006718550527126339, 0.1156889716558436),
)
# Measured native pulp pinch with room for the thicker contact skin, inside native joint limits.
H2_PINCH_JOINT_POS = {
    "left_thumb_CMC_FE": 1.8749698686599732,
    "left_thumb_CMC_AA": 0.27846200197935106,
    "left_thumb_MCP_FE": -0.49359992146492004,
    "left_thumb_MCP_AA": -0.0139111201092601,
    "left_thumb_IP": 0.030000001192092896,
    "left_index_MCP_FE": 0.9650716781616211,
    "left_index_MCP_AA": 0.30174076557159424,
    "left_index_PIP": 0.5233824253082275,
    "left_index_DIP": 0.029999999329447746,
    "right_thumb_CMC_FE": 1.8719424211978912,
    "right_thumb_CMC_AA": 0.2733340892195702,
    "right_thumb_MCP_FE": -0.49359992146492004,
    "right_thumb_MCP_AA": 0.0021895832382142384,
    "right_thumb_IP": 0.029999999329447746,
    "right_index_MCP_FE": 0.9501051306724548,
    "right_index_MCP_AA": 0.2907813787460327,
    "right_index_PIP": 0.5469146966934204,
    "right_index_DIP": 0.029999999329447746,
}
# Unused fingers remain at these positions throughout the motion.
H2_PARKED_JOINT_POS = {
    "left_middle_MCP_FE": 0.5188307166099548,
    "left_middle_MCP_AA": 0.30825042724609375,
    "left_middle_PIP": 1.3207045793533325,
    "left_middle_DIP": 1.3419662714004517,
    "(left|right)_(ring|pinky)_MCP_FE": 0.10000000149011612,
    "(left|right)_(ring|pinky)_MCP_AA": 0.30000001192092896,
    "(left|right)_ring_PIP": 1.5499999523162842,
    "(left|right)_(ring|pinky)_DIP": 1.2000000476837158,
    "left_pinky_PIP": 1.6299999952316284,
    "right_middle_MCP_FE": 0.4482609033584595,
    "right_middle_MCP_AA": 0.30533933639526367,
    "right_middle_PIP": 1.3855232000350952,
    "right_middle_DIP": 1.3496774435043335,
    "right_pinky_PIP": 1.5499999523162842,
    "(left|right)_pinky_CMC": 0.10000000149011612,
}
H2_OPEN_JOINT_POS = {
    **H2_PINCH_JOINT_POS,
    # The index stays stationary; the open thumb admits the cloth's 8 mm contact diameter.
    "left_thumb_CMC_FE": 1.830179924964905,
    "left_thumb_CMC_AA": 0.15654811978340155,
    "left_thumb_MCP_AA": 0.06805907756090157,
    "right_thumb_CMC_FE": 1.8180701351165771,
    "right_thumb_CMC_AA": 0.1360364687442781,
    "right_thumb_MCP_AA": 0.07434246167540544,
}


def _prepare_contacts(
    builder: ModelBuilder,
    world_index: int,
    world_position: np.ndarray,
    world_quaternion: np.ndarray,
    *,
    device: str,
    mesh_sdf_cache: dict[tuple, Mesh],
) -> None:
    """Cook unbaked native mesh SDFs once and allow opposing fingertip contacts."""
    from newton import GeoType, Mesh, ShapeFlags  # noqa: PLC0415

    root = f"/World/envs/env_{world_index}/Robot/"
    required_flags = ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES
    for shape, label in enumerate(builder.shape_label):
        if not label.startswith(root) or not builder.shape_flags[shape] & required_flags:
            continue
        if builder.shape_type[shape] == GeoType.CONVEX_MESH:
            raise RuntimeError(f"Expected unapproximated native H2 collision geometry: {label}")
        if builder.shape_type[shape] != GeoType.MESH:
            continue
        if builder.shape_flags[shape] & required_flags != required_flags:
            raise RuntimeError(f"Native H2 Mesh requires both collision flags for attached-SDF consumption: {label}")
        if tuple(builder.shape_scale[shape]) != (1.0, 1.0, 1.0):
            raise RuntimeError(f"Native H2 contact preparation requires unit mesh scale: {label}")
        source = builder.shape_source[shape]
        if not isinstance(source, Mesh) or source.sdf is not None:
            raise RuntimeError(f"Expected an imported native H2 Mesh without an attached SDF: {label}")

        resolution = builder.shape_sdf_max_resolution[shape]
        voxel_size = builder.shape_sdf_target_voxel_size[shape]
        if resolution is None and voxel_size is None:
            raise RuntimeError(f"Native H2 Mesh has no imported SDF cooking resolution: {label}")
        narrow_band = tuple(builder.shape_sdf_narrow_band_range[shape])
        padding = builder.shape_sdf_padding[shape]
        if padding is None:
            # Match Newton's deferred mesh-cooking fallback, not build_sdf's default 0.05 m padding.
            padding = builder.shape_gap[shape]
            if builder.shape_flags[shape] & ShapeFlags.HYDROELASTIC:
                padding += builder.shape_margin[shape]
        texture_format = builder.shape_sdf_texture_format[shape]
        paired_samples = builder.sdf_texture_paired_samples
        key = (id(source), resolution, voxel_size, narrow_band, padding, texture_format, paired_samples, device)
        cooked = mesh_sdf_cache.get(key)
        if cooked is None:
            cooked = source.copy(recompute_inertia=False)
            # The pinned Newton soft-contact path leaves baked SDF gradients unnormalized.
            # Public unbaked cooking uses its normalized runtime-scale branch instead.
            cooked.build_sdf(
                device=device,
                narrow_band_range=narrow_band,
                target_voxel_size=voxel_size,
                max_resolution=resolution,
                margin=padding,
                shape_margin=0.0,
                scale=None,
                texture_format=texture_format,
                paired_samples=paired_samples,
            )
            mesh_sdf_cache[key] = cooked
        builder.shape_source[shape] = cooked

    pairs = set()
    for side in ("left", "right"):
        indices = []
        for finger in ("thumb", "index"):
            prefix = f"{root}{side}_{finger}_DP/"
            matches = [
                index
                for index, label in enumerate(builder.shape_label)
                if label.startswith(prefix)
                and "/collisions/elastomer_" in label
                and builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one H2 native fingertip skin under {prefix}, found {len(matches)}.")
            indices.append(matches[0])
        pairs.add(tuple(sorted(indices)))
    builder.shape_collision_filter_pairs = [pair for pair in builder.shape_collision_filter_pairs if pair not in pairs]


def _configure_h2_contacts(root_path: str, stage: Usd.Stage) -> None:
    """Apply native hand precision and grip materials before scene replication."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

    from isaaclab.sim.schemas import apply_namespaced  # noqa: PLC0415
    from isaaclab.sim.spawners.materials.physics_materials import (  # noqa: PLC0415
        spawn_physics_material_from_fragments,
    )

    grasp_material = _material(
        friction=GRIP_FRICTION**2 / CLOTH_FRICTION,
        contact_stiffness=1.0e4,
        contact_damping=20.0,
    )
    grasp_material_path = f"{root_path}/GraspMaterial"
    spawn_physics_material_from_fragments(grasp_material_path, grasp_material, stage=stage)

    # Use a precision-contact skin across the entire hand. A 2 mm margin
    # on unused fingers can push the hem before the visible surfaces touch.
    for mesh in sim_utils.get_all_matching_child_prims(
        root_path,
        predicate=lambda prim: prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI),
        stage=stage,
    ):
        mesh_path = mesh.GetPath().pathString
        # Full-surface cloth contact needs native meshes with SDFs, not convex hulls.
        mesh.RemoveAppliedSchema("NewtonMeshCollisionAPI")
        UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Set("none")
        apply_namespaced(NewtonSDFCollisionCfg(sdf_max_resolution=64, sdf_padding=0.012), mesh_path, stage)
        if any(
            f"/{side}_{part}_" in mesh_path
            for side in ("left", "right")
            for part in ("hand", "thumb", "index", "middle", "ring", "pinky")
        ):
            apply_namespaced(NewtonCollisionCfg(contact_margin=0.0002), mesh_path, stage)
        if any(f"/{side}_{finger}_DP/" in mesh_path for side in ("left", "right") for finger in ("thumb", "index")):
            # Bind the grip material to native rubber, leaving plastic unchanged.
            if "/collisions/elastomer_" in mesh_path:
                # Resolve the millimeter-scale pinch without refining the whole robot.
                apply_namespaced(NewtonSDFCollisionCfg(sdf_max_resolution=128, sdf_padding=0.012), mesh_path, stage)
                sim_utils.bind_physics_material(mesh.GetPath(), grasp_material_path, stage=stage)


TABLE_SUPPORT_CFGS = {
    f"table_leg_{index}": AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/TableLeg{index}",
        spawn=_table_collider((0.024, 0.024, TABLE_TOP_Z - TABLETOP_THICKNESS)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X + x, y, 0.5 * (TABLE_TOP_Z - TABLETOP_THICKNESS))),
    )
    # Four slender legs replace the old panel proxies; no invisible wall below the hem.
    for index, (x, y) in enumerate(((-0.224, -0.403), (-0.224, 0.403), (0.224, -0.403), (0.224, 0.403)))
}


TABLECLOTH_ASSETS = {
    "robot": ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=H2_PLUS_USD,
            make_uninstanceable=True,
            fix_root_link=True,
            articulation_props={"/pelvis": [NewtonArticulationCfg(self_collision_enabled=False)]},
            rigid_props={"/pelvis": [UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]},
            collision_props=_collision(),
            physics_material=_material(friction=0.50, contact_stiffness=1.0e3, contact_damping=1.0e-2),
        ),
        # A grounded shallow stance clears the pelvis from the cloth while keeping both soles flat.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.80, 0.0, 0.9297120821774005),
            joint_pos={
                "waist_(roll|yaw)_joint": 0.0,
                "head_.*": 0.0,
                "waist_pitch_joint": 0.35,
                "(left|right)_hip_pitch_joint": -0.5,
                "(left|right)_hip_roll_joint": 0.0,
                "(left|right)_hip_yaw_joint": 0.0,
                "(left|right)_knee_joint": 1.0,
                "(left|right)_ankle_pitch_joint": -0.5702976584434509,
                "left_ankle_roll_joint": 0.06301647424697876,
                "right_ankle_roll_joint": -0.06301647424697876,
                # A measured ready pose keeps the open fingertips clear of the cloth at reset.
                "left_shoulder_pitch_joint": 0.0324070528,
                "left_shoulder_roll_joint": -0.1293495595,
                "left_shoulder_yaw_joint": 0.3300454915,
                "left_elbow_joint": -0.4972611964,
                "left_wrist_roll_joint": -1.3867278099,
                "left_wrist_pitch_joint": 0.3297203779,
                "left_wrist_yaw_joint": 0.8414497375,
                "right_shoulder_pitch_joint": 0.0907456726,
                "right_shoulder_roll_joint": -0.2215068787,
                "right_shoulder_yaw_joint": -0.0401092954,
                "right_elbow_joint": -0.4736894071,
                "right_wrist_roll_joint": 1.5572344065,
                "right_wrist_pitch_joint": -0.0186706446,
                "right_wrist_yaw_joint": -0.8570303917,
                **H2_OPEN_JOINT_POS,
                **H2_PARKED_JOINT_POS,
            },
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_(hip|knee|ankle|shoulder|elbow)_.*", r"(waist|head)_.*"],
                stiffness=5.0e5,
                damping=1.5e3,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_wrist_.*"],
                stiffness=5.0e4,
                damping=1.0e2,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_(thumb|index|middle|ring|pinky)_.*"],
                # Simulation drives only: direct VBD does not enforce native hardware torque caps.
                stiffness=4.0e3,
                damping=1.0,
            ),
        },
    ),
    "table": AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            func=_spawn_visual_usd,
            usd_path=TABLE_USD,
            scale=TABLE_SCALE,
            make_uninstanceable=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, 0.0)),
    ),
    "tabletop": AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=_table_collider((*TABLE_SIZE, TABLETOP_THICKNESS)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_TOP_Z - 0.5 * TABLETOP_THICKNESS)),
    ),
    **TABLE_SUPPORT_CFGS,
    "cloth": DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        spawn=sim_utils.MeshRectangleCfg(
            # A 4 cm free hem settles before the scripted hands approach.
            size=(0.46, TABLE_SIZE[1]),
            # 289 particles per cloth; render subdivision smooths the coarse surface in Kit.
            edge_refinement=16,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.05, 0.05)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=0.24,
                particle_radius=0.004,
                edge_ke=0.01,
                edge_kd=5.0e-3,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.01, 0.0, CLOTH_Z)),
    ),
    "bowl": RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=_TablewareCfg(
            usd_path=BOWL_USD,
            mass=0.45,
            collision_proxy_cfgs=(
                sim_utils.CylinderCfg(
                    radius=BOWL_LOCAL_RADIUS,
                    height=BOWL_LOCAL_HEIGHT,
                    visible=False,
                    collision_props=_collision(contact_margin=0.0002, contact_gap=0.0),
                    physics_material=_material(friction=0.30),
                ),
            ),
            collision_proxy_positions=((0.0, 0.0, BOWL_LOCAL_Z_MIN + 0.5 * BOWL_LOCAL_HEIGHT),),
            scale=(BOWL_SCALE,) * 3,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.08, -0.07, CLOTH_Z + TABLEWARE_CLEARANCE - BOWL_SCALE * BOWL_LOCAL_Z_MIN), rot=(0.0, 0.0, 0.0, 1.0)
        ),
    ),
    "glass": RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Glass",
        spawn=_TablewareCfg(
            usd_path=WINE_GLASS_USD,
            mass=0.35,
            center_of_mass=WINE_GLASS_CENTER_OF_MASS,
            diagonal_inertia=WINE_GLASS_DIAGONAL_INERTIA,
            # A base-only proxy lets the visual stem/cup penetrate the ground after a fall.
            collision_proxy_cfgs=(
                sim_utils.CylinderCfg(
                    radius=0.040,
                    height=0.006,
                    visible=False,
                    collision_props=_collision(contact_margin=0.0002, contact_gap=0.0),
                    physics_material=_material(friction=0.15),
                ),
                sim_utils.CylinderCfg(
                    radius=0.004,
                    height=0.096,
                    visible=False,
                    collision_props=_collision(contact_margin=0.0002, contact_gap=0.0),
                    physics_material=_material(friction=0.15),
                ),
                sim_utils.CapsuleCfg(
                    radius=0.041,
                    height=0.045,
                    visible=False,
                    collision_props=_collision(contact_margin=0.0002, contact_gap=0.0),
                    physics_material=_material(friction=0.15),
                ),
            ),
            collision_proxy_positions=((0.0, 0.0, 0.003), (0.0, 0.0, 0.054), (0.0, 0.0, 0.165)),
            visual_color=(0.55, 0.75, 0.90),
            visual_opacity=0.90,
            visual_roughness=0.12,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.10, 0.13, CLOTH_Z + TABLEWARE_CLEARANCE - WINE_GLASS_LOCAL_Z_MIN), rot=(0.0, 0.0, 0.0, 1.0)
        ),
    ),
    "fork": RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fork",
        spawn=_TablewareCfg(
            usd_path=FORK_USD,
            mass=0.07,
            collision_proxy_cfgs=(
                sim_utils.CuboidCfg(
                    size=(0.014, 0.160, 0.003),
                    visible=False,
                    collision_props=_collision(contact_margin=0.0002, contact_gap=0.0),
                    physics_material=_material(friction=0.25),
                ),
            ),
            collision_proxy_positions=((0.0, 0.0092, 0.0016),),
            scale=(FORK_SCALE,) * 3,
            center_of_mass=FORK_CENTER_OF_MASS,
            diagonal_inertia=FORK_DIAGONAL_INERTIA,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.12, -0.22, CLOTH_Z + TABLEWARE_CLEARANCE - FORK_SCALE * FORK_LOCAL_Z_MIN), rot=FORK_ROTATION
        ),
    ),
    "ground": AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(visible=False),
        collision_group=-1,
    ),
    "light": AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    ),
    # Illuminate the pull stations below the cantina's opaque ceiling.
    "interior_light": AssetBaseCfg(
        prim_path="/World/InteriorLight",
        spawn=sim_utils.DiskLightCfg(radius=1.5, intensity=16000.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.0, 3.0), rot=(0.0, 0.3007058, 0.0, 0.95371695)),
    ),
    "backdrop": AssetBaseCfg(
        prim_path="/World/Backdrop",
        spawn=sim_utils.UsdFileCfg(usd_path=CANTINA_USD),
    ),
}


# Scripted GPU manipulation and physical grasp monitoring.

H2_PULL_DISTANCE = 0.45
PULL_TIME_FACTOR = 1.875
# Settle, reach, lower, insert, align, close, withdraw, and lift.
H2_PHASE_DURATIONS = (0.20, 0.40, 0.40, 0.40, 0.40, 0.30, 0.40, 0.40)
H2_PHASE_KEYFRAMES = ((0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (4, 5), (5, 6))
STATE_PULL = wp.constant(len(H2_PHASE_DURATIONS))
STATE_HOLD = wp.constant(len(H2_PHASE_DURATIONS) + 1)

# Positions are expressed in the fixed pelvis frame.
# Ready, clearance, precontact, insertion, pinch, withdrawal, and lift poses [m].
# Center each pinch on a boundary vertex of the coarse cloth.
H2_HAND_KEYFRAMES = (
    (0.45, 0.2625379991, 0.163),
    (0.45, -0.2625441579, 0.163),
    (0.55, 0.2625379991, 0.153),
    (0.55, -0.2625441579, 0.153),
    (0.55, 0.2625379991, 0.068),
    (0.55, -0.2625441579, 0.068),
    (0.577, 0.2625379991, 0.073),
    (0.577, -0.2625441579, 0.073),
    (0.578403, 0.262525, 0.081011),
    (0.577649, -0.262555, 0.081554),
    (0.558403, 0.262525, 0.090011),
    (0.557649, -0.262555, 0.090554),
    (0.5628056417, 0.2625379991, 0.1300655547),
    (0.5628101729, -0.2625441579, 0.130071595),
)
# Control the complete wrist pose: a free roll can put the proximal fingers inside the table.
H2_READY_HAND_ROTATIONS = (
    (-0.9239897494, 0.0197845272, -0.3665561809, 0.1071824696),
    (0.9231279041, 0.0512161522, 0.3652593359, 0.1086158183),
)
H2_GRASP_HAND_ROTATIONS = (
    (-0.807865338839, -0.006732944032, -0.579201247457, 0.108785002508),
    (0.812353708056, 0.024445186614, 0.570667120832, 0.117570927880),
)
H2_PULL_HAND_ROTATIONS = H2_GRASP_HAND_ROTATIONS
# Keep the pinch attitude during lift and pull so the coarse hem stays between the pads.
H2_HAND_ROTATIONS = (*H2_READY_HAND_ROTATIONS, *(H2_GRASP_HAND_ROTATIONS * 6))
# The index joints are identical in the asset's open and closed poses; only the thumbs close.
STATE_CLOSE = wp.constant(5)


@wp.func
def _smoothstep(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@wp.func
def _minimum_jerk(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


@wp.func
def _interpolate_rotation(start: wp.vec4, end: wp.vec4, alpha: float) -> wp.vec4:
    # Quaternion signs describe the same pose; always take the shorter rotation.
    if wp.dot(start, end) < 0.0:
        end = -end
    return wp.normalize(wp.lerp(start, end, alpha))


@wp.kernel
def _advance_trajectory(
    dt: float,
    pull_speed: wp.array(dtype=float),
    keyframes: wp.array2d(dtype=wp.transform),
    pull_hand_rotations: wp.array(dtype=wp.vec4),
    phase_durations: wp.array(dtype=float),
    phase_keyframes: wp.array(dtype=wp.vec2i),
    state: wp.array(dtype=wp.int32),
    state_time: wp.array(dtype=float),
    closure_targets: wp.array(dtype=float),
    root: wp.transform,
    left_positions: wp.array(dtype=wp.vec3),
    left_rotations: wp.array(dtype=wp.vec4),
    right_positions: wp.array(dtype=wp.vec3),
    right_rotations: wp.array(dtype=wp.vec4),
):
    env_id = wp.tid()
    current_state = state[env_id]
    elapsed = state_time[env_id] + dt

    if current_state < STATE_PULL and elapsed >= phase_durations[current_state]:
        current_state += 1
        elapsed = 0.0

    closure = float(1.0)
    alpha = float(0.0)
    distance = H2_PULL_DISTANCE
    start_keyframe = keyframes.shape[0] - 1
    end_keyframe = start_keyframe
    if current_state < STATE_PULL:
        alpha = _smoothstep(elapsed / phase_durations[current_state])
        start_keyframe = phase_keyframes[current_state][0]
        end_keyframe = phase_keyframes[current_state][1]
        if current_state < STATE_CLOSE:
            closure = 0.0
        elif current_state == STATE_CLOSE:
            closure = alpha
    elif current_state == STATE_PULL:
        # Zero endpoint velocity and acceleration prevent a sudden load on the
        # pinch. The quintic's peak derivative is 1.875, preserving --speeds.
        duration = PULL_TIME_FACTOR * H2_PULL_DISTANCE / pull_speed[env_id]
        distance = H2_PULL_DISTANCE * _minimum_jerk(elapsed / duration)

    for side in range(2):
        start = keyframes[start_keyframe, side]
        end = keyframes[end_keyframe, side]
        position = wp.transform_get_translation(start)
        q = wp.transform_get_rotation(start)
        rotation = wp.vec4(q[0], q[1], q[2], q[3])
        if current_state < STATE_PULL:
            position = wp.lerp(position, wp.transform_get_translation(end), alpha)
            q = wp.transform_get_rotation(end)
            rotation = _interpolate_rotation(rotation, wp.vec4(q[0], q[1], q[2], q[3]), alpha)
        else:
            position += wp.vec3(-distance, 0.0, 0.0)
            if current_state == STATE_PULL:
                rotation = _interpolate_rotation(rotation, pull_hand_rotations[side], distance / H2_PULL_DISTANCE)
            else:
                rotation = pull_hand_rotations[side]
        position = wp.transform_point(root, position)
        q = wp.transform_get_rotation(root) * wp.quat(rotation[0], rotation[1], rotation[2], rotation[3])
        rotation = wp.vec4(q[0], q[1], q[2], q[3])
        if side == 0:
            left_positions[env_id] = position
            left_rotations[env_id] = rotation
        else:
            right_positions[env_id] = position
            right_rotations[env_id] = rotation

    closure_targets[env_id] = closure
    if current_state == STATE_PULL and distance >= H2_PULL_DISTANCE:
        current_state = STATE_HOLD
    state[env_id] = current_state
    state_time[env_id] = elapsed


@wp.kernel
def _set_elbow_position(
    body_poses: wp.array2d(dtype=wp.transform),
    root_body: int,
    elbow_body: int,
    native_root: wp.transform,
    guard_y: float,
    states: wp.array(dtype=int),
    times: wp.array(dtype=float),
    durations: wp.array(dtype=float),
    targets: wp.array(dtype=wp.vec3),
):
    lane = wp.tid()
    root_inverse = wp.transform_inverse(body_poses[lane, root_body])
    position = wp.transform_point(root_inverse, wp.transform_get_translation(body_poses[lane, elbow_body]))
    blend = float(1.0)
    if states[lane] == 0:
        blend = _smoothstep(times[lane] / durations[0])
    outward_y = wp.min(position[1], guard_y)
    if guard_y > 0.0:
        outward_y = wp.max(position[1], guard_y)
    position[1] = wp.lerp(position[1], outward_y, blend)
    targets[lane] = wp.transform_point(native_root, position)


@wp.kernel
def _gather_targets(
    solved: wp.array2d(dtype=float),
    closure: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    fingers: wp.array(dtype=int),
    finger_open: wp.array(dtype=float),
    finger_closed: wp.array(dtype=float),
    lower: wp.array(dtype=float),
    upper: wp.array(dtype=float),
    seed: wp.array2d(dtype=float),
    arm_start: wp.array2d(dtype=float),
    arm_end: wp.array2d(dtype=float),
    arm_counters: wp.array(dtype=int),
    velocity_limits: wp.array2d(dtype=float),
    dt: float,
    targets: wp.array2d(dtype=float),
    velocities: wp.array2d(dtype=float),
):
    lane, joint = wp.tid()
    value = targets[lane, joint]
    if indices[joint] >= 0:
        coordinate = indices[joint]
        value = wp.clamp(solved[lane, coordinate], lower[coordinate], upper[coordinate])
        seed[lane, coordinate] = value
        arm_start[lane, coordinate] = arm_end[lane, coordinate]
        arm_end[lane, coordinate] = value
        if coordinate == 0:
            arm_counters[lane] = 0
    if fingers[joint] >= 0:
        finger = fingers[joint]
        value = wp.lerp(finger_open[finger], finger_closed[finger], closure[lane])
    velocity = float(0.0)
    if indices[joint] >= 0:
        # Match moving arm targets without exciting the quasi-massless fingers.
        limit = wp.min(velocity_limits[lane, joint], 20.0)
        velocity = wp.clamp((value - targets[lane, joint]) / dt, -limit, limit)
    targets[lane, joint] = value
    velocities[lane, joint] = velocity


@wp.kernel(enable_backward=False)
def _interpolate_arm_targets(
    coordinates: wp.array(dtype=int),
    start: wp.array2d(dtype=float),
    end: wp.array2d(dtype=float),
    counters: wp.array(dtype=int),
    substeps: int,
    targets: wp.array2d(dtype=float),
):
    lane = wp.tid()
    tick = wp.min(counters[lane] + 1, substeps)
    for arm in range(coordinates.shape[0]):
        value = end[lane, arm]
        if tick < substeps:
            fraction = float(tick) / float(substeps)
            value = start[lane, arm] + fraction * (end[lane, arm] - start[lane, arm])
        targets[lane, coordinates[arm]] = value
    counters[lane] = tick


class _Controller:
    """Own the GPU trajectory, arm IK and contact-only finger targets."""

    def __init__(
        self,
        robot: Articulation,
        usd_path: str,
        dt: float,
        speeds: tuple[float, ...],
        *,
        state: newton.State,
        control: newton.Control,
        substeps: int,
    ) -> None:
        import newton  # noqa: PLC0415
        from isaaclab_newton.controllers.ik import (  # noqa: PLC0415
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
            NewtonIKSolver,
            NewtonIKSolverCfg,
        )

        from pxr import Gf, Usd, UsdPhysics  # noqa: PLC0415

        self.robot = robot
        self.device = robot.device
        self.num_lanes = len(speeds)
        self.num_joints = robot.num_joints
        self._dt = dt
        if robot.num_instances != self.num_lanes:
            raise ValueError("The example requires one identical fixed-base robot per environment.")
        self.hand_body_names = ("left_hand_C_MC", "right_hand_C_MC")
        self.hand_offsets = H2_HAND_OFFSETS
        arm_expr = [r"(left|right)_(shoulder|elbow|wrist)_.*"]
        finger_names = list(H2_OPEN_JOINT_POS)
        _, arm_names = robot.find_joints(arm_expr)

        # Lock non-IK coordinates in an anonymous session layer only. The
        # simulation retains every articulated finger; IK solves just the arms.
        # This also keeps LM's tiled solve within the GPU shared-memory budget.
        stage = Usd.Stage.Open(usd_path)
        stage.SetEditTarget(stage.GetSessionLayer())
        defaults = robot.data.default_joint_pos.warp.numpy()
        fixed_positions = dict(zip(robot.joint_names, defaults[0], strict=True))
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() not in arm_names:
                # A fixed joint must retain the configured posture, especially
                # a forward waist bend. Freezing the authored zero angle would
                # give IK a different arm chain from the simulated robot.
                angle = float(fixed_positions.get(prim.GetName(), 0.0))
                if angle != 0.0:
                    joint = UsdPhysics.RevoluteJoint(prim)
                    axis = Gf.Vec3f(
                        *{"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[joint.GetAxisAttr().Get()]
                    )
                    rotation = Gf.Quatf(math.cos(0.5 * angle), math.sin(0.5 * angle) * axis)
                    joint.GetLocalRot0Attr().Set(joint.GetLocalRot0Attr().Get() * rotation)
                prim.SetTypeName("PhysicsFixedJoint")
        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            floating=False,
            enable_self_collisions=False,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
        )
        model = builder.finalize(device=self.device)
        # Reserve room for compliant tracking without commanding physical hard stops.
        model.joint_limit_lower.assign(model.joint_limit_lower.numpy() + 0.08)
        model.joint_limit_upper.assign(model.joint_limit_upper.numpy() - 0.08)
        if model.joint_coord_count != len(arm_names) or model.joint_coord_count != model.joint_dof_count:
            raise ValueError("H2 IK requires one scalar coordinate per arm joint.")
        self._ik_lower = model.joint_limit_lower
        self._ik_upper = model.joint_limit_upper
        starts = model.joint_q_start.numpy()
        coordinates = {
            label.split("/")[-1]: int(start)
            for label, start, end in zip(model.joint_label, starts[:-1], starts[1:], strict=True)
            if end > start
        }
        self._arm_joint_names = list(coordinates)
        links = {label.split("/")[-1]: index for index, label in enumerate(model.body_label)}
        self._indices = wp.array(
            [coordinates.get(name, -1) for name in robot.joint_names], dtype=int, device=self.device
        )
        seed_positions = defaults[:, [robot.joint_names.index(name) for name in coordinates]]
        self._seed = wp.array(seed_positions, dtype=float, device=self.device)
        self._velocity_limits = wp.clone(robot.data.joint_vel_limits.warp)
        self._targets = wp.clone(robot.data.default_joint_pos.warp)
        self._velocity_targets = wp.zeros_like(self._targets)
        finger_ids, _ = robot.find_joints(finger_names, preserve_order=True)
        finger_indices = {joint: index for index, joint in enumerate(finger_ids)}
        self._fingers = wp.array(
            [finger_indices.get(i, -1) for i in range(self.num_joints)], dtype=int, device=self.device
        )
        # The fixed prototype pelvis defines the trajectory's coordinate frame.
        root_pose = model.body_q.numpy()[links["pelvis"]]
        self._root = wp.transform(root_pose[:3], root_pose[3:])
        objectives = [
            NewtonIKPoseObjectiveCfg(
                body_name=body,
                name=name + "_hand",
                body_offset_pos=offset,
                use_relative_mode=False,
                position_weight=500.0,
                rotation_weight=50.0,
            )
            for name, body, offset in zip(("left", "right"), self.hand_body_names, self.hand_offsets, strict=True)
        ]
        objectives.append(NewtonIKJointLimitObjectiveCfg(weight=100.0))
        objectives.extend(
            NewtonIKPoseObjectiveCfg(
                body_name=f"{side}_elbow_link",
                name=f"{side}_elbow",
                command_type="position",
                use_relative_mode=False,
                position_weight=5.0,
                rotation_weight=0.0,
            )
            for side in ("left", "right")
        )
        self._solver = NewtonIKSolver(
            NewtonIKSolverCfg(iterations=32, lambda_initial=0.1),
            model=model,
            num_envs=self.num_lanes,
            device=self.device,
            objectives=objectives,
            link_resolver=lambda name: links[name],
        )
        self._poses = [self._solver.objectives_by_name[name] for name in ("left_hand", "right_hand")]
        self._elbow_positions = [
            self._solver.objectives_by_name[f"{side}_elbow"].position_objective.target_positions
            for side in ("left", "right")
        ]
        # All motion state is owned here and allocated before physics capture.
        self._speeds = wp.array(speeds, dtype=float, device=self.device)
        self._keyframes = wp.array(
            [wp.transform(p, q) for p, q in zip(H2_HAND_KEYFRAMES, H2_HAND_ROTATIONS, strict=True)],
            dtype=wp.transform,
            device=self.device,
        ).reshape((-1, 2))
        self._pull_rotations = wp.array(H2_PULL_HAND_ROTATIONS, dtype=wp.vec4, device=self.device)
        self._phase_durations = wp.array(H2_PHASE_DURATIONS, dtype=float, device=self.device)
        self._phase_keyframes = wp.array(H2_PHASE_KEYFRAMES, dtype=wp.vec2i, device=self.device)
        self._finger_open = wp.array(tuple(H2_OPEN_JOINT_POS.values()), dtype=float, device=self.device)
        self._finger_closed = wp.array(
            [H2_PINCH_JOINT_POS[name] for name in H2_OPEN_JOINT_POS], dtype=float, device=self.device
        )
        self._phases = wp.zeros(self.num_lanes, dtype=int, device=self.device)
        self._phase_times = wp.zeros(self.num_lanes, dtype=float, device=self.device)
        self._closure = wp.zeros(self.num_lanes, dtype=float, device=self.device)

        # Bind public articulation selections and native views before graph capture.
        view = self.robot.root_view
        if view.joint_coord_count != view.joint_dof_count:
            raise ValueError("This H2 example requires scalar joint coordinates.")
        bodies, _ = self.robot.find_bodies(
            ["pelvis", "left_elbow_link", "right_elbow_link", *self.hand_body_names], preserve_order=True
        )
        bodies = self.robot.map_body_ids_to_backend(bodies)
        self._scene_body_poses = view.get_link_transforms(state)[:, 0]
        self._scene_root = bodies[0]
        self._scene_elbows = bodies[1:3]
        self._scene_hands = wp.array(bodies[3:], dtype=int, device=self.device)
        arms, _ = self.robot.find_joints(self._arm_joint_names, preserve_order=True)
        arms = self.robot.map_joint_ids_to_backend(arms)
        self._arm_target_coordinates = wp.array(arms, dtype=int, device=self.device)
        self._arm_control_targets = view.get_attribute("joint_target_q", control)[:, 0]
        self._arm_start = wp.array(view.get_dof_positions(state).numpy()[:, 0, arms], dtype=float, device=self.device)
        self._arm_end = wp.clone(self._arm_start)
        self._arm_counters = wp.zeros(self.num_lanes, dtype=int, device=self.device)
        self._arm_substeps = substeps
        self.compute()

    def interpolate_arm_targets(self) -> None:
        """Apply the next linear arm-position sample after the native actuator adapter."""
        wp.launch(
            _interpolate_arm_targets,
            dim=self.num_lanes,
            inputs=[
                self._arm_target_coordinates,
                self._arm_start,
                self._arm_end,
                self._arm_counters,
                self._arm_substeps,
                self._arm_control_targets,
            ],
            device=self.device,
        )

    def compute(self) -> None:
        """Advance the GPU controller and submit the next frame's joint targets."""
        wp.launch(
            _advance_trajectory,
            dim=self.num_lanes,
            inputs=[
                self._dt,
                self._speeds,
                self._keyframes,
                self._pull_rotations,
                self._phase_durations,
                self._phase_keyframes,
                self._phases,
                self._phase_times,
                self._closure,
                self._root,
                self._poses[0].position_objective.target_positions,
                self._poses[0].rotation_objective.target_rotations,
                self._poses[1].position_objective.target_positions,
                self._poses[1].rotation_objective.target_rotations,
            ],
            device=self.device,
        )
        # Soft posture targets discourage torso/elbow crossings while leaving X/Z free.
        for elbows, positions, guard in zip(self._scene_elbows, self._elbow_positions, (0.180, -0.180), strict=True):
            wp.launch(
                _set_elbow_position,
                dim=self.num_lanes,
                inputs=[
                    self._scene_body_poses,
                    self._scene_root,
                    elbows,
                    self._root,
                    guard,
                    self._phases,
                    self._phase_times,
                    self._phase_durations,
                    positions,
                ],
                device=self.device,
            )
        solved = self._solver.solve(self._seed)
        # Clamp soft IK limits and retain the commanded seed in the same target pass.
        wp.launch(
            _gather_targets,
            dim=(self.num_lanes, self.num_joints),
            inputs=[
                solved,
                self._closure,
                self._indices,
                self._fingers,
                self._finger_open,
                self._finger_closed,
                self._ik_lower,
                self._ik_upper,
                self._seed,
                self._arm_start,
                self._arm_end,
                self._arm_counters,
                self._velocity_limits,
                self._dt,
                self._targets,
                self._velocity_targets,
            ],
            device=self.device,
        )
        self.robot.actuators.target_command.set_position_index(value=self._targets)
        self.robot.actuators.target_command.set_velocity_index(value=self._velocity_targets)
        self.robot.write_data_to_sim()


@wp.kernel
def _measure_grasp(
    mesh: wp.uint64,
    body_poses: wp.array2d(dtype=wp.transform),
    hands: wp.array(dtype=int),
    offsets: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=int),
    faces: wp.array2d(dtype=int),
    coordinates: wp.array2d(dtype=wp.vec2),
    initial_points: wp.array2d(dtype=wp.vec3),
    initial_gaps: wp.array2d(dtype=float),
    maximum_gaps: wp.array2d(dtype=float),
    travel: wp.array2d(dtype=float),
):
    lane, side = wp.tid()
    if states[lane] >= STATE_PULL:
        hand = wp.transform_point(body_poses[lane, hands[side]], offsets[side])
        if faces[lane, side] == -1:
            # Try once at pull onset; a later accidental pickup is not a successful grasp.
            faces[lane, side] = -2
            query = wp.mesh_query_point_no_sign(mesh, hand, 0.2)
            if query.result:
                faces[lane, side] = query.face
                coordinates[lane, side] = wp.vec2(query.u, query.v)
                point = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                initial_points[lane, side] = point
                initial_gaps[lane, side] = wp.length(point - hand)
        if faces[lane, side] >= 0:
            uv = coordinates[lane, side]
            point = wp.mesh_eval_position(mesh, faces[lane, side], uv[0], uv[1])
            maximum_gaps[lane, side] = wp.max(maximum_gaps[lane, side], wp.length(point - hand))
            travel[lane, side] = initial_points[lane, side][0] - point[0]


class _GraspMonitor:
    """Track fixed cloth material points on the GPU, rather than just commanded motion.

    Sample after each physics frame inside the graph. Only :meth:`report`,
    called after the complete run, copies results to the host. This monitor
    observes material motion; it never changes the cloth or attaches it to a hand.
    """

    def __init__(self, model: newton.Model, state: newton.State, controller: _Controller) -> None:
        device = controller.device
        count = controller.num_lanes
        self._state = state
        self._body_poses = controller._scene_body_poses
        self._states = controller._phases
        self._hands = controller._scene_hands
        self._offsets = wp.array(controller.hand_offsets, dtype=wp.vec3, device=device)
        self._mesh = wp.Mesh(points=wp.clone(state.particle_q), indices=model.tri_indices.flatten())
        self._faces = wp.full((count, 2), -1, dtype=int, device=device)
        self._coordinates = wp.zeros((count, 2), dtype=wp.vec2, device=device)
        self._initial_points = wp.zeros((count, 2), dtype=wp.vec3, device=device)
        self._initial_gaps = wp.full((count, 2), 1.0, dtype=float, device=device)
        self._maximum_gaps = wp.zeros((count, 2), dtype=float, device=device)
        self._travel = wp.zeros((count, 2), dtype=float, device=device)

    def compute(self) -> None:
        """Refit the cloth surface and measure acquisition, retention, and withdrawal."""
        wp.copy(self._mesh.points, self._state.particle_q)
        self._mesh.refit()
        wp.launch(
            _measure_grasp,
            dim=self._faces.shape,
            inputs=[
                self._mesh.id,
                self._body_poses,
                self._hands,
                self._offsets,
                self._states,
                self._faces,
                self._coordinates,
                self._initial_points,
                self._initial_gaps,
                self._maximum_gaps,
                self._travel,
            ],
            device=self._faces.device,
        )

    def report(self) -> list[dict[str, object]]:
        """Return end-of-run physical grip results, in lane order."""
        initial = self._initial_gaps.numpy()
        maximum = self._maximum_gaps.numpy()
        travel = self._travel.numpy()
        return [
            {
                "retained": bool((gap < 0.005).all() and (peak < 0.015).all() and (distance > 0.30).all()),
                "initial_gap_m": gap.tolist(),
                "maximum_gap_m": peak.tolist(),
                "travel_m": distance.tolist(),
            }
            for gap, peak, distance in zip(initial, maximum, travel, strict=True)
        ]


def main() -> None:
    """Construct the scene, capture GPU control, and run the complete comparison."""
    speeds = tuple(args_cli.speeds)
    step_dt = 1.0 / FPS
    # Resolve the native SHARPA pinch during the fastest pull.
    substeps = SUBSTEPS
    collision_cfg = NewtonCollisionPipelineCfg(
        broad_phase="sap",
        # Identical, separated lanes need linear capacity, not shapes * particles.
        soft_contact_max=65_536 * len(speeds),
        soft_contact_margin=0.008,
        enable_rigid_soft_full_surface_contact=True,
    )
    solver_cfg = VBDSolverCfg(
        iterations=10,
        friction_epsilon=1.0e-3,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e7,
        rigid_joint_angular_ke=1.0e6,
        rigid_body_contact_buffer_size=256,
        rigid_body_particle_contact_buffer_size=8192,
    )
    # Frame up to three trials at 4.6 m with a 27 mm lens; stations are 1.5 m apart.
    # Wider rows use 20 mm and 1.16 m of camera distance per trial.
    camera_distance = max(4.6, 1.16 * len(speeds))
    sim_cfg = sim_utils.SimulationCfg(
        dt=step_dt / substeps,
        render_interval=substeps,
        physics=NewtonCfg(
            num_substeps=1,
            collision_decimation=0,
            default_shape_cfg=NewtonShapeCfg(gap=RIGID_GAP, ke=1.0e4, kd=1.0e1, mu=0.35),
            soft_contact_cfg=NewtonSoftContactCfg(
                soft_contact_ke=1.0e3,
                soft_contact_kd=1.0e-2,
                soft_contact_mu=CLOTH_FRICTION,
            ),
            collision_cfg=collision_cfg,
            solver_cfg=solver_cfg,
        ),
        visualizer_cfgs=VisualizerCfg(
            eye=(camera_distance, -1.5, 2.1),
            lookat=(-0.25, 0.0, 0.95),
            focal_length=27.0 if len(speeds) <= 3 else 20.0,
        ),
    )
    viewer_scene_cfg = InteractiveSceneCfg(num_envs=len(speeds), env_spacing=1.5)
    with launch_simulation(cfg={"sim": sim_cfg, "scene": viewer_scene_cfg}, launcher_args=args_cli):
        # Kit must select its OpenUSD runtime before importing native scene and Newton code.
        from isaaclab_newton.cloner import newton_builder_world_hook  # noqa: PLC0415

        from pxr import UsdGeom  # noqa: PLC0415

        from isaaclab.scene import InteractiveScene  # noqa: PLC0415

        usd_path = retrieve_file_path(H2_PLUS_USD)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(InteractiveSceneCfg(num_envs=len(speeds), env_spacing=1.5))
        positions = np.zeros((len(speeds), 3), dtype=np.float32)
        positions[:, 1] = (np.arange(len(speeds), dtype=np.float32) - 0.5 * (len(speeds) - 1)) * 1.5
        # Each trial has its own physics world, but their visual origins form one row.
        sim_utils.standardize_xform_ops(
            scene.stage.GetPrimAtPath(scene.env_prim_paths[0]), translation=tuple(map(float, positions[0]))
        )
        cfgs = {name: clone(cfg) for name, cfg in TABLECLOTH_ASSETS.items()}
        # Include cameras declared by launch in the same topology as the manually constructed assets.
        cfgs.update({name: cfg for name, cfg in vars(viewer_scene_cfg).items() if isinstance(cfg, CameraCfg)})
        cfgs["robot"].spawn.usd_path = usd_path
        plan = cloner.clone_plan_from_env_0(scene.cfg.clone_cfg, cfgs.values(), len(speeds), 1.5, positions=positions)
        scene.articulations["robot"] = instantiate(cfgs["robot"])
        _configure_h2_contacts(f"{scene.env_prim_paths[0]}/Robot", scene.stage)
        scene.deformable_objects["cloth"] = instantiate(cfgs["cloth"])
        # Render subdivision does not refine the VBD physics mesh.
        for mesh in sim_utils.get_all_matching_child_prims(
            f"{scene.env_prim_paths[0]}/Cloth", predicate=lambda prim: prim.IsA(UsdGeom.Mesh), stage=scene.stage
        ):
            UsdGeom.Mesh(mesh).GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
        for name in ("bowl", "glass", "fork"):
            scene.rigid_objects[name] = instantiate(cfgs[name])
        for name, cfg in cfgs.items():
            if isinstance(cfg, CameraCfg):
                scene.sensors[name] = instantiate(cfg)
            elif name not in ("robot", "cloth", "bowl", "glass", "fork"):
                scene.extras[name] = instantiate(cfg)
        contact_hook = partial(_prepare_contacts, device=sim.device, mesh_sdf_cache={})
        with newton_builder_world_hook(contact_hook):
            cloner.replicate(plan)
        sim.reset()
        robot = scene["robot"]
        robot.write_joint_state_to_sim_index(
            position=robot.data.default_joint_pos.warp,
            velocity=robot.data.default_joint_vel.warp,
        )
        # Fixed roots already have the cloned row origins; defaults are environment-local.
        sim.forward()
        controller = _Controller(
            robot,
            usd_path,
            step_dt,
            speeds,
            state=sim.backend_manager.backend.state_0,
            control=sim.backend_manager.backend.control,
            substeps=substeps,
        )
        sim.backend_manager.activate_newton_actuator_path()
        sim.backend_manager.register_post_actuator_callback(controller.interpolate_arm_targets)
        sim.backend_manager.set_decimation(substeps)
        sim.backend_manager.register_post_step_callback(controller.compute)
        grasp = _GraspMonitor(sim.backend_manager.backend.model, sim.backend_manager.backend.state_0, controller)
        sim.backend_manager.register_post_step_callback(grasp.compute)
        # Include per-phase frame rounding, the slowest pull and a one-second hold.
        duration = sum(H2_PHASE_DURATIONS) + len(H2_PHASE_DURATIONS) * step_dt
        duration += PULL_TIME_FACTOR * H2_PULL_DISTANCE / min(speeds)
        duration += 1.0
        complete_steps = math.ceil(duration / step_dt)
        max_steps = args_cli.max_steps or complete_steps
        logger.info("GPU H2 tablecloth pulls ready: %s m/s.", speeds)
        try:
            step = 0
            while sim.is_running() and (max_steps < 0 or step < max_steps):
                sim.step()
                scene.update(step_dt)
                step += 1
            if step >= complete_steps:
                reports = grasp.report()
                logger.info("Physical cloth grips: %s", reports)
                if not all(report["retained"] for report in reports):
                    raise RuntimeError("A robot lost its physical cloth grasp; see the material-point gaps above.")
        finally:
            sim.backend_manager.unregister_post_step_callback(grasp.compute)
            sim.backend_manager.unregister_post_step_callback(controller.compute)


if __name__ == "__main__":
    main()
