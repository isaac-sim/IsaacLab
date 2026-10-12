# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assets for the experimental bimanual tablecloth example.

USD visuals use inexpensive primitive collision proxies. Robot settings are
specific to this fixed-base VBD demonstration, not general-purpose robot presets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab_newton.sim.schemas import (
    NewtonArticulationCfg,
    NewtonCollisionCfg,
    NewtonDeformableBodyPropertiesCfg,
    NewtonSDFCollisionCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg, NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    import numpy as np
    from newton import Mesh, ModelBuilder

    from pxr import Usd

RIGID_GAP = 0.001
CLOTH_FRICTION = 0.20
GRIP_FRICTION = 1.2
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


def rigid_material(
    *, friction: float, density: float | None = None, contact_stiffness: float = 1.0e4, contact_damping: float = 1.0e1
) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Return solver-common friction and Newton compliant-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
            density=density,
        ),
        NewtonMaterialCfg(contact_stiffness=contact_stiffness, contact_damping=contact_damping),
    ]


def collision_properties(
    *, contact_margin: float = 0.002, contact_gap: float = RIGID_GAP
) -> list[UsdPhysicsCollisionCfg | NewtonCollisionCfg]:
    """Return collision properties shared by cloth-contacting rigid shapes."""
    return [
        UsdPhysicsCollisionCfg(collision_enabled=True),
        NewtonCollisionCfg(contact_margin=contact_margin, contact_gap=contact_gap),
    ]


def tabletop_collider_cfg(
    size: tuple[float, float, float], *, friction: float, contact_stiffness: float, contact_damping: float
) -> sim_utils.CuboidCfg:
    """Create a low-cost collision proxy for a visual-only table."""
    return sim_utils.CuboidCfg(
        size=size,
        visible=False,
        # Inflated rigid offsets must not support dishes above the cloth.
        collision_props=collision_properties(contact_margin=0.0002, contact_gap=0.0),
        physics_material=rigid_material(
            density=None,
            friction=friction,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
        ),
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


def spawn_visual_table_from_usd(
    prim_path: str,
    cfg: VisualTableUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn a visual-only table and remove material inputs unsupported by Newton GL."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root in sim_utils.find_matching_prims(prim_path):
        _configure_visuals(root.GetPath().pathString)
    return prim


@configclass
class VisualTableUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawner config for a visual-only dining table."""

    func: Callable | str = spawn_visual_table_from_usd


def spawn_tableware_from_usd(
    prim_path: str,
    cfg: TablewareUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn a tableware visual with configured mass and analytic collision proxies."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    from pxr import Gf, UsdPhysics  # noqa: PLC0415

    for root in sim_utils.find_matching_prims(prim_path):
        root_path = root.GetPath().pathString
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

        proxy_cfgs = (cfg.collision_proxy_cfg, *cfg.additional_collision_proxy_cfgs)
        proxy_positions = (cfg.collision_proxy_position, *cfg.additional_collision_proxy_positions)
        for index, (proxy_cfg, proxy_position) in enumerate(zip(proxy_cfgs, proxy_positions, strict=True)):
            suffix = "" if index == 0 else str(index)
            proxy_cfg.func(
                f"{body.GetPath()}/CollisionProxy{suffix}",
                proxy_cfg,
                translation=proxy_position,
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
    return prim


@configclass
class TablewareUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawner config for SimReady visuals with analytic collision proxies.

    Proxy dimensions, offsets, center of mass, and inertia are authored in the
    selected rigid body's local frame, before the USD asset's scale is applied.
    """

    func: Callable | str = spawn_tableware_from_usd
    make_uninstanceable: bool = True
    rigid_props = {"(/.*)?": [UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False)]}
    mass: float = MISSING
    """Mass of the selected rigid body [kg]."""
    collision_proxy_cfg: sim_utils.ShapeCfg = MISSING
    """Primary collision proxy; dimensions are in local meters."""
    collision_proxy_position: tuple[float, float, float] = MISSING
    """Primary collision proxy's body-local offset [m]."""
    additional_collision_proxy_cfgs: tuple[sim_utils.ShapeCfg, ...] = ()
    """Additional collision proxies, paired with their respective offsets."""
    additional_collision_proxy_positions: tuple[tuple[float, float, float], ...] = ()
    """Additional proxies' body-local offsets [m], in matching order."""
    center_of_mass: tuple[float, float, float] | None = None
    """Body-local center of mass [m], or None to retain the asset's value."""
    diagonal_inertia: tuple[float, float, float] | None = None
    """Diagonal inertia in the body-local axes [kg m²], or None to retain the asset's value."""
    visual_color: tuple[float, float, float] | None = None
    """RGB visual color in [0, 1], or None to retain the asset's color."""
    visual_opacity: float | None = None
    """Visual opacity in [0, 1], or None to retain the asset's opacity."""
    visual_roughness: float | None = None
    """Visual roughness in [0, 1], or None to retain the asset's roughness."""


FPS = 60
SUBSTEPS = 64
# Table height calibrated for the fixed-pelvis H2 stance and native fingertip pinch.
TABLE_TOP_Z = 1.0352
TABLEWARE_CLEARANCE = 0.008
# Start above the thin table contact skin and the cloth's 1 mm particle radius.
CLOTH_Z = TABLE_TOP_Z + 0.004

TABLE_SCALE = (
    TABLE_SIZE[0] / TABLE_ASSET_SIZE[0],
    TABLE_SIZE[1] / TABLE_ASSET_SIZE[1],
    TABLE_TOP_Z / TABLE_ASSET_SIZE[2],
)
TABLETOP_THICKNESS = TABLE_ASSET_TOP_THICKNESS * TABLE_SCALE[2]

H2_HAND_OFFSETS = (
    (0.0854352130714941, -0.004956624352234765, 0.11747110476887199),
    (0.08528776908502764, 0.0060654248588878044, 0.11775531545430512),
)
# Measured native pulp pinch and unused-finger parking, inside native joint limits.
H2_PINCH_JOINT_POS = {
    "left_thumb_CMC_FE": 1.8898998498916626,
    "left_thumb_CMC_AA": 0.31909996271133423,
    "left_thumb_MCP_FE": -0.49359992146492004,
    "left_thumb_MCP_AA": -0.041234519332647324,
    "left_thumb_IP": 0.030000001192092896,
    "left_index_MCP_FE": 0.9650716781616211,
    "left_index_MCP_AA": 0.30174076557159424,
    "left_index_PIP": 0.5233824253082275,
    "left_index_DIP": 0.029999999329447746,
    "right_thumb_CMC_FE": 1.8898998498916626,
    "right_thumb_CMC_AA": 0.31909996271133423,
    "right_thumb_MCP_FE": -0.49359992146492004,
    "right_thumb_MCP_AA": -0.021861376240849495,
    "right_thumb_IP": 0.029999999329447746,
    "right_index_MCP_FE": 0.9501051306724548,
    "right_index_MCP_AA": 0.2907813787460327,
    "right_index_PIP": 0.5469146966934204,
    "right_index_DIP": 0.029999999329447746,
    "left_middle_MCP_FE": 0.5188307166099548,
    "left_middle_MCP_AA": 0.30825042724609375,
    "left_middle_PIP": 1.3207045793533325,
    "left_middle_DIP": 1.3419662714004517,
    "left_ring_MCP_FE": 0.10000000149011612,
    "left_ring_MCP_AA": 0.30000001192092896,
    "left_ring_PIP": 1.5499999523162842,
    "left_ring_DIP": 1.2000000476837158,
    "left_pinky_MCP_FE": 0.10000000149011612,
    "left_pinky_MCP_AA": 0.30000001192092896,
    "left_pinky_PIP": 1.6299999952316284,
    "left_pinky_DIP": 1.2000000476837158,
    "right_middle_MCP_FE": 0.4482609033584595,
    "right_middle_MCP_AA": 0.30533933639526367,
    "right_middle_PIP": 1.3855232000350952,
    "right_middle_DIP": 1.3496774435043335,
    "right_ring_MCP_FE": 0.10000000149011612,
    "right_ring_MCP_AA": 0.30000001192092896,
    "right_ring_PIP": 1.5499999523162842,
    "right_ring_DIP": 1.2000000476837158,
    "right_pinky_MCP_FE": 0.10000000149011612,
    "right_pinky_MCP_AA": 0.30000001192092896,
    "right_pinky_PIP": 1.5499999523162842,
    "right_pinky_DIP": 1.2000000476837158,
    "left_pinky_CMC": 0.10000000149011612,
    "right_pinky_CMC": 0.10000000149011612,
}
H2_OPEN_JOINT_POS = {
    **H2_PINCH_JOINT_POS,
    # The index stays stationary while the pre-unfolded thumb closes from this near-pinch reset.
    "left_thumb_CMC_FE": 1.8749698686599732,
    "left_thumb_CMC_AA": 0.27846200197935106,
    "left_thumb_MCP_AA": -0.0139111201092601,
    "right_thumb_CMC_FE": 1.8719424211978912,
    "right_thumb_CMC_AA": 0.2733340892195702,
    "right_thumb_MCP_AA": 0.0021895832382142384,
}


def _configure_robot_colliders(root_path: str, stage: Usd.Stage) -> None:
    """Give robot meshes the SDF representation required by full-surface cloth contact."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

    from isaaclab.sim.schemas import apply_namespaced  # noqa: PLC0415

    meshes = sim_utils.get_all_matching_child_prims(
        root_path,
        predicate=lambda prim: prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI),
        stage=stage,
    )
    for mesh in meshes:
        mesh.RemoveAppliedSchema("NewtonMeshCollisionAPI")
        UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Set("none")
        apply_namespaced(
            NewtonSDFCollisionCfg(sdf_max_resolution=64, sdf_padding=0.012), mesh.GetPath().pathString, stage
        )


def enable_h2_fingertip_contacts(
    builder: ModelBuilder, world_index: int, world_position: np.ndarray, world_quaternion: np.ndarray
) -> None:
    """Enable opposing native fingertip contacts, preserving other robot collision exclusions.

    Register this initialization callback with ``newton_builder_world_hook`` during scene replication.
    """
    from newton import ShapeFlags  # noqa: PLC0415

    root = f"/World/envs/env_{world_index}/Robot"
    pairs = set()
    for side in ("left", "right"):
        indices = []
        for finger in ("thumb", "index"):
            prefix = f"{root}/{side}_{finger}_DP/"
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


def prepare_native_h2_contacts(
    builder: ModelBuilder,
    world_index: int,
    world_position: np.ndarray,
    world_quaternion: np.ndarray,
    *,
    device: str,
    mesh_sdf_cache: dict[tuple, Mesh],
) -> None:
    """Prepare native H2 mesh SDFs and enable opposing native fingertip contact.

    Register with ``newton_builder_world_hook`` during replication, before model finalization.
    Unit shape scales retain the imported grid settings without changing distance units.
    Copies protect geometry shared with source templates.

    Args:
        builder: Cumulative Newton builder containing the newly appended world.
        world_index: Replication world index.
        world_position: World translation [m], supplied by the public hook.
        world_quaternion: World orientation in xyzw order, supplied by the hook.
        device: Resolved CUDA device used for subsequent model finalization.
        mesh_sdf_cache: Empty at replication entry; shared by this run's callbacks.
    """
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

    enable_h2_fingertip_contacts(builder, world_index, world_position, world_quaternion)


@configclass
class H2ArticulationCfg(ArticulationCfg):
    """Fixed-base H2 Plus with native, contact-only SHARPA fingertips."""

    def _post_spawn(self, stage: Usd.Stage) -> None:
        super()._post_spawn(stage)
        root_path = self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path else self.prim_path
        _configure_robot_colliders(root_path, stage)
        from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.schemas import apply_namespaced  # noqa: PLC0415
        from isaaclab.sim.spawners.materials.physics_materials import (  # noqa: PLC0415
            spawn_physics_material_from_fragments,
        )

        grasp_material = rigid_material(
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


H2_CFG = H2ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=H2_PLUS_USD,
        make_uninstanceable=True,
        fix_root_link=True,
        articulation_props={"/pelvis": [NewtonArticulationCfg(self_collision_enabled=False)]},
        rigid_props={"/pelvis": [UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]},
        collision_props=collision_properties(),
        physics_material=rigid_material(friction=0.50, contact_stiffness=1.0e3, contact_damping=1.0e-2),
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
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[r"(left|right)_(hip|knee|ankle|shoulder|elbow)_.*", r"(waist|head)_.*"],
            stiffness=5.0e4,
            damping=5.0e2,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[r"(left|right)_wrist_.*"],
            stiffness=None,
            damping={
                "left_wrist_roll_joint": 32.0,
                "left_wrist_pitch_joint": 21.7,
                "left_wrist_yaw_joint": 16.2,
                "right_wrist_roll_joint": 29.7,
                "right_wrist_pitch_joint": 19.4,
                "right_wrist_yaw_joint": 14.8,
            },
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[r"(left|right)_(thumb|index|middle|ring|pinky)_.*"],
            # Simulation drives only: direct VBD does not enforce native hardware torque caps.
            stiffness=400.0,
            damping=0.3,
        ),
    },
)


GROUND_CFG = AssetBaseCfg(
    prim_path="/World/defaultGroundPlane",
    spawn=sim_utils.GroundPlaneCfg(),
    collision_group=-1,
)
TABLE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=VisualTableUsdFileCfg(
        usd_path=TABLE_USD,
        scale=TABLE_SCALE,
        make_uninstanceable=True,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, 0.0)),
)
TABLETOP_COLLIDER_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Tabletop",
    spawn=tabletop_collider_cfg(
        (*TABLE_SIZE, TABLETOP_THICKNESS),
        # Newton's geometric-mean mixing gives 0.3 cloth-table friction.
        friction=0.45,
        contact_stiffness=1.0e4,
        contact_damping=1.0e1,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_TOP_Z - 0.5 * TABLETOP_THICKNESS)),
)
TABLE_SUPPORT_CFGS = {
    f"table_leg_{index}": AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/TableLeg{index}",
        spawn=tabletop_collider_cfg(
            (0.024, 0.024, TABLE_TOP_Z - TABLETOP_THICKNESS),
            friction=0.45,
            contact_stiffness=1.0e4,
            contact_damping=1.0e1,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X + x, y, 0.5 * (TABLE_TOP_Z - TABLETOP_THICKNESS))),
    )
    # Four slender legs replace the old panel proxies; no invisible wall below the hem.
    for index, (x, y) in enumerate(((-0.224, -0.403), (-0.224, 0.403), (0.224, -0.403), (0.224, 0.403)))
}


@configclass
class TableclothCfg(DeformableObjectCfg):
    """Smooth the rendered cloth without refining the physics mesh."""

    def _post_spawn(self, stage: Usd.Stage) -> None:
        super()._post_spawn(stage)
        from pxr import UsdGeom  # noqa: PLC0415

        root_path = self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path else self.prim_path
        for prim in sim_utils.get_all_matching_child_prims(
            root_path, predicate=lambda prim: prim.IsA(UsdGeom.Mesh), stage=stage
        ):
            UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)


CLOTH_CFG = TableclothCfg(
    prim_path="{ENV_REGEX_NS}/Cloth",
    spawn=sim_utils.MeshRectangleCfg(
        # A 4 cm free hem settles before the scripted hands approach.
        size=(0.46, TABLE_SIZE[1]),
        edge_refinement=64,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.05, 0.05)),
        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
            density=0.24,
            particle_radius=0.001,
            edge_ke=0.01,
            edge_kd=5.0e-3,
        ),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.01, 0.0, CLOTH_Z)),
)
BOWL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bowl",
    spawn=TablewareUsdFileCfg(
        usd_path=BOWL_USD,
        mass=0.45,
        collision_proxy_cfg=sim_utils.CylinderCfg(
            radius=BOWL_LOCAL_RADIUS,
            height=BOWL_LOCAL_HEIGHT,
            visible=False,
            collision_props=collision_properties(contact_margin=0.0002, contact_gap=0.0),
            physics_material=rigid_material(friction=0.30),
        ),
        collision_proxy_position=(0.0, 0.0, BOWL_LOCAL_Z_MIN + 0.5 * BOWL_LOCAL_HEIGHT),
        scale=(BOWL_SCALE,) * 3,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.08, -0.07, CLOTH_Z + TABLEWARE_CLEARANCE - BOWL_SCALE * BOWL_LOCAL_Z_MIN), rot=(0.0, 0.0, 0.0, 1.0)
    ),
)
GLASS_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Glass",
    spawn=TablewareUsdFileCfg(
        usd_path=WINE_GLASS_USD,
        mass=0.35,
        collision_proxy_cfg=sim_utils.CylinderCfg(
            radius=0.040,
            height=0.006,
            visible=False,
            collision_props=collision_properties(contact_margin=0.0002, contact_gap=0.0),
            physics_material=rigid_material(friction=0.15),
        ),
        collision_proxy_position=(0.0, 0.0, 0.003),
        center_of_mass=WINE_GLASS_CENTER_OF_MASS,
        diagonal_inertia=WINE_GLASS_DIAGONAL_INERTIA,
        # A base-only proxy lets the visual stem/cup penetrate the ground after a fall.
        additional_collision_proxy_cfgs=(
            sim_utils.CylinderCfg(
                radius=0.004,
                height=0.096,
                visible=False,
                collision_props=collision_properties(contact_margin=0.0002, contact_gap=0.0),
                physics_material=rigid_material(friction=0.15),
            ),
            sim_utils.CapsuleCfg(
                radius=0.041,
                height=0.045,
                visible=False,
                collision_props=collision_properties(contact_margin=0.0002, contact_gap=0.0),
                physics_material=rigid_material(friction=0.15),
            ),
        ),
        additional_collision_proxy_positions=((0.0, 0.0, 0.054), (0.0, 0.0, 0.165)),
        visual_color=(0.55, 0.75, 0.90),
        visual_opacity=0.90,
        visual_roughness=0.12,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.10, 0.13, CLOTH_Z + TABLEWARE_CLEARANCE - WINE_GLASS_LOCAL_Z_MIN), rot=(0.0, 0.0, 0.0, 1.0)
    ),
)
FORK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Fork",
    spawn=TablewareUsdFileCfg(
        usd_path=FORK_USD,
        mass=0.07,
        collision_proxy_cfg=sim_utils.CuboidCfg(
            size=(0.014, 0.160, 0.003),
            visible=False,
            collision_props=collision_properties(contact_margin=0.0002, contact_gap=0.0),
            physics_material=rigid_material(friction=0.25),
        ),
        collision_proxy_position=(0.0, 0.0092, 0.0016),
        scale=(FORK_SCALE,) * 3,
        center_of_mass=FORK_CENTER_OF_MASS,
        diagonal_inertia=FORK_DIAGONAL_INERTIA,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.12, -0.22, CLOTH_Z + TABLEWARE_CLEARANCE - FORK_SCALE * FORK_LOCAL_Z_MIN), rot=FORK_ROTATION
    ),
)
LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/Light",
    spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
)


TABLECLOTH_ASSETS = {
    "robot": H2_CFG,
    "table": TABLE_CFG,
    "tabletop": TABLETOP_COLLIDER_CFG,
    **TABLE_SUPPORT_CFGS,
    "cloth": CLOTH_CFG,
    "bowl": BOWL_CFG,
    "glass": GLASS_CFG,
    "fork": FORK_CFG,
}
