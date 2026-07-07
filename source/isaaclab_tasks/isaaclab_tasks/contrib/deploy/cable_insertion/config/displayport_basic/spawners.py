# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom spawner for DisplayPort USD assets.

This module is intentionally separate from the env cfg to avoid importing
``pxr`` before ``SimulationApp`` starts.  It is referenced via a string
in the ``UsdFileCfg.func`` field and only loaded at scene-creation time.
"""

from __future__ import annotations

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.assets import check_file_path, retrieve_file_path


def _deinstance(stage, root_path: str):
    """Make instanced geometry under ``root_path`` concrete (editable).

    Some simready assets (e.g. the DisplayPort plug) ship their geometry as USD
    instances. Instanced prims are shared, read-only proxies that
    ``Usd.PrimRange`` does not descend into, so the per-mesh physics authoring
    below would silently skip them and the asset would keep its source-authored
    collider (convexHull). Setting ``instanceable = False`` de-instances the
    subtree so every mesh becomes a concrete prim we can author on. Runs in
    bounded passes to handle nested instances.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    for _ in range(8):
        instances = [prim for prim in Usd.PrimRange(root_prim) if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)


def _strip_child_rigid_bodies(stage, root_path: str):
    """Remove RigidBodyAPI from every descendant prim so only the root carries it."""
    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if prim.GetPath() == root_prim.GetPath():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)


def _remove_embedded_physics_scenes(stage, root_path: str):
    """Remove any PhysicsScene prims embedded inside the asset.

    Simready assets may ship with their own PhysicsScene which conflicts
    with the scene created by Isaac Lab.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    to_remove = []
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() == "PhysicsScene":
            to_remove.append(str(prim.GetPath()))
    for path in to_remove:
        stage.RemovePrim(path)


def _disable_internal_collision_bodies(stage, root_path: str):
    """Disable collision on internal socket bodies that block insertion.

    Body4 (inner frame) and Body5 (cavity tongue) require sub-millimeter
    alignment precision. Disabling them allows the outer housing (Body8)
    to provide the primary insertion constraint while we iterate on alignment.
    """
    DISABLE_BODIES = {"Body4", "Body5"}
    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        parent_name = prim.GetParent().GetName() if prim.GetParent() else ""
        if parent_name in DISABLE_BODIES and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.GetAttribute("physics:collisionEnabled").Set(False)


def _apply_mesh_collision_to_meshes(stage, root_path: str, mesh_collision_cfg):
    """Apply a mesh collision approximation to meshes that already have CollisionAPI.

    Only modifies meshes that were explicitly marked for collision in the
    original USD. This avoids accidentally adding collision to large-scale
    visual duplicate meshes present in simready assets.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        prim_path = str(prim.GetPath())
        schemas.define_mesh_collision_properties(prim_path, mesh_collision_cfg, stage)


def _apply_collision_props_recursive(stage, root_path: str, collision_props):
    """Apply collision properties to every prim that has CollisionAPI."""
    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            schemas.modify_collision_properties(str(prim.GetPath()), collision_props)


@clone
def spawn_usd_with_physics(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Load a DisplayPort USD and set up physics APIs.

    Handles both preprocessed (fixed) and simready assets by:
    1. Removing embedded PhysicsScene prims
    2. Stripping child RigidBodyAPI so only the root carries it
    3. Applying RigidBody + Mass on root
    4. Applying collision properties recursively to all collision prims
    5. Choosing a mesh collision approximation per body: kinematic (fixed)
       bodies get an exact triangle-mesh collider; dynamic bodies get SDF
    """
    usd_path = cfg.usd_path
    file_status = check_file_path(usd_path)
    if file_status == 0:
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    if file_status == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)

    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )

    # De-instance first so all subsequent authoring reaches instanced geometry
    # (the plug ships its meshes as USD instances).
    _deinstance(stage, prim_path)

    _remove_embedded_physics_scenes(stage, prim_path)
    _strip_child_rigid_bodies(stage, prim_path)

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)

    # If no mesh in the asset has CollisionAPI, add it to all meshes
    # (handles preprocessed/fixed USDs with no collision setup).
    # If some meshes already have it (simready), only those get a collider.
    root_prim = stage.GetPrimAtPath(prim_path)
    has_any_collision = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(root_prim) if p.IsA(UsdGeom.Mesh))
    if not has_any_collision:
        for p in Usd.PrimRange(root_prim):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)

    # Pick the mesh collision approximation per body.
    #
    # Socket (kinematic): exact triangle-mesh preserves the concave receptacle
    # cavity (a convex hull would fill the opening). Triangle mesh also does not
    # require a watertight surface, which the converted CAD meshes are not.
    #
    # Plug (dynamic): convexDecomposition. ConvexHull fills the blade shroud recess
    # solid (spans 92% of bbox) and rams the socket tongue (Body5), ejecting the
    # plug. SDF at 512 resolution is leaky on the 0.4 mm shroud walls, causing the
    # plug to float or fall through. ConvexDecomposition gives solid convex pieces
    # (reliable contact) while preserving the gross concavity of the blade recess
    # and overmold underside so they mate correctly with the socket.
    is_kinematic = cfg.rigid_props is not None and bool(cfg.rigid_props.kinematic_enabled)
    if is_kinematic:
        mesh_collision_cfg = schemas_cfg.TriangleMeshPropertiesCfg()
    else:
        # convexDecomposition: solid convex pieces give reliable (non-leaky)
        # contact, while the decomposition preserves the gross concavity (blade
        # shroud recess + overmold underside recess) that the socket tongue and
        # top face must tuck into. High voxel resolution + small min_thickness so
        # the 0.4 mm shroud walls and recesses are not merged solid.
        mesh_collision_cfg = schemas_cfg.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=128,
            voxel_resolution=1_000_000,
            min_thickness=0.0001,
            hull_vertex_limit=64,
            shrink_wrap=True,
        )
    _apply_mesh_collision_to_meshes(stage, prim_path, mesh_collision_cfg)

    if cfg.collision_props is not None:
        _apply_collision_props_recursive(stage, prim_path, cfg.collision_props)

    return stage.GetPrimAtPath(prim_path)
