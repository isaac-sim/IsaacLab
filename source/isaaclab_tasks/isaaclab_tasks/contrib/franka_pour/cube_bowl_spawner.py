# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural USD spawner for Franka pour cube bowls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import SchemaFragment

from .cube_bowl_mesh import make_cube_bowl_mesh

if TYPE_CHECKING:
    from .cube_bowl_spawner_cfg import CubeBowlSpawnerCfg


def _fragments(value: object) -> list[SchemaFragment] | None:
    """Return a normalized schema-fragment list, or None for a legacy config."""
    if isinstance(value, SchemaFragment):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(fragment, SchemaFragment) for fragment in value):
        return list(value)
    return None


def _apply_collision_properties(prim_path: str, properties: object, stage: Usd.Stage) -> None:
    """Apply legacy or fragment collision properties to a prim."""
    fragments = _fragments(properties)
    if fragments is None:
        sim_utils.define_collision_properties(prim_path, properties, stage=stage)
    else:
        sim_utils.apply_collision_properties(prim_path, fragments, stage=stage)


def _apply_mass_properties(prim_path: str, properties: object, stage: Usd.Stage) -> None:
    """Apply legacy or fragment mass properties to a prim."""
    fragments = _fragments(properties)
    if fragments is None:
        sim_utils.define_mass_properties(prim_path, properties, stage=stage)
    else:
        sim_utils.apply_mass_properties(prim_path, fragments, stage=stage)


def _apply_rigid_body_properties(prim_path: str, properties: object, stage: Usd.Stage) -> None:
    """Apply legacy or fragment rigid-body properties to a prim."""
    fragments = _fragments(properties)
    if fragments is None:
        sim_utils.define_rigid_body_properties(prim_path, properties, stage=stage)
    else:
        sim_utils.apply_rigid_body_properties(prim_path, fragments, stage=stage)


@sim_utils.clone
def spawn_cube_bowl(
    prim_path: str,
    cfg: CubeBowlSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn a visual hollow cube bowl with an optional grasp collider.

    Args:
        prim_path: Absolute USD path for the bowl root.
        cfg: Bowl geometry and rigid-object configuration.
        translation: Local translation relative to the parent [m]. Defaults to the origin.
        orientation: Local quaternion in ``(x, y, z, w)`` order. Defaults to identity.
        **kwargs: Additional clone-spawner options.

    Returns:
        The spawned root Xform prim.

    Raises:
        ValueError: If a prim already exists at ``prim_path``.
    """
    del kwargs
    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    root_prim = sim_utils.create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )
    geometry_path = f"{prim_path}/geometry"
    mesh_path = f"{geometry_path}/mesh"
    UsdGeom.Xform.Define(stage, geometry_path)

    vertices, indices = make_cube_bowl_mesh(
        inner_width=cfg.inner_width,
        inner_depth=cfg.inner_depth,
        cavity_depth=cfg.cavity_depth,
        wall_thickness=cfg.wall_thickness,
        bottom_thickness=cfg.bottom_thickness,
    )
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*(float(value) for value in point)) for point in vertices])
    mesh.CreateFaceVertexIndicesAttr().Set(indices.tolist())
    mesh.CreateFaceVertexCountsAttr().Set([3] * (indices.size // 3))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*(float(value) for value in vertices.min(axis=0))),
            Gf.Vec3f(*(float(value) for value in vertices.max(axis=0))),
        ]
    )
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*cfg.display_color)])

    visual_material_path = f"{geometry_path}/visual_material"
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.display_color)
    visual_material.func(visual_material_path, visual_material)
    sim_utils.bind_visual_material(mesh_path, visual_material_path, stage=stage)

    grasp_proxy_prim: Usd.Prim | None = None
    if cfg.grasp_proxy_half_extents is not None:
        half_x, half_y, half_z = cfg.grasp_proxy_half_extents
        grasp_proxy_path = f"{geometry_path}/grasp_proxy"
        grasp_proxy_prim = sim_utils.create_prim(
            grasp_proxy_path,
            prim_type="Cube",
            translation=(0.0, 0.0, half_z),
            scale=(2.0 * half_x, 2.0 * half_y, 2.0 * half_z),
            attributes={
                "size": 1.0,
                "extent": [Gf.Vec3f(-0.5), Gf.Vec3f(0.5)],
            },
            stage=stage,
        )
        UsdGeom.Imageable(grasp_proxy_prim).MakeInvisible()
        if cfg.collision_props is None:
            UsdPhysics.CollisionAPI.Apply(grasp_proxy_prim)
        else:
            _apply_collision_properties(grasp_proxy_path, cfg.collision_props, stage)

    if cfg.physics_material is not None:
        if cfg.physics_material_path.startswith("/"):
            physics_material_path = cfg.physics_material_path
        else:
            physics_material_path = f"{geometry_path}/{cfg.physics_material_path}"
        cfg.physics_material.func(physics_material_path, cfg.physics_material)
        if grasp_proxy_prim is not None:
            sim_utils.bind_physics_material(grasp_proxy_prim.GetPath(), physics_material_path, stage=stage)
        else:
            material = UsdShade.Material(stage.GetPrimAtPath(physics_material_path))
            binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
            binding_api.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

    if cfg.mass_props is not None:
        _apply_mass_properties(prim_path, cfg.mass_props, stage)
    if cfg.rigid_props is not None:
        _apply_rigid_body_properties(prim_path, cfg.rigid_props, stage)

    return root_prim
