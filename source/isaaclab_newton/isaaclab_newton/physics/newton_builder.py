# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction of Newton model builders: Isaac Lab defaults and USD helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import warp as wp
from newton import Heightfield, Model, ModelBuilder, ShapeFlags
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolver, SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from isaaclab.utils import checked_apply

from .newton_manager_cfg import NewtonCfg, NewtonShapeCfg, NewtonSolverCfg

logger = logging.getLogger(__name__)


def inject_terrain_heightfields(
    stage: Usd.Stage, builder: ModelBuilder, *, root_paths: Sequence[str], device: str = "cpu"
) -> list[str]:
    """Replace height-field-tagged terrain colliders with Newton heightfields.

    Scans the stage for prims carrying the ``newton:heightfield:resolution`` attribute authored by
    :class:`~isaaclab.terrains.TerrainImporter`. Each collision mesh is rasterized into a
    :class:`newton.Heightfield` and added to ``builder`` as a static shape. Heightfields compile on MuJoCo roughly
    two orders of magnitude faster than the equivalent terrain mesh while colliding identically at the same
    horizontal resolution.

    Args:
        stage: The USD stage being imported.
        builder: The Newton model builder receiving the heightfield shapes.
        root_paths: Concrete subtree roots to scan.

    Returns:
        Prim paths of converted colliders, which the caller excludes from the USD import.
    """
    ignore_paths: list[str] = []
    xform_cache = UsdGeom.XformCache()
    for prim in (prim for root_path in root_paths for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))):
        attr = prim.GetAttribute("newton:heightfield:resolution")
        if not attr or not attr.HasAuthoredValue():
            continue
        resolution = float(attr.Get())
        mesh_prim = (
            prim if prim.IsA(UsdGeom.Mesh) else next((p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)), None)
        )
        if mesh_prim is None:
            continue
        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        # Transform vertices into world frame (USD uses row-vector convention).
        mat = np.array(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64).reshape(4, 4)
        world = (points @ mat[:3, :3] + mat[3, :3]).astype(np.float32)
        wp_mesh = wp.Mesh(
            points=wp.array(world, dtype=wp.vec3, device=device),
            indices=wp.array(faces, dtype=wp.int32, device=device),
        )
        heightfield, xform = Heightfield.create_from_mesh(wp_mesh, resolution)
        # The mesh path identifies the collider that the heightfield replaces.
        builder.add_shape_heightfield(heightfield=heightfield, xform=xform, label=str(mesh_prim.GetPath()))
        logger.info(
            "Converted terrain collider %s (%d faces) to a %dx%d heightfield.",
            prim.GetPath().pathString,
            faces.shape[0] // 3,
            heightfield.nrow,
            heightfield.ncol,
        )
        ignore_paths.append(prim.GetPath().pathString)
    return ignore_paths


def import_builder_lights(
    builder: ModelBuilder, stage: Usd.Stage, *, global_paths: Sequence[str], source_paths: Sequence[str]
) -> None:
    """Translate static global USD lights into Newton camera inputs before model finalization.

    Newton exposes uniform ambient lighting and one directional light. Plain dome lights add to the ambient
    radiance; a distant light supplies its world-space emission direction and linear RGB radiance. Intensity and
    exposure multiply color directly. Textures, area lights, light linking and per-world lights are unsupported
    and reported rather than approximated. Unsupported lights retain the native fallback; supported lights
    authored invisible or with zero intensity intentionally suppress it. Native ONCE attributes keep illumination
    independent of the number of cloned worlds and retain it across hard resets without modifying Newton.
    """
    ambient = np.zeros(3)
    distant_color = np.zeros(3)
    distant_direction = np.zeros(3)
    authored = False
    shadows = True
    xforms = UsdGeom.XformCache()
    seen = set()
    global_roots = tuple(map(Sdf.Path, global_paths))
    for root_path in (*global_paths, *source_paths):
        root = stage.GetPrimAtPath(root_path)
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if prim.GetPath() in seen or not prim.HasAPI(UsdLux.LightAPI):
                continue
            seen.add(prim.GetPath())
            light = UsdLux.LightAPI(prim)
            if not any(prim.GetPath().HasPrefix(path) for path in global_roots):
                logger.warning("Newton cameras do not support per-world light %s; it is ignored.", prim.GetPath())
                continue
            if not prim.IsA(UsdLux.DomeLight) and not prim.IsA(UsdLux.DistantLight):
                logger.warning(
                    "Newton cameras do not support %s at %s; it is ignored.", prim.GetTypeName(), prim.GetPath()
                )
                continue
            texture = UsdLux.DomeLight(prim).GetTextureFileAttr().Get() if prim.IsA(UsdLux.DomeLight) else None
            if texture and texture.path:
                logger.warning("Newton cameras do not support textured dome %s; it is ignored.", prim.GetPath())
                continue
            collections = (light.GetLightLinkCollectionAPI(), light.GetShadowLinkCollectionAPI())
            if any(links.GetIncludesRel().GetTargets() or links.GetExcludesRel().GetTargets() for links in collections):
                logger.warning("Newton cameras do not support light linking at %s; it is ignored.", prim.GetPath())
                continue
            authored = True
            if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            color = np.asarray(light.GetColorAttr().Get(), dtype=float)
            if light.GetEnableColorTemperatureAttr().Get():
                color *= np.asarray(UsdLux.BlackbodyTemperatureAsRgb(light.GetColorTemperatureAttr().Get()))
            color *= light.GetIntensityAttr().Get() * 2.0 ** light.GetExposureAttr().Get()
            if prim.IsA(UsdLux.DomeLight):
                ambient += color
            else:
                if np.any(distant_direction):
                    raise ValueError("Newton cameras support only one global distant light.")
                direction = xforms.GetLocalToWorldTransform(prim).TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
                distant_direction[:] = direction
                distant_color[:] = color
                shadow_attr = UsdLux.ShadowAPI(prim).GetShadowEnableAttr()
                shadows = bool(shadow_attr.Get()) if shadow_attr else True
    for name, dtype, value in (
        ("authored", wp.bool, authored),
        ("ambient", wp.vec3, wp.vec3(*ambient)),
        ("distant_color", wp.vec3, wp.vec3(*distant_color)),
        ("distant_direction", wp.vec3, wp.vec3(*distant_direction)),
        ("shadows", wp.bool, shadows),
    ):
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                namespace="isaaclab_lighting",
                dtype=dtype,
                frequency=Model.AttributeFrequency.ONCE,
                values={0: value},
            )
        )


def register_builder_attributes(builder: ModelBuilder, cfg: NewtonSolverCfg | None) -> None:
    """Register the custom attributes the configured solver reads on a builder that receives imported assets."""
    for solver_cls in dict.fromkeys(cfg.class_type.solver_classes(cfg) if cfg is not None else ()):
        solver_cls.register_custom_attributes(builder)


def usd_schema_resolvers(cfg: NewtonSolverCfg | None) -> list[SchemaResolver]:
    """Return ordered schema resolvers for physics-model USD imports.

    MJC is enabled when the configured solver registers ``SolverMuJoCo`` attributes. Visualization and
    articulation-ordering builders keep their fixed pair because solver attributes do not affect their outputs.
    """
    resolvers: list[SchemaResolver] = [SchemaResolverNewton(), SchemaResolverPhysx()]
    if cfg is not None and any(
        issubclass(solver_cls, SolverMuJoCo) for solver_cls in cfg.class_type.solver_classes(cfg)
    ):
        resolvers.append(SchemaResolverMjc())
    return resolvers


def create_builder(physics_cfg: NewtonCfg | None = None, up_axis: str | None = None, **kwargs) -> ModelBuilder:
    """Create a :class:`ModelBuilder` configured with Isaac Lab's defaults.

    Forwards :class:`NewtonShapeCfg` defaults onto Newton's upstream ``ModelBuilder.default_shape_cfg`` via
    :func:`~isaaclab.utils.checked_apply`, and registers the custom attributes the configured solver reads. Without a
    Newton configuration the default shape settings still apply, so rough-terrain margin and gap hold during early
    construction.

    Args:
        physics_cfg: Newton physics configuration; ``None`` uses the default shape settings and no solver attributes.
        up_axis: Override for the up-axis. Defaults to ``"Z"``.
        **kwargs: Forwarded to :class:`ModelBuilder`.

    Returns:
        New builder with up-axis and per-shape defaults (gap, margin) applied.
    """
    newton_cfg = physics_cfg if isinstance(physics_cfg, NewtonCfg) else None
    builder = ModelBuilder(up_axis=up_axis or "Z", **kwargs)
    builder.default_bvh_cfg = ModelBuilder.BvhConfig(
        mesh_constructor=newton_cfg.bvh_constructor_geometry if newton_cfg else None,
        gaussian_constructor=newton_cfg.bvh_constructor_gaussian if newton_cfg else None,
        shape_constructor=newton_cfg.bvh_constructor_scene if newton_cfg else None,
        shape_flags=ShapeFlags.VISIBLE,
    )
    register_builder_attributes(builder, newton_cfg.solver_cfg if newton_cfg else None)
    checked_apply(newton_cfg.default_shape_cfg if newton_cfg else NewtonShapeCfg(), builder.default_shape_cfg)
    return builder
