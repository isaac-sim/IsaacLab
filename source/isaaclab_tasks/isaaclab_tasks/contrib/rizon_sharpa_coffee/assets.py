# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Hosted coffee props, kitchen preparation and dimensions shared by scene consumers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

if TYPE_CHECKING:
    from pxr import Usd

    from .demo_env_cfg import CoffeeKitchenCfg

COFFEE_ASSET_DIR = os.environ.get(
    "ISAACLAB_COFFEE_ASSET_ROOT", f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaCoffee"
).rstrip("/")
"""USD bundle root; override to use an internal development release or local bundle."""

TABLE_TOP_Z = 0.0
BOWL_POSITION = (0.52, -0.12, TABLE_TOP_Z)
CUP_POSITION = (0.60, 0.02, TABLE_TOP_Z)
BOWL_INNER_RADIUS = 0.065
BOWL_HEIGHT = 0.040
BOWL_BOTTOM_THICKNESS = 0.004
CUP_INNER_RADIUS = 0.039
CUP_HEIGHT = 0.056
CUP_BOTTOM_THICKNESS = 0.004
SPOON_BOWL_LENGTH = 0.050
SPOON_BOWL_WIDTH = 0.032
COFFEE_HEAP_APEX_HEIGHT = 0.080

# Ordered from the flat cavity bottom to its upper rim. This profile also owns
# particle seeding clearance, so changes to the bowl do not invalidate its fill.
_BOWL_INNER_PROFILE = (
    (0.038, BOWL_BOTTOM_THICKNESS),
    (0.043, 0.006),
    (0.049, 0.012),
    (0.056, 0.022),
    (0.062, 0.032),
    (BOWL_INNER_RADIUS, BOWL_HEIGHT - 0.001),
    (0.066, BOWL_HEIGHT),
)


def spawn_kitchen(
    prim_path: str,
    cfg: CoffeeKitchenCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn the hosted kitchen as static geometry using its authored placement.

    The kitchen's authored transform aligns it with the coffee table. Appliance bodies and joints
    are removed so rigid backends can import the same geometry as USD renderers. Malformed normals
    metadata is repaired without changing normals or materials. The scene's studio floor is hidden
    only within each spawned kitchen's parent, leaving other environments untouched.

    Args:
        prim_path: Destination prim path or environment path expression.
        cfg: Kitchen source and optional lighting and ceiling overrides.
        translation: Accepted by the spawning interface; the kitchen retains its authored translation [m].
        orientation: Accepted by the spawning interface; the kitchen retains its authored XYZW orientation.
        **kwargs: Additional options forwarded to the USD file spawner.

    Returns:
        The first spawned kitchen prim.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    kitchen = sim_utils.spawn_from_usd(prim_path, cfg, translation=None, orientation=None, **kwargs)
    stage = kitchen.GetStage()
    for root in sim_utils.find_matching_prims(prim_path, stage=stage):
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        for prim in prims:
            disable = (
                prim.IsA(UsdGeom.Camera)
                or prim.IsA(UsdPhysics.Joint)
                or (prim.IsA(UsdGeom.Curves) and prim.GetName().startswith("grasp_identifier_curve"))
                or (cfg.disable_lights and prim.HasAPI(UsdLux.LightAPI))
                or (cfg.remove_ceiling and prim.GetPath() == root.GetPath().AppendPath("KitchenRoom/Ceiling"))
            )
            normals = prim.GetAttribute("primvars:normals")
            repair_normals = (
                prim.IsA(UsdGeom.Mesh) and normals and normals.GetTypeName() == Sdf.ValueTypeNames.TokenArray
            )
            rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsInstanceProxy() and (disable or rigid_body or repair_normals):
                raise ValueError(f"Kitchen prim {prim.GetPath()} is inside an instance and cannot be prepared.")
            if disable:
                prim.SetActive(False)
            if rigid_body:
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if repair_normals:
                (interpolation,) = normals.Get()
                UsdGeom.Mesh(prim).SetNormalsInterpolation(interpolation)
                normals.Block()
        for prim in Usd.PrimRange(root.GetParent()):
            if prim.GetName() == "StudioFloor":
                UsdGeom.Imageable(prim).MakeInvisible()
    return kitchen


def coffee_cavity_radius(height: float | np.ndarray) -> float | np.ndarray:
    """Return the circular seed radius through the cavity and an above-rim heap.

    The circular profile fits the centered oval cavity below its rim.
    Above the rim it tapers to zero at ``COFFEE_HEAP_APEX_HEIGHT`` without adding
    collision walls. Seeders should subtract their particle radius and collider
    margin from this radius. Heights outside the profile return zero.
    """
    profile = np.asarray((*_BOWL_INNER_PROFILE, (0.0, COFFEE_HEAP_APEX_HEIGHT)))
    radius = np.interp(height, profile[:, 1], profile[:, 0], left=0.0, right=0.0)
    return float(radius) if np.isscalar(height) else radius
