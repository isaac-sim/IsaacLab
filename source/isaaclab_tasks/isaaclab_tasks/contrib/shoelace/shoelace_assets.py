# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD spawning and baked startup data for the shoelace task."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg, spawn_physics_material
from isaaclab.utils import configclass


@sim_utils.clone
def spawn_shoelace_usd(
    prim_path: str,
    cfg: ShoelaceUsdCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Load a USD asset and apply task material overrides before Newton imports it.

    Args:
        prim_path: Destination prim path or environment path expression.
        cfg: USD asset and material overrides.
        translation: Root translation relative to its parent [m].
        orientation: Root orientation as an xyzw quaternion.
        **kwargs: Additional arguments forwarded to the standard USD spawner.

    Returns:
        The spawned root prim.
    """
    if any(not math.isfinite(value) or value < 0.0 for value in cfg.friction_overrides.values()):
        raise ValueError("Friction overrides must be finite and nonnegative")
    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    visual_path = str(root.GetPath()) + "/Shoe/Visual"
    if root.GetStage().GetPrimAtPath(visual_path):
        # Reserve the four optional eyelet visuals before cable import. Newton views require
        # the same shape offsets and types in each world; visual-only meshes have no contacts.
        for index in range(1, 5):
            path = f"{visual_path}/ExtraEyelet_{index}"
            if not root.GetStage().GetPrimAtPath(path):
                placeholder = UsdGeom.Mesh.Define(root.GetStage(), path)
                placeholder.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0e-6, 0.0, 0.0), (0.0, 1.0e-6, 0.0)])
                placeholder.CreateFaceVertexCountsAttr([3])
                placeholder.CreateFaceVertexIndicesAttr([0, 1, 2])
                placeholder.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                placeholder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for prim in Usd.PrimRange(root):
        relative_path = str(prim.GetPath()).removeprefix(str(root.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.BasisCurves):
            for pattern, friction in cfg.friction_overrides.items():
                if not re.fullmatch(pattern, relative_path):
                    continue
                previous, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                material = UsdShade.Material.Define(root.GetStage(), str(prim.GetPath()) + "/ContactOverride")
                if previous and previous.GetPath() != material.GetPath():
                    material.GetPrim().GetReferences().AddInternalReference(previous.GetPath())
                spawn_physics_material(
                    str(material.GetPath()),
                    UsdPhysicsRigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
                    stage=root.GetStage(),
                )
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return root


@configclass
class ShoelaceUsdCfg(sim_utils.UsdFileCfg):
    """Load the baked example or Franka USD with optional task material overrides."""

    func: Callable = spawn_shoelace_usd
    friction_overrides: dict[str, float] = {}
    """Asset-relative collision-prim expressions and their friction coefficients."""


def read_shoelace_attribute(curve: Usd.Prim, name: str, shape: tuple[int, ...], quaternion: bool = False) -> np.ndarray:
    """Read and validate task-local startup data baked into a cable prim.

    Args:
        curve: Source cable geometry prim.
        name: Attribute name without the ``shoelace:`` prefix.
        shape: Expected array shape; ``()`` denotes a scalar.
        quaternion: Whether to require unit-length xyzw quaternions.

    Returns:
        Float64 values with the requested shape. Positions and lengths are in [m];
        orientations are dimensionless xyzw quaternions.

    Raises:
        ValueError: If the attribute is missing, has an unexpected shape, contains
            nonfinite values, or fails the requested quaternion validation.
    """
    value = curve.GetAttribute(f"shoelace:{name}").Get()
    if value is None:
        raise ValueError(f"{curve.GetPath()}: missing shoelace:{name}; regenerate shoelace.usda")
    values = np.asarray(value, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{curve.GetPath()}: shoelace:{name} must have finite shape {shape}")
    if quaternion and not np.allclose(np.linalg.norm(values, axis=-1), 1.0, atol=2.0e-5):
        raise ValueError(f"{curve.GetPath()}: shoelace:{name} must contain unit xyzw quaternions")
    return values
