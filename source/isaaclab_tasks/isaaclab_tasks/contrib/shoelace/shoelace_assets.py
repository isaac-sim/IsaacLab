# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD-level overrides for the example's existing material configuration knobs."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg, spawn_physics_material
from isaaclab.utils import configclass

from . import shoelace_constants as constants


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


def rigid_material(friction: float, damping: float) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Build standard-friction and Newton-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0),
        NewtonMaterialCfg(contact_stiffness=constants.CONTACT_KE, contact_damping=damping),
    ]


def ground_asset_cfg(size: float) -> AssetBaseCfg:
    """Build the shared square ground plane with side length ``size`` [m]."""
    return AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.08, 0.08, 0.08),
            size=(size, size),
            physics_material=rigid_material(constants.GROUND_MU, constants.GROUND_CONTACT_KD),
        ),
        collision_group=-1,
    )


def light_asset_cfg() -> AssetBaseCfg:
    """Build the shared dome light."""
    return AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 1.0)),
    )


def ground_size(num_envs: int, env_spacing: float) -> float:
    """Return a ground-plane side length [m] covering the centered environment grid."""
    return max(2.0, env_spacing * (math.ceil(math.sqrt(num_envs)) + 1))
