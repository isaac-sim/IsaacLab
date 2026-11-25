# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion collision profiles over Unitree's corrected floating Dex3 G1."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.unitree_g1_oem import get_corrected_g1_oem_usd

if TYPE_CHECKING:
    from isaaclab.sim import UsdFileCfg


LAB_G1_29DOF_USD = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd"

# Bodies whose collisions stay active in the minimal profile: contact terminations and the feet.
_MINIMAL_BODIES = ("torso_link", "head_link", "pelvis_contour_link", "left_ankle_roll_link", "right_ankle_roll_link")


def get_oem_locomotion_usd(oem_uri: str, profile: Literal["minimal", "full"]) -> str:
    """Return the composed asset used by the locomotion spawners and the Rough task's startup parameters."""
    return str(Path(_build_locomotion_root(oem_uri, profile).name) / "g1_oem_locomotion.usda")


@cache
def _build_locomotion_root(oem_uri: str, profile: Literal["minimal", "full"]) -> TemporaryDirectory:
    """Stack the locomotion sole and collision profile over the corrected OEM package.

    1. Add the sole of Unitree's 29-DOF model (a 150 x 60 x 15 mm box and a toe sphere) to both feet; the box is
       named ``foot_plate`` because the Rough rewards read its size and offset.
    2. For the minimal profile, deactivate every collision branch except the contact-termination bodies and feet.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    source = get_corrected_g1_oem_usd(oem_uri)
    layer = Sdf.Layer.CreateAnonymous("g1_oem_locomotion.usda")
    layer.subLayerPaths = [source]
    layer.defaultPrim = Sdf.Layer.FindOrOpen(source).defaultPrim
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = stage.GetDefaultPrim()

    # 1. Sole
    for side in ("left", "right"):
        body = root.GetPath().AppendChild(f"{side}_ankle_roll_link")
        plate = UsdGeom.Cube.Define(stage, body.AppendChild("foot_plate"))
        plate.CreateSizeAttr(1.0)
        plate.AddTranslateOp().Set((0.035, 0.0, -0.016))
        plate.AddScaleOp().Set((0.15, 0.06, 0.015))
        toe = UsdGeom.Sphere.Define(stage, body.AppendChild("foot_toe"))
        toe.CreateRadiusAttr(0.005)
        toe.AddTranslateOp().Set((0.14, 0.0, -0.028))
        for gprim in (plate, toe):
            UsdPhysics.CollisionAPI.Apply(gprim.GetPrim())
            gprim.CreatePurposeAttr(UsdGeom.Tokens.guide)

    # 2. Collision profile
    if profile == "minimal":
        for prim in Usd.PrimRange(root):
            if prim.GetName() == "collisions" and prim.GetParent().GetName() not in _MINIMAL_BODIES:
                stage.OverridePrim(prim.GetPath()).SetActive(False)

    if stage.GetCompositionErrors():
        raise ValueError(f"Could not compose the OEM G1 {profile} profile: {stage.GetCompositionErrors()}")
    temporary = TemporaryDirectory(prefix="isaaclab-g1-oem-locomotion-")
    if not layer.Export(str(Path(temporary.name) / "g1_oem_locomotion.usda")):
        raise RuntimeError(f"Could not export the OEM G1 {profile} profile.")
    return temporary


def spawn_g1_oem_minimal(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Spawn the corrected OEM asset with the sole and the minimal collision profile used by rough locomotion."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    local_cfg = replace(cfg, usd_path=get_oem_locomotion_usd(cfg.usd_path, "minimal"))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)


def spawn_g1_oem_full(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Spawn the corrected OEM asset with the sole and all native colliders, used by the terrain courses."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    local_cfg = replace(cfg, usd_path=get_oem_locomotion_usd(cfg.usd_path, "full"))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)
