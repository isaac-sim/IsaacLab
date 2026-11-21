# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Revision-pinned Unitree G1 USD packages retrieved through SimReady."""

from __future__ import annotations

import math
import subprocess
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import replace

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim import UsdFileCfg

G1_OEM_BARE_URI = (
    "hf://unitreerobotics/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0@323e350252b9c3aee9c40acbfdad84f6ce46a5ac"
)
G1_OEM_FLOATING_DEX3_URI = (
    "hf://unitreerobotics/unitree_sim_isaaclab_usds/assets/robots/g1-29dof_wholebody_dex3"
    "@27b13d380531c0c6ea6ab109511ee4872594e271"
)
G1_OEM_FIXED_DEX3_URI = (
    "hf://unitreerobotics/unitree_sim_isaaclab_usds/assets/robots/g1-29dof-dex3-base-fix-usd"
    "@27b13d380531c0c6ea6ab109511ee4872594e271"
)
G1_OEM_FIXED_INSPIRE_URI = (
    "hf://unitreerobotics/unitree_sim_isaaclab_usds/assets/robots/g1-29dof-inspire-base-fix-usd"
    "@27b13d380531c0c6ea6ab109511ee4872594e271"
)

G1_DEX3_HAND_CONTACT_LINKS = "(?:palm|thumb_[0-2]|(?:index|middle)_[01])"
"""Dex3 palm and finger link names, excluding the OEM package's wrist camera mounts."""

# Unitree URDF limits since unitree_ros 5ae4fa58d3; the 29-DOF package still authors 50 N*m and 37 rad/s.
_URDF_LIMIT_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)


def retrieve_g1_oem_usd(uri: str) -> str:
    """Return the primary USD of a pinned Unitree package from SimReady Retrieve's cache."""
    try:
        result = subprocess.run(["simready-retrieve", "fetch", uri], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "The G1 OEM preset requires the 'simready-retrieve' CLI on PATH. "
            "See https://nvidia.github.io/simready-foundation/latest/guides/libraries.html for installation."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"SimReady Retrieve could not fetch {uri}: {error.stderr.strip()}") from error

    path = Path(result.stdout.strip())
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"SimReady Retrieve did not return a local USD for {uri}: {result.stdout.strip()!r}")
    return str(path)


def correct_g1_oem_stage(stage: Usd.Stage) -> None:
    """Author the two upstream corrections for a Unitree G1 package into the stage's edit target.

    1. Move collision APIs from instanceable ``collisions`` Xforms onto their child gprims; UsdPhysics applies
       CollisionAPI only to gprims, so Newton and OpenUSD's parser otherwise load no colliders.
    2. Set the ankle and waist roll/pitch drive limits to the Unitree URDF values.

    Args:
        stage: Stage composing a Unitree package; edits go to its current edit target.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim()
    # 1. Collision APIs on Xforms
    xform_paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim)
    ]
    for path in xform_paths:
        xform = stage.GetPrimAtPath(path)
        approximation = UsdPhysics.MeshCollisionAPI(xform).GetApproximationAttr().Get()
        xform.SetInstanceable(False)
        for gprim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not gprim.IsA(UsdGeom.Gprim):
                continue
            UsdPhysics.CollisionAPI.Apply(gprim)
            if gprim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(gprim).CreateApproximationAttr(approximation)
        for schema in ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxMeshMergeCollisionAPI"):
            xform.RemoveAppliedSchema(schema)
    # 2. URDF drive limits
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() in _URDF_LIMIT_JOINTS and prim.IsA(UsdPhysics.RevoluteJoint):
            UsdPhysics.DriveAPI.Get(prim, "angular").GetMaxForceAttr().Set(35.0)
            prim.GetAttribute("physxJoint:maxJointVelocity").Set(math.degrees(30.0))


def get_corrected_g1_oem_usd(uri: str) -> str:
    """Return a local root that composes the upstream corrections over the pinned package."""
    return str(Path(_build_corrected_root(uri).name) / "g1_oem.usda")


@cache
def _build_corrected_root(uri: str) -> TemporaryDirectory:
    from pxr import Sdf, Usd, UsdGeom

    source = retrieve_g1_oem_usd(uri)
    layer = Sdf.Layer.CreateAnonymous("g1_oem.usda")
    layer.subLayerPaths = [source]
    layer.defaultPrim = Sdf.Layer.FindOrOpen(source).defaultPrim
    stage = Usd.Stage.Open(layer)
    source_stage = Usd.Stage.Open(source)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    correct_g1_oem_stage(stage)
    if stage.GetCompositionErrors():
        raise ValueError(f"Could not compose the corrected G1 package {uri}: {stage.GetCompositionErrors()}")

    temporary = TemporaryDirectory(prefix="isaaclab-g1-oem-")
    if not layer.Export(str(Path(temporary.name) / "g1_oem.usda")):
        raise RuntimeError(f"Could not export the corrected G1 package {uri}.")
    return temporary


def spawn_g1_oem_usd(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Use the standard USD spawner on the corrected OEM package."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    local_cfg = replace(cfg, usd_path=get_corrected_g1_oem_usd(cfg.usd_path))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)
