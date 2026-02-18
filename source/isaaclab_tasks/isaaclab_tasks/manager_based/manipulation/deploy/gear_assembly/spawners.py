# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom spawner for gear base: two-body fixed-base articulation for Newton."""

from __future__ import annotations

from collections.abc import Callable

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.spawners.from_files import from_files_cfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass


@clone
def spawn_gear_base_fixed(
    prim_path: str,
    cfg: "FixedGearBaseCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the gear base as a two-body fixed-base articulation.

    Newton requires an articulation with at least two rigid bodies connected
    by a joint. This spawner creates:

    .. code-block:: text

        prim_path                        (Xform + ArticulationRootAPI)
        ├── base                         (Xform + RigidBodyAPI)  ← dummy body, fixed to world
        │   └── WorldFixedJoint
        └── body                         (Xform, gear USD reference)
            └── <gear_prim>              (RigidBodyAPI from USD)  ← actual gear body
                └── FixedJoint (body0 → base, body1 → gear_prim)

    The key subtlety: the gear USD file already contains a rigid body prim
    (e.g. ``factory_gear_base``) with its own ``RigidBodyAPI``.  We must NOT
    add a second ``RigidBodyAPI`` on the intermediate ``body/`` Xform, and we
    must connect the FixedJoint to the actual rigid body prim inside the USD,
    otherwise Newton sees an orphan body and assigns it a free joint.
    """
    stage = get_current_stage()

    # 1. Create root Xform with ArticulationRootAPI
    UsdGeom.Xform.Define(stage, prim_path)
    root_prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # 2. Create dummy "base" body -- invisible rigid body fixed to world.
    #    Must have valid mass and inertia (positive eigenvalues) for MuJoCo solver,
    #    since this body has no geometry for Newton to auto-compute from.
    base_path = f"{prim_path}/base"
    base_xform = UsdGeom.Xform.Define(stage, base_path)
    base_prim = base_xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    mass_api = UsdPhysics.MassAPI.Apply(base_prim)
    mass_api.CreateMassAttr(1.0)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))

    # FixedJoint anchoring base to world (body0 = world by default)
    world_joint_path = f"{base_path}/WorldFixedJoint"
    world_joint = UsdPhysics.FixedJoint.Define(stage, world_joint_path)
    world_joint.CreateBody1Rel().SetTargets([base_prim.GetPath()])
    world_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    world_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    world_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    world_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    world_joint.CreateJointEnabledAttr().Set(True)

    # 3. Spawn the gear USD at a CHILD path so it doesn't conflict with the
    #    root prim that already exists. _spawn_from_usd_file skips the USD
    #    reference if the target prim already exists, so we must use a fresh path.
    #    NOTE: Do NOT apply RigidBodyAPI to body_path — the gear USD already
    #    contains a child prim with RigidBodyAPI (e.g. factory_gear_base).
    body_path = f"{prim_path}/body"
    _spawn_from_usd_file(body_path, cfg.usd_path, cfg, translation, orientation)

    # 4. Find the actual rigid body prim inside the spawned USD.
    #    The gear USD's default prim (e.g. factory_gear_base) has its own
    #    RigidBodyAPI. We must connect the FixedJoint to THIS prim, not to
    #    the intermediate body/ Xform.
    body_container = stage.GetPrimAtPath(body_path)
    gear_body_prim = None
    for child in Usd.PrimRange(body_container):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            gear_body_prim = child
            break

    # Fallback: if the USD has no RigidBodyAPI child, apply it to body_path itself
    if gear_body_prim is None:
        print(f"[GearSpawner] WARNING: No RigidBodyAPI child found under {body_path}, "
              f"applying RigidBodyAPI to {body_path}")
        UsdPhysics.RigidBodyAPI.Apply(body_container)
        gear_body_prim = body_container

    print(f"[GearSpawner] Found gear rigid body at: {gear_body_prim.GetPath()}")

    # 5. FixedJoint connecting the actual gear body to the dummy base
    fixed_joint_path = gear_body_prim.GetPath().AppendChild("FixedJointToBase")
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    fixed_joint.CreateBody0Rel().SetTargets([base_prim.GetPath()])
    fixed_joint.CreateBody1Rel().SetTargets([gear_body_prim.GetPath()])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateJointEnabledAttr().Set(True)

    print(f"[GearSpawner] Created two-body fixed articulation: "
          f"world -> {base_path} -> {gear_body_prim.GetPath()}")

    return root_prim


@configclass
class FixedGearBaseCfg(from_files_cfg.UsdFileCfg):
    """USD file config that creates a two-body fixed-base articulation for Newton."""

    func: Callable = spawn_gear_base_fixed
