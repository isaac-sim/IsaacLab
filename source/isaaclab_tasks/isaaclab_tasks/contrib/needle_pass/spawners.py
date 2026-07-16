# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime USD spawners for the dVRK needle-pass task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths, get_current_stage

if TYPE_CHECKING:
    from .needle_pass_env_cfg import UsdFileWithRigidMaterialCfg


def spawn_usd_with_rigid_material(
    prim_path: str,
    cfg: UsdFileWithRigidMaterialCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn a USD and strongly bind one explicit rigid-body material.

    Current Isaac Lab ``UsdFileCfg`` does not expose a physics-material field.
    The stock USD spawner first creates or clones the asset; this wrapper then
    creates one material beneath every resolved clone and recursively binds it
    to collision descendants. Binding happens during scene construction, not
    during reset, and therefore cannot write or adapt needle state.

    Args:
        prim_path: Destination USD prim path.
        cfg: USD and rigid-body material configuration.
        translation: Optional translation relative to the parent prim [m].
        orientation: Optional xyzw quaternion relative to the parent prim.
        **kwargs: Additional arguments forwarded to the stock USD spawner.

    Returns:
        Spawned USD prim.
    """

    # The pinned needle authors its rigid body on a descendant without a
    # ``MassAPI``. The stock USD spawner only modifies existing mass schemas,
    # so applying ``mass_props`` at the referenced asset root is a no-op. Defer
    # mass authoring until the unique rigid-body descendant has been resolved.
    spawn_cfg = cfg.replace(mass_props=None)
    prim = spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    resolved_prim_paths = find_matching_prim_paths(prim_path)
    if not resolved_prim_paths:
        raise RuntimeError(f"USD material binding resolved no prims for {prim_path!r}")
    stage = get_current_stage()
    for resolved_prim_path in resolved_prim_paths:
        material_path = f"{resolved_prim_path}/physicsMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(
            resolved_prim_path,
            material_path,
            stronger_than_descendants=True,
        )
        if cfg.mass_props is not None:
            root_prim = stage.GetPrimAtPath(resolved_prim_path)
            rigid_body_prims = [prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(rigid_body_prims) != 1:
                raise RuntimeError(
                    f"needle physical-property binding expected one rigid body beneath {resolved_prim_path!r}, "
                    f"found {[str(prim.GetPath()) for prim in rigid_body_prims]}"
                )
            rigid_body_prim = rigid_body_prims[0]
            sim_utils.define_mass_properties(str(rigid_body_prim.GetPath()), cfg.mass_props, stage=stage)
    return prim


__all__ = ["spawn_usd_with_rigid_material"]
