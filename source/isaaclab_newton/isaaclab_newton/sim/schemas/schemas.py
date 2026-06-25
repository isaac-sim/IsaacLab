# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backend schema writers (custom appliers for irregular fragments)."""

from __future__ import annotations

import dataclasses

from pxr import Usd

from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from .schemas_cfg import MujocoFixedTendonCfg

__all__ = ["apply_mujoco_fixed_tendon"]


def apply_mujoco_fixed_tendon(cfg: MujocoFixedTendonCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Write ``mjc:*`` fixed-tendon attributes on a ``MjcTendon`` prim.

    Custom ``func`` override for :class:`~isaaclab_newton.sim.schemas.MujocoFixedTendonCfg`.
    No-op (returns False) on any prim whose type is not ``MjcTendon``.

    Args:
        cfg: The :class:`MujocoFixedTendonCfg` fragment to apply.
        prim_path: The prim path of the ``MjcTendon`` prim.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if the prim is a ``MjcTendon`` and was tuned, False otherwise.

    Raises:
        ValueError: If the prim at ``prim_path`` does not exist in the stage.
    """
    if stage is None:
        stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    values = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if f.name != "func" and getattr(cfg, f.name) is not None
    }
    # Descend the whole subtree (matching legacy apply_nested): ``MjcTendon`` prims may sit below
    # the prim_path the spawner targets. Write ``mjc:*`` on every ``MjcTendon`` descendant.
    found = False
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "MjcTendon":
            continue
        found = True
        for attr_name, value in values.items():
            safe_set_attribute_on_usd_prim(prim, f"mjc:{to_camel_case(attr_name, 'cC')}", value, camel_case=False)
    return found
