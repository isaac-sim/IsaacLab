# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrappers for deformable schema writers and PhysX fragment appliers.

The deformable schema writers are backend-aware but remain unified in
:mod:`isaaclab.sim.schemas`. This module additionally hosts the PhysX-specific fragment
applier funcs that override :attr:`~isaaclab.sim.schemas.SchemaFragment.func` for the
multi-instance tendon schemas, keeping the backend func out of the core package.
"""

from __future__ import annotations

import dataclasses

from pxr import Usd

from isaaclab.sim.schemas.schemas import (
    define_deformable_body_properties,
    modify_deformable_body_properties,
    modify_fixed_tendon_properties,
    modify_spatial_tendon_properties,
)

from . import schemas_cfg

__all__ = [
    "apply_fixed_tendon",
    "apply_spatial_tendon",
    "define_deformable_body_properties",
    "modify_deformable_body_properties",
]


def _strip_fragment_fields(cfg) -> dict:
    """Collect a fragment's non-``None`` data fields, excluding the ``func`` plumbing field.

    Args:
        cfg: The fragment instance to read fields from.

    Returns:
        A mapping of set field names to their values, suitable for building a legacy cfg.
    """
    return {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if f.name != "func" and getattr(cfg, f.name) is not None
    }


def apply_fixed_tendon(cfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Apply a :class:`PhysxFixedTendonCfg` fragment by delegating to the legacy writer.

    Thin override-``func`` wrapper for fixed-tendon fragments. The fragment's data fields are
    repackaged into a :class:`PhysxFixedTendonPropertiesCfg` (dropping the ``func`` plumbing
    field) and handed to :func:`~isaaclab.sim.schemas.modify_fixed_tendon_properties`, which
    tunes every existing ``PhysxTendonAxisRootAPI:<inst>`` instance on the prim (and the
    ``MjcTendon`` ``mjc:*`` branch when present).

    Args:
        cfg: The :class:`PhysxFixedTendonCfg` fragment to apply.
        prim_path: The prim path to the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    legacy = schemas_cfg.PhysxFixedTendonPropertiesCfg(**_strip_fragment_fields(cfg))
    # ``modify_fixed_tendon_properties`` is wrapped by ``apply_nested`` and returns ``None``;
    # report success explicitly to honor the documented ``bool`` contract.
    modify_fixed_tendon_properties(prim_path, legacy, stage)
    return True


def apply_spatial_tendon(cfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Apply a :class:`PhysxSpatialTendonCfg` fragment by delegating to the legacy writer.

    Thin override-``func`` wrapper for spatial-tendon fragments. The fragment's data fields are
    repackaged into a :class:`PhysxSpatialTendonPropertiesCfg` (dropping the ``func`` plumbing
    field) and handed to :func:`~isaaclab.sim.schemas.modify_spatial_tendon_properties`, which
    tunes every existing ``PhysxTendonAttachmentRootAPI:<inst>`` /
    ``PhysxTendonAttachmentLeafAPI:<inst>`` instance on the prim.

    Args:
        cfg: The :class:`PhysxSpatialTendonCfg` fragment to apply.
        prim_path: The prim path to the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    legacy = schemas_cfg.PhysxSpatialTendonPropertiesCfg(**_strip_fragment_fields(cfg))
    # ``modify_spatial_tendon_properties`` is wrapped by ``apply_nested`` and returns ``None``;
    # report success explicitly to honor the documented ``bool`` contract.
    modify_spatial_tendon_properties(prim_path, legacy, stage)
    return True
