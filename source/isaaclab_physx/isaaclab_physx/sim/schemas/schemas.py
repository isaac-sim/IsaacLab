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
)
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from .schemas_cfg import PhysxFixedTendonCfg, PhysxSpatialTendonCfg

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
        A mapping of set field names to their values, ready to author as namespaced USD attributes.
    """
    return {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if f.name != "func" and getattr(cfg, f.name) is not None
    }


def _tune_multi_instance_tendon(cfg, prim_path: str, stage: Usd.Stage | None, markers: tuple[str, ...]) -> bool:
    """Tune every multi-instance tendon schema (matching one of ``markers``) under ``prim_path``.

    Shared backend for :func:`apply_fixed_tendon` / :func:`apply_spatial_tendon`. These schemas are
    *tune-not-apply* (instances are authored in the source asset) and are typically applied on the
    descendant joint prims rather than the ``prim_path`` the spawner targets, so this descends the
    whole subtree (matching the legacy ``apply_nested`` traversal) and writes each set fragment field
    as ``<schema_name>:<camelCase(field)>`` on every prim carrying a matching instance. Applies no
    schema. The fragment's ``_usd_namespace`` is unused (these are not flat-namespace fragments); the
    schema marker is matched explicitly via ``markers``.

    Args:
        cfg: The tendon fragment whose set fields are written.
        prim_path: The prim path (or articulation root) whose subtree carries the schemas.
        stage: The stage to resolve the prim on. Defaults to the current stage.
        markers: Substrings identifying the applied schema(s) to tune (e.g. ``("PhysxTendonAxisRootAPI",)``).

    Returns:
        True if at least one matching instance was tuned, False if none is applied.
    """
    if stage is None:
        stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    values = _strip_fragment_fields(cfg)
    found = False
    for prim in Usd.PrimRange(root):
        applied_schemas = prim.GetAppliedSchemas()
        if not any(m in s for s in applied_schemas for m in markers):
            continue
        found = True
        for schema_name in applied_schemas:
            if not any(m in schema_name for m in markers):
                continue
            for attr_name, value in values.items():
                safe_set_attribute_on_usd_prim(
                    prim, f"{schema_name}:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
                )
    return found


def apply_fixed_tendon(cfg: PhysxFixedTendonCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Tune the multi-instance ``PhysxTendonAxisRootAPI`` schemas on a prim.

    Custom ``func`` override for :class:`PhysxFixedTendonCfg`. The fixed-tendon schema is
    multi-instance and *tune-not-apply* (instances are authored in the source asset), so this
    writes each set fragment field as ``<schema_name>:<camelCase(field)>`` across every applied
    ``PhysxTendonAxisRootAPI`` instance and applies no schema. Writes nothing for the ``mjc:``
    Mujoco path — a separate ``MjcTendon``-aware Newton fragment handles that path.

    Args:
        cfg: The :class:`PhysxFixedTendonCfg` fragment to apply.
        prim_path: The prim path carrying the fixed-tendon schemas.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if at least one ``PhysxTendonAxisRootAPI`` instance was tuned, False if none is applied.
    """
    return _tune_multi_instance_tendon(cfg, prim_path, stage, ("PhysxTendonAxisRootAPI",))


def apply_spatial_tendon(cfg: PhysxSpatialTendonCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Tune the multi-instance ``PhysxTendonAttachment{Root,Leaf}API`` schemas on a prim.

    Custom ``func`` override for :class:`PhysxSpatialTendonCfg`. Writes each set fragment field
    across every applied attachment-root and attachment-leaf instance and applies no schema.

    Args:
        cfg: The :class:`PhysxSpatialTendonCfg` fragment to apply.
        prim_path: The prim path carrying the spatial-tendon schemas.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if at least one attachment instance was tuned, False if none is applied.
    """
    return _tune_multi_instance_tendon(
        cfg, prim_path, stage, ("PhysxTendonAttachmentRootAPI", "PhysxTendonAttachmentLeafAPI")
    )
