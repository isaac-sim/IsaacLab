# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._fabric_notices import disabled_fabric_change_notifies
from .clone_plan import path as cloner_path

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim import SimulationContext
    from .clone_plan import ClonePlan


def usd_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
) -> None:
    """Replicate USD prims from a raw source-to-environment mapping.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination templates containing ``"{}"`` for the environment id.
        env_ids: Environment identifiers.
        mask: Optional per-source or shared mask. ``None`` selects all.
        positions: Optional positions [m], shape ``[num_envs, 3]``. Authored only
            for environment-root destinations (``.../env_{}``).
        quaternions: Optional xyzw orientations, shape ``[num_envs, 4]``. Authored only
            for environment-root destinations.
    """
    # pxr must bind to Kit's USD runtime when Kit is active.
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    layer = stage.GetRootLayer()
    # Parents must be copied before independently declared descendants.
    source_indices = sorted(range(len(sources)), key=lambda index: destinations[index].count("/"))
    with disabled_fabric_change_notifies(stage), Sdf.ChangeBlock():
        for source_index in source_indices:
            source, template = sources[source_index], destinations[source_index]
            columns = (
                np.arange(len(env_ids))
                if mask is None
                else np.flatnonzero(mask if mask.ndim == 1 else mask[source_index])
            )
            is_env_root = template.rstrip("/").endswith("{}")
            for column in columns:
                destination = template.format(int(env_ids[column]))
                Sdf.CreatePrimInLayer(layer, destination)
                # CopySpec beneath an "over" ancestor does not compose as defined.
                ancestor = Sdf.Path(destination).GetParentPath()
                while ancestor != Sdf.Path.absoluteRootPath:
                    spec = layer.GetPrimAtPath(ancestor)
                    if spec is None or spec.specifier != Sdf.SpecifierOver:
                        break
                    spec.specifier = Sdf.SpecifierDef
                    ancestor = ancestor.GetParentPath()
                if source != destination:
                    Sdf.CopySpec(layer, Sdf.Path(source), layer, Sdf.Path(destination))

                if not is_env_root or (positions is None and quaternions is None):
                    continue
                spec = layer.GetPrimAtPath(destination)
                op_names = []
                if positions is not None:
                    attr = spec.GetAttributeAtPath(destination + ".xformOp:translate") or Sdf.AttributeSpec(
                        spec, "xformOp:translate", Sdf.ValueTypeNames.Double3
                    )
                    attr.default = Gf.Vec3d(*map(float, positions[column]))
                    op_names.append("xformOp:translate")
                if quaternions is not None:
                    q = quaternions[column]
                    attr = spec.GetAttributeAtPath(destination + ".xformOp:orient") or Sdf.AttributeSpec(
                        spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd
                    )
                    attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3])))
                    op_names.append("xformOp:orient")
                op_order = spec.GetAttributeAtPath(destination + ".xformOpOrder") or Sdf.AttributeSpec(
                    spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                )
                op_order.default = Vt.TokenArray(op_names)


class UsdReplicateContext:
    """Apply routed clone-plan sources to one USD stage."""

    # USD destinations must exist before native physics contexts consume them.
    replicate_priority = -100

    def __init__(self, sim: SimulationContext):
        self.stage = sim.stage

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Replicate this context's declared sources with the same low-level USD operation."""
        from pxr import Gf, Sdf, Vt  # noqa: PLC0415

        sources = cloner_path.get_asset_prototype_paths(plan)
        templates, starts, world_ids, world_starts = cloner_path.get_world_prototype_asset_templates(
            plan, include_world_indices=True
        )
        copies = {}
        for group, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            targets = world_ids[world_starts[group] : world_starts[group + 1]]
            if not len(targets):
                continue
            members = [
                index for index in range(start, end) if plan.topology.world_prototypes[index] in asset_prototype_ids
            ]
            destinations = [templates[index] for index in members]
            for index, parent in zip(members, cloner_path.get_parent_indices(destinations), strict=True):
                source, template = sources[plan.topology.world_prototypes[index]], templates[index]
                if parent != -1:
                    ancestor = members[parent]
                    suffix = cloner_path.relative_to(template, templates[ancestor])
                    if source == sources[plan.topology.world_prototypes[ancestor]] + suffix:
                        continue
                copies.setdefault((source, template), []).append(targets)
        env_ids = np.arange(len(plan.topology.world_prototype_layout))
        with disabled_fabric_change_notifies(self.stage), Sdf.ChangeBlock():
            for source, template in sorted(copies, key=lambda copy: copy[1].count("/")):
                usd_replicate(self.stage, (source,), (template,), np.concatenate(copies[source, template]))
            if plan.positions is not None:
                # Environment frames come from the plan, not copies of an undeclared USD subtree.
                layer = self.stage.GetRootLayer()
                for env_id, position in zip(env_ids, plan.positions, strict=True):
                    path = plan.env_template.format(int(env_id))
                    spec = Sdf.CreatePrimInLayer(layer, path)
                    spec.specifier = Sdf.SpecifierDef
                    if not spec.typeName:
                        spec.typeName = "Xform"
                    attr = spec.GetAttributeAtPath(path + ".xformOp:translate") or Sdf.AttributeSpec(
                        spec, "xformOp:translate", Sdf.ValueTypeNames.Double3
                    )
                    attr.default = Gf.Vec3d(*map(float, position))
                    order = spec.GetAttributeAtPath(path + ".xformOpOrder") or Sdf.AttributeSpec(
                        spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray
                    )
                    order.default = Vt.TokenArray(["xformOp:translate"])
