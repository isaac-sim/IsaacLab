# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .clone_plan import path as cloner_path
from .fabric_notices import disabled_fabric_change_notifies
from .replicate_context import ReplicateContext

if TYPE_CHECKING:
    from pxr import Usd

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
    with disabled_fabric_change_notifies(stage), Sdf.ChangeBlock():
        for source_index in sorted(range(len(sources)), key=lambda index: destinations[index].count("/")):
            source, template = sources[source_index], destinations[source_index]
            if mask is None:
                columns = range(len(env_ids))
            else:
                columns = np.flatnonzero(mask if mask.ndim == 1 else mask[source_index])
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
                    name = "xformOp:translate"
                    attr = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.Double3)
                    attr.default = Gf.Vec3d(*map(float, positions[column]))
                    op_names.append(name)
                if quaternions is not None:
                    name, q = "xformOp:orient", quaternions[column]
                    attr = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.Quatd)
                    attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3])))
                    op_names.append(name)
                name = UsdGeom.Tokens.xformOpOrder
                op_order = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.TokenArray)
                op_order.default = Vt.TokenArray(op_names)


class UsdReplicateContext(ReplicateContext):
    """Apply routed clone-plan sources to one USD stage."""

    # USD destinations must exist before native physics contexts consume them.
    replicate_priority = -100

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Replicate this context's declared sources with the same low-level USD operation."""
        from pxr import Gf, Sdf, Vt  # noqa: PLC0415

        sources = cloner_path.get_asset_prototype_paths(plan)
        templates, starts, world_ids, world_starts = cloner_path.get_world_prototype_asset_templates(
            plan, include_world_indices=True
        )
        assets = plan.topology.world_prototypes
        # Group copies by source/template, omitting descendants already covered by an identical parent copy.
        copies = {}
        for group in np.flatnonzero(np.diff(world_starts)):
            start, end = starts[group : group + 2]
            targets = world_ids[world_starts[group] : world_starts[group + 1]]
            members = [index for index in range(start, end) if assets[index] in asset_prototype_ids]
            destinations = [templates[index] for index in members]
            for index, parent in zip(members, cloner_path.get_parent_indices(destinations), strict=True):
                source, template = sources[assets[index]], templates[index]
                if parent != -1:
                    ancestor = members[parent]
                    suffix = cloner_path.relative_to(template, templates[ancestor])
                    if source == sources[assets[ancestor]] + suffix:
                        continue
                copies.setdefault((source, template), []).append(targets)
        env_ids = range(len(plan.topology.world_prototype_layout))
        with disabled_fabric_change_notifies(self._sim.stage), Sdf.ChangeBlock():
            for source, template in sorted(copies, key=lambda copy: copy[1].count("/")):
                usd_replicate(self._sim.stage, (source,), (template,), np.concatenate(copies[source, template]))
            if plan.positions is not None:
                # Environment frames come from the plan, not copies of an undeclared USD subtree.
                layer = self._sim.stage.GetRootLayer()
                for env_id, position in zip(env_ids, plan.positions, strict=True):
                    path = plan.env_template.format(env_id)
                    spec = Sdf.CreatePrimInLayer(layer, path)
                    spec.specifier = Sdf.SpecifierDef
                    if not spec.typeName:
                        spec.typeName = "Xform"
                    name = "xformOp:translate"
                    attr = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.Double3)
                    attr.default = Gf.Vec3d(*map(float, position))
                    name = "xformOpOrder"
                    order = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.TokenArray)
                    ops = list(order.default or ())
                    if attr.name not in ops:
                        ops.insert(ops.index("!resetXformStack!") + 1 if "!resetXformStack!" in ops else 0, attr.name)
                    order.default = Vt.TokenArray(ops)
