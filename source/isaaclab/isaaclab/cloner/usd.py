# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._fabric_notices import disabled_fabric_change_notifies
from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .path import match, split, under
from .query import get_world_prototypes, iter_clones

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
    source_indices = sorted(range(len(sources)), key=lambda index: destinations[index].count("/"))
    with disabled_fabric_change_notifies(stage), Sdf.ChangeBlock():
        for source_index in source_indices:
            source, template = sources[source_index], destinations[source_index]
            columns = (
                np.arange(len(env_ids))
                if mask is None
                else np.flatnonzero(mask if mask.ndim == 1 else mask[source_index])
            )
            is_env_root = "{}" in template and split(template)[1] == ""
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

    def __init__(
        self,
        stage: Usd.Stage,
        plan: ClonePlan,
        *,
        env_template: str = DEFAULT_ENV_TEMPLATE,
        positions: np.ndarray | None = None,
    ):
        """Bind USD naming and placement without adding either to the topology."""
        self.stage = stage
        self.plan = plan
        self.env_template = env_template
        self.positions = positions
        targets = {}
        for world_prototype_id, asset_prototype_ids, world_ids in get_world_prototypes(plan):
            names = set()
            for asset_prototype_id in asset_prototype_ids:
                cfg = plan.asset_prototypes[asset_prototype_id]
                matched = match(cfg.prim_path, self.env_template)
                template = self.env_template + matched.suffix if matched is not None else cfg.prim_path
                if world_prototype_id == -1:
                    template = template.format("shared")
                elif matched is None:
                    template = self.env_template + "/" + cfg.prim_path.rsplit("/", 1)[-1]
                name, occurrence = template, 0
                while template in names:
                    occurrence += 1
                    template = f"{name}_{occurrence}"
                names.add(template)
                targets.setdefault((int(asset_prototype_id), template), []).append(world_ids)
        # Unselected declarations still bound nearest-owner path queries, but are never imported.
        declared = {asset_prototype_id for asset_prototype_id, _ in targets}
        for asset_prototype_id, cfg in enumerate(plan.asset_prototypes):
            matched = match(cfg.prim_path, self.env_template)
            if asset_prototype_id not in declared and matched is not None and getattr(cfg, "spawn", None) is not None:
                targets[asset_prototype_id, self.env_template + matched.suffix] = [np.empty(0, dtype=np.int64)]
        targets = [(index, template, np.concatenate(groups)) for (index, template), groups in targets.items()]
        source_paths = {}
        for asset_prototype_id, template, world_ids in targets:
            if len(world_ids) and asset_prototype_id not in source_paths:
                spawn = getattr(plan.asset_prototypes[asset_prototype_id], "spawn", None)
                path = getattr(spawn, "spawn_path", None)
                source_paths[asset_prototype_id] = path if path is not None else template.format(int(world_ids[0]))
        self.instances = tuple(
            (asset_prototype_id, source_paths.get(asset_prototype_id), template, world_ids)
            for asset_prototype_id, template, world_ids in targets
        )

    @property
    def global_paths(self) -> tuple[str, ...]:
        """USD roots instantiated in the shared world, not inferred from their namespace."""
        paths = tuple(template for _, _, template, world_ids in self.instances if len(world_ids) and world_ids[0] == -1)
        return tuple(path for path in paths if not any(path != root and under(path, root) for root in paths))

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Replicate this context's declared sources with the same low-level USD operation."""
        from pxr import Gf, Sdf, Vt  # noqa: PLC0415

        env_ids = np.arange(len(plan.world_prototype_layout))
        with disabled_fabric_change_notifies(self.stage), Sdf.ChangeBlock():
            for _, source, template, targets in iter_clones(
                instance for instance in self.instances if instance[0] in asset_prototype_ids
            ):
                usd_replicate(self.stage, (source,), (template,), targets)
            if self.positions is not None:
                # Environment frames come from the plan, not copies of an undeclared USD subtree.
                layer = self.stage.GetRootLayer()
                for env_id, position in zip(env_ids, self.positions, strict=True):
                    path = self.env_template.format(int(env_id))
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
