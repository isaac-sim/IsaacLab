# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ._fabric_notices import disabled_fabric_change_notifies
from .path import split
from .query import _clone_mapping

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


def _select_env_ids(env_ids: torch.Tensor, mask: torch.Tensor | None, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    if mask is None:
        return env_ids
    row_mask = mask if mask.dim() == 1 else mask[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


class UsdReplicateContext:
    """Apply routed clone-plan rows to one USD stage."""

    # USD destinations must exist before native physics contexts consume them.
    replicate_priority = -100

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to author replicated prim specs into.
        """
        self.stage = stage

    def replicate(self, plan: ClonePlan) -> None:
        """Apply this context's routed rows from a clone plan.

        Args:
            plan: Replication layout shared by every clone backend.
        """
        rows = plan.context_rows[type(self)]
        sources, destinations, mask = _clone_mapping(plan, rows, whole_env=True)
        items = []
        for row, source in enumerate(sources):
            columns = mask[row].to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
            target_envs = plan.env_ids[columns.to(device=plan.env_ids.device)]
            positions = plan.positions[columns.to(device=plan.positions.device)]
            items.append((source, destinations[row], target_envs, positions, None))
        if not items:
            return

        # Suspend Fabric's per-Sdf.CopySpec notice listener for the duration of the copy work;
        # no-op outside a live Kit application.
        with disabled_fabric_change_notifies(self.stage):
            self._apply(items)

    def _apply(
        self,
        items: list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
    ) -> None:
        """Author the supplied copy specs into the stage's root layer."""
        rl = self.stage.GetRootLayer()

        def dp_depth(template: str) -> int:
            """Return destination prim path depth for stable parent-first replication."""
            dp = template.format(0)
            return Sdf.Path(dp).pathElementCount

        depth_to_items: dict[int, list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]] = {}
        for item in items:
            depth_to_items.setdefault(dp_depth(item[1]), []).append(item)

        for depth in sorted(depth_to_items.keys()):
            with Sdf.ChangeBlock():
                for src, tmpl, target_envs, positions, quaternions in depth_to_items[depth]:
                    _, clone_suffix = split(tmpl)
                    is_instance_root = clone_suffix == ""

                    for column, wid in enumerate(target_envs.tolist()):
                        wid = int(wid)
                        dp = tmpl.format(wid)
                        Sdf.CreatePrimInLayer(rl, dp)
                        # ``CreatePrimInLayer`` authors missing intermediate ancestors (e.g. the
                        # ``Groceries`` scope in ``env_{}/Groceries/Object``) as ``over`` specs. A
                        # ``def`` copied below an ``over`` ancestor never composes as defined, so
                        # Hydra skips it and its references stay unexpanded. Promote such ancestors
                        # to ``def``; for ancestors already defined elsewhere this is a no-op.
                        ancestor = Sdf.Path(dp).GetParentPath()
                        while ancestor != Sdf.Path.absoluteRootPath:
                            ancestor_spec = rl.GetPrimAtPath(ancestor)
                            if ancestor_spec is None or ancestor_spec.specifier != Sdf.SpecifierOver:
                                break
                            ancestor_spec.specifier = Sdf.SpecifierDef
                            ancestor = ancestor.GetParentPath()
                        if src != dp:
                            Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                        # Author positions/quaternions for instance roots only.
                        if is_instance_root and (positions is not None or quaternions is not None):
                            ps = rl.GetPrimAtPath(dp)
                            op_names = []
                            if positions is not None:
                                p = positions[column]
                                t_attr = ps.GetAttributeAtPath(dp + ".xformOp:translate")
                                if t_attr is None:
                                    t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                                t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                                op_names.append("xformOp:translate")
                            if quaternions is not None:
                                q = quaternions[column]
                                o_attr = ps.GetAttributeAtPath(dp + ".xformOp:orient")
                                if o_attr is None:
                                    o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                                o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                                op_names.append("xformOp:orient")
                            if op_names:
                                op_order = ps.GetAttributeAtPath(dp + ".xformOpOrder") or Sdf.AttributeSpec(
                                    ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                                )
                                op_order.default = Vt.TokenArray(op_names)


def usd_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
) -> None:
    """Replicate USD prims directly for standalone tooling and tests.

    Production clone lifecycles route a :class:`~isaaclab.cloner.ClonePlan` through
    :meth:`UsdReplicateContext.replicate`; this wrapper retains direct control over raw
    mappings for tools that do not own a clone plan.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination formattable templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mask: Optional per-source or shared mask. ``None`` selects all.
        positions: Optional positions [m], shape ``[E, 3]``. Authored as ``xformOp:translate`` only
            for env-instance root destinations (``.../env_{}``).
        quaternions: Optional orientations in xyzw order, shape ``[E, 4]``. Authored as
            ``xformOp:orient`` only for env-instance root destinations (``.../env_{}``).
    """
    items = []
    for row, source in enumerate(sources):
        target_envs = _select_env_ids(env_ids, mask, row)
        indices = target_envs.to(device=positions.device) if positions is not None else target_envs
        row_positions = None if positions is None else positions[indices]
        indices = target_envs.to(device=quaternions.device) if quaternions is not None else target_envs
        row_quaternions = None if quaternions is None else quaternions[indices]
        items.append((source, destinations[row], target_envs, row_positions, row_quaternions))
    context = UsdReplicateContext(stage)
    if items:
        with disabled_fabric_change_notifies(stage):
            context._apply(items)
