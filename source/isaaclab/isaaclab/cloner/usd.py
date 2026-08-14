# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ._fabric_notices import disabled_fabric_change_notifies
from .path import split


class UsdReplicateContext:
    """Queue and apply USD replication work for one stage."""

    replicate_priority = 100

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to author replicated prim specs into.
        """
        self.stage = stage
        self._queue: list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []

    def queue(
        self,
        source: str,
        destination: str,
        env_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Queue one USD source row for replication.

        Args:
            source: Source prim path.
            destination: Destination path template with ``"{}"`` for env id.
            env_ids: Environment ids selected for this source row.
            positions: Optional world positions [m], one row per env id. Authored on the instance
                root identified by the destination template, including nested destinations.
            quaternions: Optional xyzw orientations, one row per env id. Authored on the instance
                root identified by the destination template, including nested destinations.
        """
        if positions is not None and len(positions) != len(env_ids):
            raise ValueError(f"positions must have {len(env_ids)} rows, got {len(positions)}.")
        if quaternions is not None and len(quaternions) != len(env_ids):
            raise ValueError(f"quaternions must have {len(env_ids)} rows, got {len(quaternions)}.")
        self._queue.append((source, destination, env_ids, positions, quaternions))

    def queue_mapping(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Queue replication rows from the current flat clone mapping.

        Args:
            sources: Source prim paths.
            destinations: Destination path templates with ``"{}"`` for env id.
            env_ids: Environment indices.
            mask: Optional per-source or shared mask.
            positions: Optional per-environment world positions [m], aligned with env-id columns.
                Authored on the instance root identified by each destination template.
            quaternions: Optional per-environment xyzw orientations, aligned with env-id columns.
                Authored on the instance root identified by each destination template.
        """
        for i, source in enumerate(sources):
            if mask is None:
                row_mask = torch.ones_like(env_ids, dtype=torch.bool)
            else:
                row_mask = mask if mask.dim() == 1 else mask[i]
                row_mask = row_mask.to(dtype=torch.bool)
            self.queue(
                source,
                destinations[i],
                env_ids[row_mask],
                positions=positions[row_mask] if positions is not None else None,
                quaternions=quaternions[row_mask] if quaternions is not None else None,
            )

    def replicate(self, payload: object | None = None) -> None:
        """Drain queued USD copy specs in parent-before-child order.

        Args:
            payload: Optional lifecycle event payload. Ignored.
        """
        del payload
        queue, self._queue = self._queue, []
        if not queue:
            return

        # Suspend Fabric's per-Sdf.CopySpec notice listener for the duration of the copy work;
        # no-op outside a live Kit application.
        with disabled_fabric_change_notifies(self.stage):
            self._apply_queue(queue)

    def _apply_queue(
        self, queue: Sequence[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]
    ) -> None:
        """Author the queued copy specs into the stage's root layer."""
        rl = self.stage.GetRootLayer()

        depth_to_items: dict[int, list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]] = {}
        for item in queue:
            depth_to_items.setdefault(Sdf.Path(item[1].format(0)).pathElementCount, []).append(item)

        for depth in sorted(depth_to_items.keys()):
            with Sdf.ChangeBlock():
                for src, tmpl, target_envs, positions, quaternions in depth_to_items[depth]:
                    clone_prefix, clone_suffix = split(tmpl)
                    is_instance_root = clone_suffix == ""

                    for pose_row, wid in enumerate(target_envs.tolist()):
                        wid = int(wid)
                        dp = tmpl.format(wid)
                        instance_root_path = f"{clone_prefix}{wid}"
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
                        # A root copy must precede its transform overrides; a child copy follows
                        # root setup so the missing instance root is positioned before CopySpec.
                        if is_instance_root and src != dp:
                            Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                        if positions is not None or quaternions is not None:
                            ps = rl.GetPrimAtPath(instance_root_path)
                            ps.specifier = Sdf.SpecifierDef
                            ps.typeName = "Xform"
                            op_names = []
                            if positions is not None:
                                p = positions[pose_row]
                                t_attr = ps.GetAttributeAtPath(instance_root_path + ".xformOp:translate")
                                if t_attr is None:
                                    t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                                t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                                op_names.append("xformOp:translate")
                            if quaternions is not None:
                                q = quaternions[pose_row]
                                o_attr = ps.GetAttributeAtPath(instance_root_path + ".xformOp:orient")
                                if o_attr is None:
                                    o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                                o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                                op_names.append("xformOp:orient")
                            op_order = ps.GetAttributeAtPath(instance_root_path + ".xformOpOrder") or Sdf.AttributeSpec(
                                ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                            )
                            op_order.default = Vt.TokenArray(op_names)

                        if not is_instance_root and src != dp:
                            Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))


def usd_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
) -> None:
    """Replicate USD prims to per-environment destinations.

    Copies each source prim spec to destination templates for selected environments
    (``mask``). Optionally authors translate/orient from position/quaternion buffers.
    Replication runs in path-depth order (parents before children) for robust composition.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination formattable templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mask: Optional per-source or shared mask. ``None`` selects all.
        positions: Optional positions [m], shape ``[E, 3]``. Authored as ``xformOp:translate`` on
            the instance root identified by each destination template.
        quaternions: Optional orientations in xyzw order, shape ``[E, 4]``. Authored as
            ``xformOp:orient`` on the instance root identified by each destination template.
    """
    ctx = UsdReplicateContext(stage)
    ctx.queue_mapping(sources, destinations, env_ids, mask, positions=positions, quaternions=quaternions)
    ctx.replicate()
