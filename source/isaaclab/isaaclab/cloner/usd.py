# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ._fabric_notices import disabled_fabric_change_notifies
from .replicate_session import REPLICATION_QUEUE


def _select_env_ids(env_ids: torch.Tensor, mask: torch.Tensor | None, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    if mask is None:
        return env_ids
    row_mask = mask if mask.dim() == 1 else mask[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


def _source_is_nested_destination_instance(source: str, destination_template: str) -> bool:
    """Return True when ``source`` is a concrete nested instance of ``destination_template``.

    Env-origin positions belong on clone roots such as ``/World/envs/env_{}``.
    Rows such as ``/World/envs/env_{}/Camera`` already carry a local transform
    copied from their source prim and must not receive the env origin as their
    local translate.
    """
    if "{}" not in destination_template:
        return False

    prefix, _, suffix = destination_template.partition("{}")
    if not suffix:
        return False

    source = source.rstrip("/") or "/"
    prefix = prefix.rstrip("/") if prefix != "/" else prefix
    suffix = suffix.rstrip("/")
    if not source.startswith(prefix) or not source.endswith(suffix):
        return False

    slot_end = len(source) - len(suffix)
    slot_value = source[len(prefix) : slot_end]
    return bool(slot_value) and "/" not in slot_value.strip("/")


def _author_xform_op_order(prim_spec: Sdf.PrimSpec, prim_path: str, authored_op_names: Sequence[str]) -> None:
    """Preserve copied xform ops and append newly-authored ops to ``xformOpOrder``."""
    op_order = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder") or Sdf.AttributeSpec(
        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
    )
    ordered_ops = list(op_order.default) if op_order.default is not None else []
    for op_name in authored_op_names:
        if op_name not in ordered_ops:
            ordered_ops.append(op_name)
    op_order.default = Vt.TokenArray(ordered_ops)


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
            positions: Optional per-environment world positions [m].
            quaternions: Optional per-environment orientations in xyzw order.
        """
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
            positions: Optional per-environment world positions [m].
            quaternions: Optional per-environment orientations in xyzw order.
        """
        for i, source in enumerate(sources):
            # Clone-plan positions are env-root origins. Nested rows already have their
            # local transform authored in the source prim, so keep that local transform.
            row_positions = None if _source_is_nested_destination_instance(source, destinations[i]) else positions
            self.queue(
                source,
                destinations[i],
                _select_env_ids(env_ids, mask, i),
                positions=row_positions,
                quaternions=quaternions,
            )

    def replicate(self) -> None:
        """Apply all queued USD copy specs in parent-before-child order."""
        if not self._queue:
            return

        # Suspend Fabric's per-Sdf.CopySpec notice listener for the duration of the copy work;
        # no-op outside a live Kit application.
        with disabled_fabric_change_notifies(self.stage):
            self._apply_queue()

    def _apply_queue(self) -> None:
        """Author the queued copy specs into the stage's root layer."""
        rl = self.stage.GetRootLayer()

        def dp_depth(template: str) -> int:
            """Return destination prim path depth for stable parent-first replication."""
            dp = template.format(0)
            return Sdf.Path(dp).pathElementCount

        depth_to_items: dict[int, list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]] = {}
        for item in self._queue:
            depth_to_items.setdefault(dp_depth(item[1]), []).append(item)

        for depth in sorted(depth_to_items.keys()):
            with Sdf.ChangeBlock():
                for src, tmpl, target_envs, positions, quaternions in depth_to_items[depth]:
                    for wid in target_envs.tolist():
                        wid = int(wid)
                        dp = tmpl.format(wid)
                        Sdf.CreatePrimInLayer(rl, dp)
                        if src != dp:
                            Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                        if positions is not None or quaternions is not None:
                            ps = rl.GetPrimAtPath(dp)
                            op_names = []
                            if positions is not None:
                                p = positions[wid]
                                t_attr = ps.GetAttributeAtPath(dp + ".xformOp:translate")
                                if t_attr is None:
                                    t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                                t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                                op_names.append("xformOp:translate")
                            if quaternions is not None:
                                q = quaternions[wid]
                                o_attr = ps.GetAttributeAtPath(dp + ".xformOp:orient")
                                if o_attr is None:
                                    o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                                o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                                op_names.append("xformOp:orient")
                            if op_names:
                                _author_xform_op_order(ps, dp, op_names)


def queue_usd_replication(cfg: Any) -> None:
    """Register ``cfg`` for USD replication when :func:`~isaaclab.cloner.replicate` next runs.

    Appends ``(cfg, UsdReplicateContext)`` to :data:`~isaaclab.cloner.REPLICATION_QUEUE`.
    The actual row resolution and dispatch happen inside :func:`~isaaclab.cloner.replicate`,
    so this helper is safe to call from any asset constructor — no active session is required.
    """
    REPLICATION_QUEUE.append((cfg, UsdReplicateContext))


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
        positions: Optional positions [m], shape ``[E, 3]``, authored as ``xformOp:translate``.
        quaternions: Optional orientations in xyzw order, shape ``[E, 4]``, authored as ``xformOp:orient``.
    """
    ctx = UsdReplicateContext(stage)
    ctx.queue_mapping(sources, destinations, env_ids, mask, positions=positions, quaternions=quaternions)
    ctx.replicate()
