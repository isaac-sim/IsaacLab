# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab.scene_data import SceneDataFormat

from ._fabric_notices import disabled_fabric_change_notifies
from .path import split

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.scene_data import SceneDataProvider

    from .clone_plan import ClonePlan


def _select_columns(env_ids: np.ndarray, mask: np.ndarray | None, row: int) -> np.ndarray:
    """Return the mask columns selected by a replication row."""
    if mask is None:
        return np.arange(len(env_ids))
    row_mask = mask if mask.ndim == 1 else mask[row]
    return np.flatnonzero(row_mask)


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
        self._fabric_output: SceneDataFormat.FabricMatrix44 | None = None

    def replicate(self, plan: ClonePlan) -> None:
        """Apply this context's routed rows from a clone plan.

        Args:
            plan: Replication layout shared by every clone backend.
        """
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required for replication.")
        rows = plan.context_rows[type(self)]
        replication_rows = []
        for row in rows:
            columns = _select_columns(plan.env_ids, plan.clone_mask, row)
            target_envs = plan.env_ids[columns]
            positions = None if plan.positions is None else plan.positions[columns]
            replication_rows.append((plan.sources[row], plan.destinations[row], target_envs, positions, None))
        if not replication_rows:
            return

        # Suspend Fabric's per-Sdf.CopySpec notice listener for the duration of the copy work;
        # no-op outside a live Kit application.
        with disabled_fabric_change_notifies(self.stage):
            self._apply(replication_rows)

    def _prepare_fabric(self, provider: SceneDataProvider, device: str) -> None:
        """Bind native publication paths to their existing Fabric destinations."""
        if self._fabric_output is not None:
            return
        # Fabric is supplied by the running Kit application, not the standalone USD wheel.
        import usdrt  # noqa: PLC0415
        import usdrt.hierarchy  # noqa: PLC0415
        from pxr import UsdUtils  # noqa: PLC0415

        stage_id = UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        self._fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
        self._fabric_stage.SynchronizeToFabric()
        self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            self._fabric_stage.GetFabricId(), self._fabric_stage.GetStageIdAsStageId()
        )
        self._fabric_hierarchy.update_world_xforms()
        gpu_options = getattr(usdrt.hierarchy, "FabricHierarchyGpuUpdateOptions", None)
        self._fabric_update_options = (
            gpu_options.RIGID_BODY | gpu_options.FORCE_UPDATE
            if gpu_options is not None and hasattr(self._fabric_hierarchy, "update_world_xforms_gpu_with_options")
            else None
        )
        self._fabric_selection = self._fabric_stage.SelectPrims(
            require_applied_schemas=["PhysicsRigidBodyAPI"],
            require_attrs=[(usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite)],
            device=device,
            want_paths=True,
        )
        self._fabric_provider = provider
        self._fabric_device = device
        self._fabric_paths = provider.backend.transform_paths
        self._fabric_generation = -1
        self._fabric_output = SceneDataFormat.FabricMatrix44()
        provider._prepare_fabric_output = self._prepare_fabric_output

    def _prepare_fabric_output(self) -> SceneDataFormat.FabricMatrix44:
        """Refresh a cached Fabric selection after topology changes."""
        changed = self._fabric_selection.PrepareForReuse()
        if changed or self._fabric_output.matrices is None:
            slots = {str(path): index for index, path in enumerate(self._fabric_selection.GetPaths())}
            paths = [path for path in self._fabric_paths if path in slots]
            indices = wp.array([slots[path] for path in paths], dtype=wp.int32, device=self._fabric_device)
            self._fabric_output = SceneDataFormat.FabricMatrix44(
                matrices=wp.indexedfabricarray(
                    fa=wp.fabricarray(self._fabric_selection, "omni:fabric:worldMatrix"), indices=indices
                ),
                mapping=self._fabric_provider.create_mapping(paths),
            )
            self._fabric_generation = -1
        return self._fabric_output

    def _update_fabric(self) -> None:
        """Consume SDP poses and propagate them without rebuilding Fabric connectivity."""
        if self._fabric_update_options is not None:
            self._fabric_hierarchy.track_world_xform_changes(False)
            self._fabric_hierarchy.track_local_xform_changes(False)
        try:
            self._fabric_provider.request_transforms(SceneDataFormat.FabricMatrix44)
            generation = self._fabric_provider.transform_generation
            if generation != self._fabric_generation:
                wp.synchronize_device(self._fabric_device)
                if self._fabric_update_options is None:
                    self._fabric_hierarchy.update_world_xforms()
                else:
                    self._fabric_hierarchy.update_world_xforms_gpu_with_options(self._fabric_update_options)
                self._fabric_generation = generation
        finally:
            if self._fabric_update_options is not None:
                self._fabric_hierarchy.track_world_xform_changes(True)
                self._fabric_hierarchy.track_local_xform_changes(True)

    def _apply(
        self,
        replication_rows: list[tuple[str, str, np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> None:
        """Author the supplied copy specs into the stage's root layer."""
        # pxr must be imported after Kit starts; importing it with this module can bind
        # a different USD runtime before Kit initializes its plugins.
        from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

        rl = self.stage.GetRootLayer()

        def dp_depth(template: str) -> int:
            """Return destination prim path depth for stable parent-first replication."""
            dp = template.format(0)
            return Sdf.Path(dp).pathElementCount

        rows_by_depth: dict[int, list[tuple[str, str, np.ndarray, np.ndarray | None, np.ndarray | None]]] = {}
        for row in replication_rows:
            rows_by_depth.setdefault(dp_depth(row[1]), []).append(row)

        for depth in sorted(rows_by_depth):
            with Sdf.ChangeBlock():
                for src, tmpl, target_envs, positions, quaternions in rows_by_depth[depth]:
                    _, clone_suffix = split(tmpl)
                    is_instance_root = clone_suffix == ""

                    for column, wid in enumerate(target_envs):
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
    env_ids: np.ndarray,
    mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
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
    replication_rows = []
    for row, source in enumerate(sources):
        columns = _select_columns(env_ids, mask, row)
        target_envs = env_ids[columns]
        row_positions = None if positions is None else positions[columns]
        row_quaternions = None if quaternions is None else quaternions[columns]
        replication_rows.append((source, destinations[row], target_envs, row_positions, row_quaternions))
    context = UsdReplicateContext(stage)
    if replication_rows:
        with disabled_fabric_change_notifies(stage):
            context._apply(replication_rows)
