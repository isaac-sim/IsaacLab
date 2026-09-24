# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry facts compiled from declared clone prototypes before replication."""

from __future__ import annotations

from collections import defaultdict

from pxr import Sdf, Usd, UsdGeom

from ..scene_data.deformable_discovery import deformable_entry
from ..sim.utils.queries import has_deformable_body_api, has_deformable_curve_api
from .clone_plan import ClonePlan


def compile_geometry(plan: ClonePlan, stage: Usd.Stage) -> None:
    """Read geometry once from the plan's spawned prototypes and shared roots.

    Nested source rows own their geometry independently of ancestor rows. Inactive rows do not
    contribute geometry, and undeclared stage objects are never considered.

    Args:
        plan: Plan populated in place so every clone context retains the same object.
        stage: Stage containing the declared, spawned source prims.
    """
    if plan.deformables:
        return
    deformables = {row: [] for row in (*range(len(plan.sources)), None)}
    cables = {row: [] for row in deformables}
    source_rows = defaultdict(list)
    active = plan.clone_mask.any(axis=1)
    for row, source in enumerate(plan.sources):
        source_rows[Sdf.Path(source)].append(row)
    roots = Sdf.Path.RemoveDescendentPaths(
        [source for row, source in enumerate(plan.sources) if active[row]] + list(plan.global_paths)
    )
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            path = prim.GetPath()
            is_curve = prim.IsA(UsdGeom.BasisCurves)
            is_deformable = not (is_curve or prim.IsA(UsdGeom.Points)) and has_deformable_body_api(prim)
            if not (is_deformable or is_curve):
                continue
            owner = path
            while owner != Sdf.Path.absoluteRootPath and owner not in source_rows:
                owner = owner.GetParentPath()
            rows = tuple(row for row in source_rows[owner] if active[row]) if owner in source_rows else (None,)
            if not rows:
                continue
            if is_deformable:
                if (entry := deformable_entry(prim)) is not None:
                    for row in rows:
                        deformables[row].append(entry)
            elif is_curve and has_deformable_curve_api(prim):
                curve = UsdGeom.BasisCurves(prim)
                counts = curve.GetCurveVertexCountsAttr().Get()
                if (
                    counts is not None
                    and len(counts) == 1
                    and counts[0] >= 2
                    and curve.GetTypeAttr().Get() == UsdGeom.Tokens.linear
                    and curve.GetWrapAttr().Get() != UsdGeom.Tokens.periodic
                ):
                    for row in rows:
                        cables[row].append((str(path), int(counts[0]) - 1))
    plan.deformables.update((row, tuple(entries)) for row, entries in deformables.items())
    plan.cables.update((row, tuple(entries)) for row, entries in cables.items())
