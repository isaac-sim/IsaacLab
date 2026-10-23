# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-selective visual exclusions for the single-environment XR export."""

import re
from contextlib import contextmanager


@contextmanager
def partition_visual_meshes(stage, patterns):
    """Exclude matching visual meshes from environment cameras during export.

    A temporary session sublayer is flattened into the borrowed render Stage.
    The application's source layers, visibility, physics and cameras are not
    changed. Patterns match mesh paths, including instance proxies, not links.
    """
    if not patterns:
        yield ()
        return

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    envs = stage.GetPrimAtPath("/World/envs")
    if not envs or [str(p.GetPath()) for p in envs.GetChildren()] != ["/World/envs/env_0"]:
        raise ValueError("XR visual partitions require exactly env_0")
    expressions = [re.compile(pattern) for pattern in patterns]
    matches = [[] for _ in expressions]
    for prim in Usd.PrimRange(envs, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        for index, expression in enumerate(expressions):
            if expression.fullmatch(str(prim.GetPath())):
                if prim.HasAPI(UsdPhysics.CollisionAPI) or "collisions" in prim.GetPath().pathString.split("/"):
                    raise ValueError(f"XR visual exclusion matched collision geometry: {prim.GetPath()}")
                matches[index].append(str(prim.GetPath()))
    if any(not paths for paths in matches):
        raise ValueError("XR visual exclusion pattern matched no visual mesh")
    paths = tuple(dict.fromkeys(path for group in matches for path in group))
    layer = Sdf.Layer.CreateAnonymous("xr-visual-partitions.usda")
    session = stage.GetSessionLayer()
    session.subLayerPaths.insert(0, layer.identifier)
    try:
        with Usd.EditContext(stage, layer):
            for path in paths:
                while stage.GetPrimAtPath(path).IsInstanceProxy():
                    parent = stage.GetPrimAtPath(path).GetParent()
                    while parent and not parent.IsInstance():
                        parent = parent.GetParent()
                    if not parent:
                        raise RuntimeError(f"Cannot author XR visual partition: {path}")
                    parent.SetInstanceable(False)
                    if parent.IsInstance():
                        raise RuntimeError(f"Stronger source opinion prevents XR deinstancing: {parent.GetPath()}")
                partition = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).CreatePrimvar(
                    "omni:scenePartition", Sdf.ValueTypeNames.Token, UsdGeom.Tokens.constant
                )
                partition.Set("xr_excluded")
                if partition.Get() != "xr_excluded":
                    raise RuntimeError(f"Stronger source opinion prevents XR visual partition: {path}")
            settings_path = stage.GetMetadata("renderSettingsPrimPath") or "/Render/IsaacLabSettings"
            settings = stage.DefinePrim(settings_path, "RenderSettings")
            show_all = settings.CreateAttribute(
                "omni:rtx:scenePartitioning:showAllPartitionsByDefault", Sdf.ValueTypeNames.Bool
            )
            show_all.Set(False)
            if show_all.Get() is not False:
                raise RuntimeError("Stronger source opinion prevents XR camera-selective partitions")
        yield paths
    finally:
        session.subLayerPaths.remove(layer.identifier)
