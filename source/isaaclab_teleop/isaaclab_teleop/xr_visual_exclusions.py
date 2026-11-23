# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual exclusions for the single-environment XR render export."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from isaaclab.cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE

if TYPE_CHECKING:
    from pxr import Usd


@contextmanager
def exclude_visual_meshes(
    stage: Usd.Stage,
    patterns: Sequence[str],
    *,
    environment_root_path: str = DEFAULT_ENV_TEMPLATE.format(0),
) -> Iterator[tuple[str, ...]]:
    """Hide matching visual meshes from the XR render export.

    A temporary session sublayer makes the meshes invisible while the borrowed render Stage
    is exported. The application's source layers, physics and cameras are not changed. Every
    camera of the export, including the eye cameras and camera panels, omits the meshes.
    Patterns match mesh paths, including instance proxies, not links.

    Args:
        stage: Application Stage containing the selected environment root.
        patterns: Regular expressions that must each fully match at least one visual mesh path.
        environment_root_path: Absolute root of the single environment selected by the caller.
            Defaults to ``/World/envs/env_0``. Only meshes beneath this root are considered.

    Yields:
        The excluded mesh paths, or an empty tuple without patterns.

    Raises:
        ValueError: If the selected environment root is absent or invalid, a pattern matches
            no visual mesh beneath that root, or a match is collision geometry.
        RuntimeError: If a stronger source opinion prevents the visibility override or deinstancing.
    """
    if not patterns:
        yield ()
        return

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root_path = Sdf.Path(environment_root_path)
    if not root_path.IsAbsolutePath() or not root_path.IsPrimPath():
        raise ValueError(f"XR environment root must be an absolute prim path: {environment_root_path!r}")
    environment = stage.GetPrimAtPath(root_path)
    if not environment:
        raise ValueError(f"XR environment root is absent from the Stage: {environment_root_path!r}")
    expressions = [
        re.compile(pattern.replace("{ENV_REGEX_NS}", re.escape(environment_root_path))) for pattern in patterns
    ]
    matches = [[] for _ in expressions]
    for prim in Usd.PrimRange(environment, Usd.TraverseInstanceProxies()):
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
    layer = Sdf.Layer.CreateAnonymous("xr-visual-exclusions.usda")
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
                        raise RuntimeError(f"Cannot author XR visual exclusion: {path}")
                    parent.SetInstanceable(False)
                    if parent.IsInstance():
                        raise RuntimeError(f"Stronger source opinion prevents XR deinstancing: {parent.GetPath()}")
                visibility = UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr()
                visibility.Set(UsdGeom.Tokens.invisible)
                if visibility.Get() != UsdGeom.Tokens.invisible:
                    raise RuntimeError(f"Stronger source opinion prevents XR visual exclusion: {path}")
        yield paths
    finally:
        session.subLayerPaths.remove(layer.identifier)
