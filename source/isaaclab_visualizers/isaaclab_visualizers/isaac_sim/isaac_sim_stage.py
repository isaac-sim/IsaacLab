# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers that relate Isaac Lab's simulated bodies to the prims shown by the Isaac Sim 7 viewer."""

from __future__ import annotations

import math

import warp as wp

from pxr import Gf, Usd, UsdGeom


def authored_scales(stage: Usd.Stage, paths: list[str]) -> wp.array | None:
    """Return per-body authored scales, or ``None`` when every body is unit scale.

    Simulation poses omit scale, and the viewer renders each body with an ``omni:resetXformStack`` local matrix,
    so the authored scale has to be folded back into the published pose.

    Args:
        stage: Authored Isaac Lab stage.
        paths: Body prim paths, in publication order.

    Returns:
        Scales [-] with shape ``[len(paths)]`` of ``vec3f``, or ``None`` for unit scale.
    """
    cache = UsdGeom.XformCache()
    scales = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsA(UsdGeom.Xformable):
            scales.append(tuple(map(float, Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale())))
        else:
            scales.append((1.0, 1.0, 1.0))
    if all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for scale in scales for axis in scale):
        return None
    return wp.array(scales, dtype=wp.vec3f)
