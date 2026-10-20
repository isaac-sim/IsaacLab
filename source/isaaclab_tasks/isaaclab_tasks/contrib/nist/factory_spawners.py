# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from pxr import Usd


_NISTBOARD_COLLIDER_THICKNESS = 0.03  # [m]


@sim_utils.clone
def spawn_nistboard_with_deep_collider(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the NIST board with a 30 mm collider extending below its fixed top face."""
    from pxr import Gf, UsdGeom  # noqa: PLC0415

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    collider = sim_utils.get_current_stage().GetPrimAtPath(f"{prim_path}/collisions/mesh_0")
    if not collider.IsA(UsdGeom.Cube):
        raise RuntimeError(f"Expected the NIST board collider to be a cube: {collider.GetPath()}")

    size_value = UsdGeom.Cube(collider).GetSizeAttr().Get()
    scale_attr = collider.GetAttribute("xformOp:scale")
    translate_attr = collider.GetAttribute("xformOp:translate")
    orient = collider.GetAttribute("xformOp:orient").Get()
    scale = scale_attr.Get()
    center = translate_attr.Get()
    if (
        size_value is None
        or scale is None
        or center is None
        or orient is None
        or orient.GetReal() != 1.0
        or tuple(orient.GetImaginary()) != (0.0, 0.0, 0.0)
    ):
        raise RuntimeError(f"Invalid NIST board collider transform: {collider.GetPath()}")

    size = float(size_value)
    authored_thickness = size * float(scale[2])
    if size <= 0.0 or authored_thickness <= 0.0 or authored_thickness > _NISTBOARD_COLLIDER_THICKNESS:
        raise RuntimeError(f"Invalid NIST board collider dimensions: {collider.GetPath()}")

    # The configured board is rotated 180 degrees about Y, so its local lower face is
    # the world-facing top. Keep that face fixed and extend only the opposite face.
    fixed_face = float(center[2]) - 0.5 * size * float(scale[2])
    scale_attr.Set(Gf.Vec3d(scale[0], scale[1], _NISTBOARD_COLLIDER_THICKNESS / size))
    translate_attr.Set(Gf.Vec3d(center[0], center[1], fixed_face + 0.5 * _NISTBOARD_COLLIDER_THICKNESS))
    return prim
