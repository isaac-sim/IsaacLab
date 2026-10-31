# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small display-color extension for native cuboid assets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd


def _author_display_color(prim_path: str, cfg: ColoredCuboidCfg, stage: Usd.Stage) -> None:
    """Author a portable USD display color on the standard cuboid geometry."""
    from pxr import Gf, UsdGeom

    mesh_prim = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
    UsdGeom.Gprim(mesh_prim).CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*cfg.display_color)])


def _spawn_colored_cuboid_impl(
    prim_path: str,
    cfg: ColoredCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn and color one standard cuboid before the asset is cloned."""
    from isaaclab.sim.spawners.shapes import spawn_cuboid
    from isaaclab.sim.utils import get_current_stage

    prim = spawn_cuboid.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _author_display_color(prim_path, cfg, get_current_stage())
    return prim


def spawn_colored_cuboid(
    prim_path: str,
    cfg: ColoredCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn a standard cuboid with a kitless-compatible display color."""
    from isaaclab.sim.utils import clone

    return clone(_spawn_colored_cuboid_impl)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


@configclass
class ColoredCuboidCfg(sim_utils.CuboidCfg):
    """Native cuboid with an authored USD ``displayColor``."""

    func = spawn_colored_cuboid

    display_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    """Display color of the render-only geometry."""
