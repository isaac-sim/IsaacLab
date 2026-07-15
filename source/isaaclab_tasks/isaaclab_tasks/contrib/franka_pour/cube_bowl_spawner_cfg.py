# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka pour cube-bowl USD spawner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class CubeBowlSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for a visual cube bowl with an optional rigid grasp proxy."""

    func: Callable | str = "{DIR}.cube_bowl_spawner:spawn_cube_bowl"

    inner_width: float = MISSING
    """Inner cavity size along x [m]."""

    inner_depth: float = MISSING
    """Inner cavity size along y [m]."""

    cavity_depth: float = MISSING
    """Cavity height from the inner floor to the rim [m]."""

    wall_thickness: float = MISSING
    """Side-wall thickness [m]."""

    bottom_thickness: float = MISSING
    """Base thickness below the inner floor [m]."""

    display_color: tuple[float, float, float] = (0.95, 0.82, 0.16)
    """RGB display color in linear color space."""

    grasp_proxy_half_extents: tuple[float, float, float] | None = None
    """Optional collision-proxy half extents along x, y, and z [m]."""

    physics_material_path: str = "material"
    """Path of the rigid-body physics material.

    A relative path is resolved below the bowl's ``geometry`` prim.
    """

    physics_material: RigidBodyMaterialBaseCfg | None = None
    """Optional rigid-body physics material for contact metadata."""
