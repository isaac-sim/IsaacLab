# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three-box terrain with a flat start, inclined ramp and runout.

Difficulty interpolates the ramp angle from 2 to 20 degrees. The origin lies on the incline
so origin-relative robot resets place the robot on the ramp.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

MICRODUCK_RAMP_DEG_MIN = 2.0
"""Ramp angle [deg] at difficulty 0 -- the gentlest slope the curriculum starts every robot on."""

MICRODUCK_RAMP_DEG_MAX = 20.0
"""Ramp angle [deg] at difficulty 1 -- the steepest slope the curriculum promotes towards."""


def flat_ramp_terrain(difficulty: float, cfg: FlatRampTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a starting platform, a descending ramp and a run-out platform.

    Note:
        The draw comes from the global :mod:`numpy` random state, because Isaac Lab sub-terrain
        functions take no generator and the terrain generator seeds only its own. That is how the
        stock ``repeated_objects_terrain`` behaves as well. Pin
        :attr:`FlatRampTerrainCfg.ramp_length_range` to a single value when a tile must be
        reproducible.

    Args:
        difficulty: The difficulty of the terrain, between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The three boxes making up the tile, and the spawn origin [m] on the incline.

    Raises:
        ValueError: If the longest configured ramp does not fit in the tile along ``x``.
    """
    longest = cfg.flat_length + cfg.ramp_length_range[1] + cfg.runout_length
    if longest > cfg.size[0]:
        raise ValueError(
            f"The longest flat-ramp tile does not fit its sub-terrain: 'flat_length' +"
            f" max 'ramp_length_range' + 'runout_length' is {longest} m but 'size[0]' is"
            f" {cfg.size[0]} m."
        )

    angle = math.radians(cfg.deg_min + float(np.clip(difficulty, 0.0, 1.0)) * (cfg.deg_max - cfg.deg_min))
    ramp_length = float(np.random.uniform(cfg.ramp_length_range[0], cfg.ramp_length_range[1]))
    drop = ramp_length * math.tan(angle)
    width = cfg.size[1]
    thickness = cfg.thickness

    # the starting platform: surface at z = 0, spanning x in [0, flat_length]
    flat = trimesh.creation.box(
        (cfg.flat_length, width, thickness),
        trimesh.transformations.translation_matrix((cfg.flat_length / 2.0, width / 2.0, -thickness / 2.0)),
    )

    # the ramp: rotated about +y so that the +x edge descends
    ramp_transform = np.eye(4)
    ramp_transform[0:3, 0:3] = trimesh.transformations.rotation_matrix(angle, (0.0, 1.0, 0.0))[0:3, 0:3]
    ramp_transform[0:3, -1] = (
        cfg.flat_length + ramp_length / 2.0 - (thickness / 2.0) * math.sin(angle),
        width / 2.0,
        -(drop / 2.0) - (thickness / 2.0) * math.cos(angle),
    )
    ramp = trimesh.creation.box((ramp_length / math.cos(angle), width, thickness), ramp_transform)

    # the run-out platform: surface level with the foot of the ramp
    runout = trimesh.creation.box(
        (cfg.runout_length, width, thickness),
        trimesh.transformations.translation_matrix(
            (
                cfg.flat_length + ramp_length + cfg.runout_length / 2.0,
                width / 2.0,
                -drop - thickness / 2.0,
            )
        ),
    )

    # Spawn a little way onto the ramp: the robot is already on the slope, so gravity spins the
    # wheels up instead of the reset having to shove a base that would then skid.
    origin = np.array(
        [
            cfg.flat_length + cfg.spawn_on_ramp,
            width / 2.0,
            -cfg.spawn_on_ramp * math.tan(angle),
        ]
    )
    return [flat, ramp, runout], origin


@configclass
class FlatRampTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a starting platform, a descending ramp and a run-out platform."""

    function = flat_ramp_terrain

    convert_to_heightfield: bool = True
    """Collide against a heightfield rather than the raw triangle mesh. Defaults to True.

    The stock default is ``False``, and this is the one field that departs from it. Two reasons, and
    the second is measured rather than stylistic:
    """

    flat_length: float = 2.0
    """Length [m] of the starting platform, whose surface is at ``z = 0``. Defaults to 2.0."""

    ramp_length_range: tuple[float, float] = (3.0, 8.0)
    """Range of **horizontal** ramp lengths [m], drawn once per tile. Defaults to (3.0, 8.0).

    Pin both ends to the same value to make a tile reproducible.
    """

    runout_length: float = 4.0
    """Length [m] of the platform at the foot of the ramp. Defaults to 4.0.

    It exists so that reaching the bottom is not the same thing as falling off the terrain.
    """

    spawn_on_ramp: float = 0.3
    """Distance [m] past the top of the ramp at which the origin sits. Defaults to 0.3.

    Far enough onto the incline that gravity starts the descent on its own.
    """

    deg_min: float = MICRODUCK_RAMP_DEG_MIN
    """Ramp angle [deg] at difficulty 0. Defaults to :data:`MICRODUCK_RAMP_DEG_MIN`."""

    deg_max: float = MICRODUCK_RAMP_DEG_MAX
    """Ramp angle [deg] at difficulty 1. Defaults to :data:`MICRODUCK_RAMP_DEG_MAX`."""

    thickness: float = 0.5
    """Thickness [m] of the three boxes. Defaults to 0.5.

    It only has to be deep enough that nothing tunnels through; it is also what the ramp's
    ``x`` shim is derived from.
    """
