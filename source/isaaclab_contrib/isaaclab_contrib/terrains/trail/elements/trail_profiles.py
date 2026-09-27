# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

"""
This module contains functions to generate 2D profiles that represent the
shape of the trail.

When creating new objects, consider the following:
     * Objects are defined in the yz plane.
     * The y axis points to the right and the z axis points downward (z
        coordinates are flipped).
     * The object's anchor point is at (y, z) = (0, 0).
     * The object is parameterized as a NumPy array with shape (N, 2).
     * The outline of the object is not closed (i.e., the first and last point
        are not identical).

                    y=0
         +--------+----------------------> y
         |        |
         |        |
      z=0+--------+ (beginning of object)
         |
         |
         z
"""
import random
from typing import TYPE_CHECKING

import numpy as np

from ..utils.math import sample, sample_sign
from . import trail_walls as walls

if TYPE_CHECKING:
    from ..trail_cfg import WallParameters


def trail_profile(
    width: float,
    thickness: float,
    wp: WallParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Helper that generates the trail profile polygon.

    Notes:
        * The trail starts at polygon coordinates y=0, z=0.
        * The z axis is flipped (points downward).
        * The trail is swept along the positive x axis.

    The picture below illustrates the outline polygon with linear walls
    and wall_height > 0::

               y=0
       +--------+-------------------------> y
       |   +--+             +--+
       |   |   \           /   |
    z=0+   |    +---------+    |
       |   |    |         |    |
       |   |    +---------+    |
       |   +-------------------+
       |
       z

    Args:
        width: Width of the trail (excluding walls) [m].
        thickness: Thickness of the ground plane [m].
        wp: WallParameters instance used to parameterize the trail walls.

    Returns:
        A tuple containing:
            * A polygon specifying the profile outline, parameterized as a
              NumPy array with shape (N, 2).
            * The indices corresponding to the left and right border of the
              outline.
            * The indices corresponding to all non-trail vertices.
    """
    # Dummy border width (used so we can later extend vertices corresponding to the border).
    # Do not set this to zero; otherwise trimesh may remove the vertex.
    border_width = 0.01

    # Select the wall signs (+1 means up, -1 means down).
    # Sample wall direction from the configured probability distribution.
    wall_dir_options = wp.wall_direction
    wall_directions = list(wall_dir_options.keys())
    wall_dir_weights = list(wall_dir_options.values())
    wall_direction = random.choices(wall_directions, weights=wall_dir_weights, k=1)[0]

    if wall_direction == "up":
        wall_sign = {"left": 1.0, "right": 1.0}
    elif wall_direction == "down":
        wall_sign = {"left": -1.0, "right": -1.0}
    elif wall_direction == "up-down":
        s = sample_sign()
        wall_sign = {"left": s, "right": -s}
    else:
        raise RuntimeError(f"Wall direction: '{wall_direction}' is not supported.")

    # Sample wall parameters
    def sample_wall(wp: WallParameters) -> tuple[np.ndarray, np.ndarray, bool]:
        # Pick a random wall function, weighted by configured probabilities.
        wall_options = wp.wall_functions
        wall_functions = list(wall_options.keys())
        wall_weights = list(wall_options.values())
        wf_name = random.choices(wall_functions, weights=wall_weights, k=1)[0]

        if not hasattr(walls, wf_name):
            raise RuntimeError(f"Unknown wall function '{wf_name}'.")
        wf = getattr(walls, wf_name)

        # Compute wall points
        return wf(
            wall_width=sample(wp.wall_dim["width"]),
            wall_height=sample(wp.wall_dim["height"]),
            num_segments=sample(wp.num_segments),
        )

    wall_left_yi, wall_left_zi, is_smooth_left = sample_wall(wp=wp)
    wall_right_yi, wall_right_zi, is_smooth_right = sample_wall(wp=wp)

    # Decide whether walls are part of the trail segmentation
    include_walls_in_trail = {
        "left": wall_sign["left"] > 0.0 and is_smooth_left,
        "right": wall_sign["right"] > 0.0 and is_smooth_right,
    }

    # increase trail thickness if the walls are facing down
    if wall_sign["left"] < 0.0 or wall_sign["right"] < 0.0:
        thickness += wp.wall_dim["height"] if isinstance(wp.wall_dim["height"], float) else max(wp.wall_dim["height"])

    # remove the first point of the wall (it is part of the floor already)
    wall_left_yi = wall_left_yi[1:]
    wall_right_yi = wall_right_yi[1:]
    wall_left_zi = wall_left_zi[1:]
    wall_right_zi = wall_right_zi[1:]

    # process with sign
    wall_right_zi *= wall_sign["right"]
    wall_left_zi *= wall_sign["left"]

    # define the knot points of the trail outline
    floor_i = np.linspace(0.0, 1.0, sample(wp.num_segments_floor))
    yi = np.concatenate(
        [
            floor_i * width,
            width + wall_left_yi,
            [
                width + wall_left_yi[-1] + border_width,
                width + wall_left_yi[-1] + border_width,
                -wall_right_yi[-1] - border_width,
                -wall_right_yi[-1] - border_width,
            ],
            -np.flip(wall_right_yi),
        ],
        axis=0,
    )

    zi = np.concatenate(
        [
            floor_i * 0.0,
            -wall_left_zi,
            [-wall_left_zi[-1], thickness, thickness, -wall_right_zi[-1]],
            -np.flip(wall_right_zi),
        ],
        axis=0,
    )

    # remember border indices
    start_id = len(floor_i) + len(wall_left_yi)
    border_ids = np.array([start_id, start_id + 1, start_id + 2, start_id + 3])

    # remember all indices not part of the trail
    if include_walls_in_trail["left"]:
        non_trail_ids_left = np.array([start_id - 1, start_id, start_id + 1])
    else:
        non_trail_ids_left = np.arange(start_id - len(wall_left_yi), start_id + 1)

    if include_walls_in_trail["right"]:
        non_trail_ids_right = np.array([start_id + 2, start_id + 3, start_id + 4])
    else:
        non_trail_ids_right = np.arange(start_id + 2, start_id + 4 + len(wall_right_yi))
    non_trail_ids = np.concatenate([non_trail_ids_right, non_trail_ids_left], axis=0)

    return (np.stack([yi, zi], axis=1), border_ids, non_trail_ids)
