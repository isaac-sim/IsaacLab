# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hf_terrains_cfg import HfTerrainBaseCfg


def height_field_to_mesh(func: Callable) -> Callable:
    """Decorator to convert a height field function to a mesh function.

    This decorator converts a height field function to a mesh function by sampling the heights
    at a specified resolution and performing interpolation to obtain the intermediate heights.
    Additionally, it adds a border around the terrain to avoid artifacts at the edges.

    Args:
        func: The height field function to convert. The function should return a 2D numpy array
            with the heights of the terrain.

    Returns:
        The mesh function. The mesh function returns a tuple containing a list of ``trimesh``
        mesh objects and the origin of the terrain.
    """

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        # check valid border width
        if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
            raise ValueError(
                f"The border width ({cfg.border_width}) must be greater than or equal to the"
                f" horizontal scale ({cfg.horizontal_scale})."
            )
        # allocate buffer for height field (with border)
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        # override size of the terrain to account for the border
        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * cfg.horizontal_scale for dim in sub_terrain_size]
        # update the config
        terrain_size = copy.deepcopy(cfg.size)
        cfg.size = tuple(sub_terrain_size)
        # generate the height field
        z_gen = func(difficulty, cfg)
        # handle the border for the terrain
        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen
        # set terrain size back to config
        cfg.size = terrain_size

        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # compute origin
        x1 = int((cfg.size[0] * 0.5 - 1) / cfg.horizontal_scale)
        x2 = int((cfg.size[0] * 0.5 + 1) / cfg.horizontal_scale)
        y1 = int((cfg.size[1] * 0.5 - 1) / cfg.horizontal_scale)
        y2 = int((cfg.size[1] * 0.5 + 1) / cfg.horizontal_scale)
        origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
        origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])
        # return mesh and origin
        return [mesh], origin

    return wrapper


def convert_height_field_to_mesh(
    height_field: np.ndarray, horizontal_scale: float, vertical_scale: float, slope_threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a height-field array to a triangle mesh represented by vertices and triangles.

    This function converts a height-field array to a triangle mesh represented by vertices and triangles.
    The height-field array is assumed to be a 2D array of floats, where each element represents the height
    of the terrain at that location. The height-field array is assumed to be in the form of a matrix, where
    the first dimension represents the x-axis and the second dimension represents the y-axis.

    The function can also correct vertical surfaces above the provide slope threshold. This is helpful to
    avoid having long vertical surfaces in the mesh. The correction is done by moving the vertices of the
    vertical surfaces to minimum of the two neighboring vertices.

    The correction is done in the following way:
    If :math:`\\frac{y_2 - y_1}{x_2 - x_1} > threshold`, then move A to A' (i.e., set :math:`x_1' = x_2`).
    This is repeated along all directions.

    .. code-block:: none

                B(x_2,y_2)
                    /|
                   / |
                  /  |
        (x_1,y_1)A---A'(x_1',y_1)

    Args:
        height_field: The input height-field array.
        horizontal_scale: The discretization of the terrain along the x and y axis.
        vertical_scale: The discretization of the terrain along the z axis.
        slope_threshold: The slope threshold above which surfaces are made vertical.
            Defaults to None, in which case no correction is applied.

    Returns:
        The vertices and triangles of the mesh:
        - **vertices** (np.ndarray(float)): Array of shape (num_vertices, 3).
          Each row represents the location of each vertex (in m).
        - **triangles** (np.ndarray(int)): Array of shape (num_triangles, 3).
          Each row represents the indices of the 3 vertices connected by this triangle.
    """
    # read height field
    num_rows, num_cols = height_field.shape
    # create a mesh grid of the height field
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    # copy height field to avoid modifying the original array
    hf = height_field.copy()

    # correct vertical surfaces above the slope threshold
    if slope_threshold is not None:
        # scale slope threshold based on the horizontal and vertical scale
        slope_threshold *= horizontal_scale / vertical_scale
        # allocate arrays to store the movement of the vertices
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        # move vertices along the x-axis
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        # move vertices along the y-axis
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        # move vertices along the corners
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create vertices for the mesh
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    # create triangles for the mesh
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles
