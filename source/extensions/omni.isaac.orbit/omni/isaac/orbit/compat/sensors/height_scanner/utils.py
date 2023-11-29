# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities to create and visualize 2D height-maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def create_points_from_grid(size: tuple[float, float], resolution: float) -> np.ndarray:
    """Creates a list of points from 2D mesh-grid.

    The terrain scan is approximated with a grid map of the input resolution.
    By default, we consider the origin as the center of the local map and the scan size ``(X, Y)`` is the
    map size. Given these settings, the elevation map spans from: ``(- X / 2, - Y / 2)`` to
    ``(+ X / 2, + Y / 2)``.

    Example:
        For a grid of size (0.2, 0.2) with resolution of 0.1, the created points will first x-axis fixed, while the
        y-axis changes, i.e.:

        .. code-block:: none

            [
                [-0.1, -0.1], [-0.1, 0.0], [-0.1, 0.1],
                [0.0, -0.1], [0.0, 0.], [0.0, 0.1],
                [0.1, -0.1], [0.1, 0.0], [0.1, 0.1],
            ]

    Args:
        size: The 2D scan region along x and y directions (in meters).
        resolution: The resolution of the scanner (in meters/cell).

    Returns:
        A set of points of shape (N, 2) or (N, 3), where first x is fixed while y changes.
    """
    # Compute the scan grid
    # Note: np.arange does not include end-point when dealing with floats. That is why we add resolution.
    x = np.arange(-size[0] / 2, size[0] / 2 + resolution, resolution)
    y = np.arange(-size[1] / 2, size[1] / 2 + resolution, resolution)
    grid = np.meshgrid(x, y, sparse=False, indexing="ij")
    # Concatenate the scan grid into points array (N, 2): first x is fixed while y changes
    return np.vstack(list(map(np.ravel, grid))).T


def plot_height_grid(
    hit_distance: np.ndarray, size: tuple[float, float], resolution: float, ax: Axes = None
) -> AxesImage:
    """Plots the sensor height-map distances using matplotlib.

    If the axes is not provided, a new figure is created.

    Note:
        This method currently only supports if the grid is evenly spaced, i.e. the scan points are created using
        :meth:`create_points_from_grid` method.

    Args:
        hit_distance: The ray hit distance measured from the sensor.
        size: The 2D scan region along x and y directions (in meters).
        resolution: The resolution of the scanner (in meters/cell).
        ax: The current matplotlib axes to plot in.. Defaults to None.

    Returns:
        Image axes of the created plot.
    """
    # Check that request of keys has same length as available axes.
    if ax is None:
        # Create axes if not provided
        # Setup a figure
        _, ax = plt.subplots()
    # turn axes off
    ax.clear()
    # resolve shape of the heightmap
    x = np.arange(-size[0] / 2, size[0] / 2 + resolution, resolution)
    y = np.arange(-size[1] / 2, size[1] / 2 + resolution, resolution)
    shape = (len(x), len(y))
    # convert the map shape
    heightmap = hit_distance.reshape(shape)
    # plot the scanned distance
    caxes = ax.imshow(heightmap, cmap="turbo", interpolation="none", vmin=0)
    # set the label
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    # set the ticks
    ax.set_xticks(np.arange(shape[1]), minor=False)
    ax.set_yticks(np.arange(shape[0]), minor=False)
    ax.set_xticklabels([round(value, 2) for value in y])
    ax.set_yticklabels([round(value, 2) for value in x])
    # add grid
    ax.grid(color="w", linestyle="--", linewidth=1)
    # return the color axes
    return caxes
