# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

import carb

try:
    from omni.isaac.debug_draw import _debug_draw

    DRAW_INTERFACE_AVAILABLE = True
except ImportError:
    DRAW_INTERFACE_AVAILABLE = False


@dataclass
class _PointsInfo:
    """Internal dataclass to store information about points to be plotted.
    Attributes:
        start_pts: A list of tuples representing the positions of points.
        colors: A list of tuples representing the colors of points in RGBA format.
        sizes: A list of floats representing the sizes of points.
    """

    xyz: list[tuple[float, float, float]]
    colors: list[tuple[float, float, float, float]]
    sizes: list[float]


@dataclass
class _LinesInfo:
    """Internal dataclass to store information about lines to be plotted.
    Attributes:
        start_pts: A list of tuples representing the starting points of lines.
        end_pts: A list of tuples representing the ending points of lines.
        colors: A list of tuples representing the colors of lines in RGBA format.
        sizes: A list of floats representing the sizes of lines.
    """

    start_pts: list[tuple[float, float, float]]
    end_pts: list[tuple[float, float, float]]
    colors: list[tuple[float, float, float, float]]
    sizes: list[float]


class DrawingInterface:
    """Drawing Interface for orbit.
    This interface simplifies the process of drawing in the simulator by providing
    a higher-level abstraction. It caches drawing commands and refreshes them at
    regular intervals.
    Note:
        This interface should not be accessed directly. Instead, use the singleton
        `SimulationContext.instance().draw_interface` to access it.
    """

    def __init__(self):
        self._debug_draw_interface = None

        # Persistent data to draw (will not be erased with every sim step)
        self._persistent_lines = _LinesInfo([], [], [], [])

        # Data to draw (will be erased with every sim step)
        self._lines_to_draw = _LinesInfo([], [], [], [])

        self._persistent_points = _PointsInfo([], [], [])
        self._points_to_draw = _PointsInfo([], [], [])
        self._dirty = False

    @property
    def enabled(self) -> bool:
        """Returns whether the simulation is enabled.

        True if the simulation is enabled and running, False otherwise.
        """
        return DRAW_INTERFACE_AVAILABLE

    @property
    def draw_interface(self):
        """The debug_draw interface from Isaac Sim used internally."""
        if self._debug_draw_interface is None:
            if not DRAW_INTERFACE_AVAILABLE:
                carb.log_warn("Debug draw interface is not available but was requested. Drawing will not work.")
                import traceback

                traceback.print_stack()
                return None

            self._debug_draw_interface = _debug_draw.acquire_debug_draw_interface()
        return self._debug_draw_interface

    def update(self, remove_persistent: bool = False) -> None:
        """Updates the drawing interface by rendering the lines to be drawn.
        This function internally steps the following:
        1. Clears all drawn lines and points.
        2. Re-renders persistent data and new visualizations.
        3. Clears visualizations from the internal cache, only keeping persistent lines if `remove_persistent` is False.
        Args:
            remove_persistent: A boolean indicating whether to remove persistent lines.
        """

        if self._dirty:
            # Clear everything that has been drawn to the simulator before.
            self._clear()
            # Re-Render persistent data and new visualizations.z
            self._render()
            # Clear visualizations from the internal cache.
            self._reset(force=remove_persistent)
            self._dirty = False

    def plot_points(
        self,
        points: list[tuple[float, float, float]],
        color: tuple[float, float, float, float] | list[tuple[float, float, float, float]] = (1.0, 0.0, 0.0, 1.0),
        size: float | list[float] = 2.0,
        persistent: bool = False,
    ) -> None:
        """Plots points at the given positions.
        This function only caches the requested points, `update` needs to be called to actually write it to sim.
        Args:
            points: A list of tuples representing the positions of points (x, y, z).
            color: A tuple or a list of tuples representing the colors of points in RGBA format.
            size: A float or a list of floats representing the sizes of points.
            persistent: A boolean indicating whether the plotted points are persistent.
        Note:
            The length of `color` and `size` should match the length of `points`.
        """
        points_to_draw = self._points_to_draw if not persistent else self._persistent_points

        n_pts = len(points)
        if isinstance(color[0], (float, int)):
            # only one color specified for the full sequence
            color = [color] * n_pts

        if isinstance(size, (float, int)):
            # only one size specified for the full sequence
            size = [size] * n_pts

        if len(color) != n_pts:
            carb.log_warn(f"Number of points and colors to plot points do not match! {n_pts} != {len(color)}.")
            return

        if len(size) != n_pts:
            carb.log_warn(f"Number of points and sizes to plot points do not match! {n_pts} != {len(size)}.")
            return

        points_to_draw.xyz.extend(points)
        points_to_draw.colors.extend(color)
        points_to_draw.sizes.extend(size)
        self._dirty = True

    def _number_to_seven_segment(self, number: float, size=0.02, direction="y", start_pt=(0, 0, 0)) -> None:
        # convert to string
        number_str = f"{number:.2f}" if isinstance(number, float) else str(number)

        points = []
        lines = []

        if direction == "x":
            TOP_LINE = [[0, 0, 2 * size], [1 * size, 0, 2 * size]]
            TOP_RIGHT_LINE = [[1 * size, 0, 2 * size], [1 * size, 0, 1 * size]]
            BOTTOM_RIGHT_LINE = [[1 * size, 0, 1 * size], [1 * size, 0, 0]]
            BOTTOM_LINE = [[0, 0, 0], [1 * size, 0, 0]]
            BOTTOM_LEFT_LINE = [[0, 0, 0], [0, 0, 1 * size]]
            TOP_LEFT_LINE = [[0, 0, 1 * size], [0, 0, 2 * size]]
            MIDDLE_LINE = [[0, 0, 1 * size], [1 * size, 0, 1 * size]]
        elif direction == "y":
            TOP_LINE = [[0, 0, 2 * size], [0.0, 1 * size, 2 * size]]
            TOP_RIGHT_LINE = [[0, 1 * size, 2 * size], [0, 1 * size, 1 * size]]
            BOTTOM_RIGHT_LINE = [[0, 1 * size, 1 * size], [0, 1 * size, 0]]
            BOTTOM_LINE = [[0, 0, 0], [0, 1 * size, 0]]
            BOTTOM_LEFT_LINE = [[0, 0, 0], [0, 0, 1 * size]]
            TOP_LEFT_LINE = [[0, 0, 1 * size], [0, 0, 2 * size]]
            MIDDLE_LINE = [[0, 0, 1 * size], [0, 1 * size, 1 * size]]

        numbers_to_lines = {
            "0": [
                TOP_LINE,
                TOP_RIGHT_LINE,
                BOTTOM_RIGHT_LINE,
                BOTTOM_LINE,
                BOTTOM_LEFT_LINE,
                TOP_LEFT_LINE,
            ],
            "1": [TOP_RIGHT_LINE, BOTTOM_RIGHT_LINE],
            "2": [TOP_LINE, TOP_RIGHT_LINE, MIDDLE_LINE, BOTTOM_LEFT_LINE, BOTTOM_LINE],
            "3": [
                TOP_LINE,
                TOP_RIGHT_LINE,
                MIDDLE_LINE,
                BOTTOM_RIGHT_LINE,
                BOTTOM_LINE,
            ],
            "4": [TOP_LEFT_LINE, MIDDLE_LINE, TOP_RIGHT_LINE, BOTTOM_RIGHT_LINE],
            "5": [TOP_LINE, TOP_LEFT_LINE, MIDDLE_LINE, BOTTOM_RIGHT_LINE, BOTTOM_LINE],
            "6": [
                TOP_LINE,
                TOP_LEFT_LINE,
                MIDDLE_LINE,
                BOTTOM_RIGHT_LINE,
                BOTTOM_LINE,
                BOTTOM_LEFT_LINE,
            ],
            "7": [TOP_LINE, TOP_RIGHT_LINE, BOTTOM_RIGHT_LINE],
            "8": [
                TOP_LINE,
                TOP_RIGHT_LINE,
                BOTTOM_RIGHT_LINE,
                BOTTOM_LINE,
                BOTTOM_LEFT_LINE,
                TOP_LEFT_LINE,
                MIDDLE_LINE,
            ],
            "9": [
                TOP_LINE,
                TOP_RIGHT_LINE,
                BOTTOM_RIGHT_LINE,
                BOTTOM_LINE,
                TOP_LEFT_LINE,
                MIDDLE_LINE,
            ],
            "-": [MIDDLE_LINE],
        }

        spacing = 0.5 * size
        direction_idx = ord(direction) - ord("x")
        for idx, digit in enumerate(number_str):
            spacer = idx * size + idx * spacing

            if digit == ".":
                pt = [s for s in start_pt]
                pt[direction_idx] += spacer
                points.append(pt)
                continue
            if digit not in numbers_to_lines:
                continue

            for line in numbers_to_lines[digit]:
                start, end = line[0], line[1]
                pts = [
                    [
                        start[0] + start_pt[0],
                        start[1] + start_pt[1],
                        start[2] + start_pt[2],
                    ],
                    [end[0] + start_pt[0], end[1] + start_pt[1], end[2] + start_pt[2]],
                ]
                pts[0][direction_idx] += spacer
                pts[1][direction_idx] += spacer
                lines.append(pts)

        return points, lines

    def draw_numbers(
        self,
        points: list[tuple[float, float, float]],
        numbers: list[float],
        color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        size: float = 2.0,
        persistent: bool = False,
        up="z",
        direction="x",
    ) -> None:
        axes = {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        }
        x3 = axes[direction]
        x1 = axes[up]
        x2 = np.cross(x3, x1)
        rotmat = np.array([x1, x2, x3]).T

        for idx, (number, point) in enumerate(zip(numbers, points)):
            pts, lines = self._number_to_seven_segment(number, size, start_pt=point)
            if len(pts) == 0:
                continue

            pts_np = np.array(pts) - point  # Shape: (n, 3)
            lines_np = np.array(lines) - point  # Shape: (n, 2, 3)
            pts = ((rotmat @ pts_np[..., None]).squeeze(-1) + point).tolist()
            lines = ((rotmat @ lines_np[..., None]).squeeze(-1) + point).tolist()

            # rotate them to match convention

            if isinstance(color[0], list):
                col = color[idx]
            else:
                col = color
            self.plot_points(pts, col, 4.0, persistent)
            self.plot_lines(
                [line[0] for line in lines],
                [line[1] for line in lines],
                col,
                4.0,
                persistent,
            )

    def plot_lines(
        self,
        start_pts: list[tuple[float, float, float]],
        end_pts: list[tuple[float, float, float]],
        color: tuple[float, float, float, float] | list[tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0),
        size: float | list[float] = 2.0,
        persistent: bool = False,
    ) -> None:
        """Plots lines between given start and end points.
        This function only caches the requested lines, `update` needs to be called to actually write it to sim.
        Args:
            start_pts: A list of tuples representing the starting points of lines (x,y,z).
            end_pts: A list of tuples representing the ending points of lines (x, y, z).
            color: A tuple or a list of tuples representing the colors of lines in RGBA format.
            size: A float or a list of floats representing the sizes of lines.
            persistent: A boolean indicating whether the plotted lines are persistent.
        Note:
            The length of `color` and `size` should match the length of `start_pts`
            and `end_pts`.
        """
        lines_to_draw = self._lines_to_draw if not persistent else self._persistent_lines

        n_pts = len(start_pts)
        if n_pts != len(end_pts):
            carb.log_warn(f"Number of start and end-points to plot lines do not match! {n_pts} != {len(end_pts)}.")
            return

        if isinstance(end_pts, np.ndarray):
            end_pts = end_pts.tolist()
            start_pts = start_pts.tolist()

        if isinstance(color[0], float):
            # only one color specified for the full sequence
            color = [color] * n_pts

        if isinstance(size, float):
            # only one size specified for the full sequence
            size = [size] * n_pts

        if len(color) != n_pts:
            carb.log_warn(f"Number of points and colors to plot lines do not match! {n_pts} != {len(color)}.")
            return

        if len(size) != n_pts:
            carb.log_warn(f"Number of points and sizes to plot lines do not match! {n_pts} != {len(size)}.")
            return

        lines_to_draw.start_pts.extend(start_pts)
        lines_to_draw.end_pts.extend(end_pts)
        lines_to_draw.colors.extend(color)
        lines_to_draw.sizes.extend(size)
        self._dirty = True

    ###
    # Internal Methods
    ###

    def _draw_points(self, points_to_plot: _PointsInfo) -> None:
        """Draws the points to the debug draw interface."""
        if self.draw_interface is not None and len(points_to_plot.xyz) > 0:
            self.draw_interface.draw_points(
                points_to_plot.xyz,
                points_to_plot.colors,
                points_to_plot.sizes,
            )

    def _draw_lines(self, lines_to_plot: _LinesInfo) -> None:
        """Draws the lines to the debug draw interface."""
        if self.draw_interface is not None and len(lines_to_plot.start_pts) > 0:
            self.draw_interface.draw_lines(
                lines_to_plot.start_pts,
                lines_to_plot.end_pts,
                lines_to_plot.colors,
                lines_to_plot.sizes,
            )

    def _clear(self) -> None:
        """Clears all drawn lines and points from isaac sim."""
        if self.draw_interface is not None:
            self.draw_interface.clear_lines()
            self.draw_interface.clear_points()

    def _reset(self, force: bool = False) -> None:
        """Resets the lines to draw.
        Args:
            force: A boolean indicating whether to force a reset of persistent lines.
        """
        self._lines_to_draw = _LinesInfo([], [], [], [])
        self._points_to_draw = _PointsInfo([], [], [])

        if force:
            self._persistent_lines = _LinesInfo([], [], [], [])
            self._persistent_points = _PointsInfo([], [], [])

    def _render(self) -> None:
        """Renders the lines to be drawn."""
        self._draw_lines(self._lines_to_draw)
        self._draw_lines(self._persistent_lines)
        self._draw_points(self._points_to_draw)
        self._draw_points(self._persistent_points)
