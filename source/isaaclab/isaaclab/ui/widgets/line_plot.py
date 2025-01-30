# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import colorsys
import numpy as np
from contextlib import suppress
from typing import TYPE_CHECKING

import omni
from isaacsim.core.api.simulation_context import SimulationContext

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils

from .ui_widget_wrapper import UIWidgetWrapper

if TYPE_CHECKING:
    import isaacsim.gui.components
    import omni.ui


class LiveLinePlot(UIWidgetWrapper):
    """A 2D line plot widget to display live data.


    This widget is used to display live data in a 2D line plot. It can be used to display multiple series
    in the same plot.

    It has the following Layout:
    +-------------------------------------------------------+
    |                  containing_frame                     |
    |+-----------------------------------------------------+|
    |                   main_plot_frame                     |
    ||+---------------------------------------------------+||
    |||         plot_frames + grid lines (Z_stacked)      |||
    |||                                                   |||
    |||                                                   |||
    |||               (Live Plot Data)                    |||
    |||                                                   |||
    |||                                                   |||
    |||+-------------------------------------------------+|||
    |||                   legends_frame                   |||
    |||                                                   |||
    |||    [x][Series 1] [x][Series 2] [ ][Series 3]      |||
    |||+-------------------------------------------------+|||
    |||                   limits_frame                    |||
    |||                                                   |||
    |||        [Y-Limits] [min] [max] [Autoscale]         |||
    |||+-------------------------------------------------+|||
    |||                   filter_frame                    |||
    |||                                                   |||
    |||                                                   |||
    |+-----------------------------------------------------+|
    +-------------------------------------------------------+

    """

    def __init__(
        self,
        y_data: list[list[float]],
        y_min: float = -10,
        y_max: float = 10,
        plot_height: int = 150,
        show_legend: bool = True,
        legends: list[str] | None = None,
        max_datapoints: int = 200,
    ):
        """Create a new LiveLinePlot widget.

        Args:
            y_data: A list of lists of floats containing the data to plot. Each list of floats represents a series in the plot.
            y_min: The minimum y value to display. Defaults to -10.
            y_max: The maximum y value to display. Defaults to 10.
            plot_height: The height of the plot in pixels. Defaults to 150.
            show_legend: Whether to display the legend. Defaults to True.
            legends: A list of strings containing the legend labels for each series. If None, the default labels are "Series_0", "Series_1", etc. Defaults to None.
            max_datapoints: The maximum number of data points to display. If the number of data points exceeds this value, the oldest data points are removed. Defaults to 200.
        """
        super().__init__(self._create_ui_widget())
        self.plot_height = plot_height
        self.show_legend = show_legend
        self._legends = legends if legends is not None else ["Series_" + str(i) for i in range(len(y_data))]
        self._y_data = y_data
        self._colors = self._get_distinct_hex_colors(len(y_data))
        self._y_min = y_min if y_min is not None else -10
        self._y_max = y_max if y_max is not None else 10
        self._max_data_points = max_datapoints
        self._show_legend = show_legend
        self._series_visible = [True for _ in range(len(y_data))]
        self._plot_frames = []
        self._plots = []
        self._plot_selected_values = []
        self._is_built = False
        self._filter_frame = None
        self._filter_mode = None
        self._last_values = None
        self._is_paused = False

        # Gets populated when widget is built
        self._main_plot_frame = None

        self._autoscale_model = omni.ui.SimpleBoolModel(True)

    """Properties"""

    @property
    def autoscale_mode(self) -> bool:
        return self._autoscale_model.as_bool

    @property
    def y_data(self) -> list[list[float]]:
        """The current data in the plot."""
        return self._y_data

    @property
    def y_min(self) -> float:
        """The current minimum y value."""
        return self._y_min

    @property
    def y_max(self) -> float:
        """The current maximum y value."""
        return self._y_max

    @property
    def legends(self) -> list[str]:
        """The current legend labels."""
        return self._legends

    """ General Functions """

    def clear(self):
        """Clears the plot."""
        self._y_data = [[] for _ in range(len(self._y_data))]
        self._last_values = None

        for plt in self._plots:
            plt.set_data()

        # self._container_frame.rebuild()

    def add_datapoint(self, y_coords: list[float]):
        """Add a data point to the plot.

        The data point is added to the end of the plot. If the number of data points exceeds the maximum number
        of data points, the oldest data point is removed.

        ``y_coords`` is assumed to be a list of floats with the same length as the number of series in the plot.

        Args:
            y_coords: A list of floats containing the y coordinates of the new data points.
        """

        for idx, y_coord in enumerate(y_coords):

            if len(self._y_data[idx]) > self._max_data_points:
                self._y_data[idx] = self._y_data[idx][1:]

            if self._filter_mode == "Lowpass":
                if self._last_values is not None:
                    alpha = 0.8
                    y_coord = self._y_data[idx][-1] * alpha + y_coord * (1 - alpha)
            elif self._filter_mode == "Integrate":
                if self._last_values is not None:
                    y_coord = self._y_data[idx][-1] + y_coord
            elif self._filter_mode == "Derivative":
                if self._last_values is not None:
                    y_coord = (y_coord - self._last_values[idx]) / SimulationContext.instance().get_rendering_dt()

            self._y_data[idx].append(float(y_coord))

        if self._main_plot_frame is None:
            # Widget not built, not visible
            return

        # Check if the widget has been built, i.e. the plot references have been created.
        if not self._is_built or self._is_paused:
            return

        if len(self._y_data) != len(self._plots):
            # Plots gotten out of sync, rebuild the widget
            self._main_plot_frame.rebuild()
            return

        if self.autoscale_mode:
            self._rescale_btn_pressed()

        for idx, plt in enumerate(self._plots):
            plt.set_data(*self._y_data[idx])

        self._last_values = y_coords
        # Autoscale the y-axis to the current data

    """
        Internal functions for building the UI.
    """

    def _build_stacked_plots(self, grid: bool = True):
        """Builds multiple plots stacked on top of each other to display multiple series.

        This is an internal function to build the plots. It should not be called from outside the class and only
        from within the build function of a frame.

        The built widget has the following layout:
        +-------------------------------------------------------+
        |                   main_plot_frame                     |
        ||+---------------------------------------------------+||
        |||                                                   |||
        ||| y_max|*******-------------------*******|          |||
        |||      |-------*****-----------**--------|          |||
        |||     0|------------**-----***-----------|          |||
        |||      |--------------***----------------|          |||
        ||| y_min|---------------------------------|          |||
        |||                                                   |||
        |||+-------------------------------------------------+|||


        Args:
            grid: Whether to display grid lines. Defaults to True.
        """

        # Reset lists which are populated in the build function
        self._plot_frames = []

        # Define internal builder function
        def _build_single_plot(y_data: list[float], color: int, plot_idx: int):
            """Build a single plot.

            This is an internal function to build a single plot with the given data and color. This function
            should only be called from within the build function of a frame.

            Args:
                y_data: The data to plot.
                color: The color of the plot.
            """
            plot = omni.ui.Plot(
                omni.ui.Type.LINE,
                self._y_min,
                self._y_max,
                *y_data,
                height=self.plot_height,
                style={"color": color, "background_color": 0x0},
            )

            if len(self._plots) <= plot_idx:
                self._plots.append(plot)
                self._plot_selected_values.append(omni.ui.SimpleStringModel(""))
            else:
                self._plots[plot_idx] = plot

        # Begin building the widget
        with omni.ui.HStack():
            # Space to the left to add y-axis labels
            omni.ui.Spacer(width=20)

            # Built plots for each time series stacked on top of each other
            with omni.ui.ZStack():
                # Background rectangle
                omni.ui.Rectangle(
                    height=self.plot_height,
                    style={
                        "background_color": 0x0,
                        "border_color": omni.ui.color.white,
                        "border_width": 0.4,
                        "margin": 0.0,
                    },
                )

                # Draw grid lines and labels
                if grid:
                    # Calculate the number of grid lines to display
                    # Absolute range of the plot
                    plot_range = self._y_max - self._y_min
                    grid_resolution = 10 ** np.floor(np.log10(0.5 * plot_range))

                    plot_range /= grid_resolution

                    # Fraction of the plot range occupied by the first and last grid line
                    first_space = (self._y_max / grid_resolution) - np.floor(self._y_max / grid_resolution)
                    last_space = np.ceil(self._y_min / grid_resolution) - self._y_min / grid_resolution

                    # Number of grid lines to display
                    n_lines = int(plot_range - first_space - last_space)

                    plot_resolution = self.plot_height / plot_range

                    with omni.ui.VStack():
                        omni.ui.Spacer(height=plot_resolution * first_space)

                        # Draw grid lines
                        with omni.ui.VGrid(row_height=plot_resolution):
                            for grid_line_idx in range(n_lines):
                                # Create grid line
                                with omni.ui.ZStack():
                                    omni.ui.Line(
                                        style={
                                            "color": 0xAA8A8777,
                                            "background_color": 0x0,
                                            "border_width": 0.4,
                                        },
                                        alignment=omni.ui.Alignment.CENTER_TOP,
                                        height=0,
                                    )
                                    with omni.ui.Placer(offset_x=-20):
                                        omni.ui.Label(
                                            f"{(self._y_max - first_space * grid_resolution - grid_line_idx * grid_resolution):.3f}",
                                            width=8,
                                            height=8,
                                            alignment=omni.ui.Alignment.RIGHT_TOP,
                                            style={
                                                "color": 0xFFFFFFFF,
                                                "font_size": 8,
                                            },
                                        )

                # Create plots for each series
                for idx, (data, color) in enumerate(zip(self._y_data, self._colors)):
                    plot_frame = omni.ui.Frame(
                        build_fn=lambda y_data=data, plot_idx=idx, color=color: _build_single_plot(
                            y_data, color, plot_idx
                        ),
                    )
                    plot_frame.visible = self._series_visible[idx]
                    self._plot_frames.append(plot_frame)

                # Create an invisible frame on top that will give a helpful tooltip
                self._tooltip_frame = omni.ui.Plot(
                    height=self.plot_height,
                    style={"color": 0xFFFFFFFF, "background_color": 0x0},
                )

                self._tooltip_frame.set_mouse_pressed_fn(self._mouse_moved_on_plot)

                # Create top label for the y-axis
                with omni.ui.Placer(offset_x=-20, offset_y=-8):
                    omni.ui.Label(
                        f"{self._y_max:.3f}",
                        width=8,
                        height=2,
                        alignment=omni.ui.Alignment.LEFT_TOP,
                        style={"color": 0xFFFFFFFF, "font_size": 8},
                    )

                # Create bottom label for the y-axis
                with omni.ui.Placer(offset_x=-20, offset_y=self.plot_height):
                    omni.ui.Label(
                        f"{self._y_min:.3f}",
                        width=8,
                        height=2,
                        alignment=omni.ui.Alignment.LEFT_BOTTOM,
                        style={"color": 0xFFFFFFFF, "font_size": 8},
                    )

    def _mouse_moved_on_plot(self, x, y, *args):
        # Show a tooltip with x,y and function values
        if len(self._y_data) == 0 or len(self._y_data[0]) == 0:
            # There is no data in the plots, so do nothing
            return

        for idx, plot in enumerate(self._plots):
            x_pos = plot.screen_position_x
            width = plot.computed_width

            location_x = (x - x_pos) / width

            data = self._y_data[idx]
            n_samples = len(data)
            selected_sample = int(location_x * n_samples)
            value = data[selected_sample]
            # save the value in scientific notation
            self._plot_selected_values[idx].set_value(f"{value:.3f}")

    def _build_legends_frame(self):
        """Build the frame containing the legend for the plots.

        This is an internal function to build the frame containing the legend for the plots. This function
        should only be called from within the build function of a frame.

        The built widget has the following layout:
        +-------------------------------------------------------+
        |                   legends_frame                       |
        ||+---------------------------------------------------+||
        |||                                                   |||
        |||    [x][Series 1] [x][Series 2] [ ][Series 3]      |||
        |||                                                   |||
        |||+-------------------------------------------------+|||
        |+-----------------------------------------------------+|
        +-------------------------------------------------------+
        """
        if not self._show_legend:
            return

        with omni.ui.HStack():
            omni.ui.Spacer(width=32)

            # Find the longest legend to determine the width of the frame
            max_legend = max([len(legend) for legend in self._legends])
            CHAR_WIDTH = 8
            with omni.ui.VGrid(
                row_height=isaacsim.gui.components.ui_utils.LABEL_HEIGHT,
                column_width=max_legend * CHAR_WIDTH + 6,
            ):
                for idx in range(len(self._y_data)):
                    with omni.ui.HStack():
                        model = omni.ui.SimpleBoolModel()
                        model.set_value(self._series_visible[idx])
                        omni.ui.CheckBox(model=model, tooltip="", width=4)
                        model.add_value_changed_fn(lambda val, idx=idx: self._change_plot_visibility(idx, val.as_bool))
                        omni.ui.Spacer(width=2)
                        with omni.ui.VStack():
                            omni.ui.Label(
                                self._legends[idx],
                                width=max_legend * CHAR_WIDTH,
                                alignment=omni.ui.Alignment.LEFT,
                                style={"color": self._colors[idx], "font_size": 12},
                            )
                            omni.ui.StringField(
                                model=self._plot_selected_values[idx],
                                width=max_legend * CHAR_WIDTH,
                                alignment=omni.ui.Alignment.LEFT,
                                style={"color": self._colors[idx], "font_size": 10},
                                read_only=True,
                            )

    def _build_limits_frame(self):
        """Build the frame containing the controls for the y-axis limits.

        This is an internal function to build the frame containing the controls for the y-axis limits. This function
        should only be called from within the build function of a frame.

        The built widget has the following layout:
        +-------------------------------------------------------+
        |                   limits_frame                        |
        ||+---------------------------------------------------+||
        |||                                                   |||
        |||         Limits    [min] [max] [Re-Sacle]          |||
        |||         Autoscale[x]                              |||
        |||    -------------------------------------------    |||
        |||+-------------------------------------------------+|||
        """
        with omni.ui.VStack():
            with omni.ui.HStack():
                omni.ui.Label(
                    "Limits",
                    width=isaacsim.gui.components.ui_utils.LABEL_WIDTH,
                    alignment=omni.ui.Alignment.LEFT_CENTER,
                )

                self.lower_limit_drag = omni.ui.FloatDrag(name="min", enabled=True, alignment=omni.ui.Alignment.CENTER)
                y_min_model = self.lower_limit_drag.model
                y_min_model.set_value(self._y_min)
                y_min_model.add_value_changed_fn(lambda x: self._set_y_min(x.as_float))
                omni.ui.Spacer(width=2)

                self.upper_limit_drag = omni.ui.FloatDrag(name="max", enabled=True, alignment=omni.ui.Alignment.CENTER)
                y_max_model = self.upper_limit_drag.model
                y_max_model.set_value(self._y_max)
                y_max_model.add_value_changed_fn(lambda x: self._set_y_max(x.as_float))
                omni.ui.Spacer(width=2)

                omni.ui.Button(
                    "Re-Scale",
                    width=isaacsim.gui.components.ui_utils.BUTTON_WIDTH,
                    clicked_fn=self._rescale_btn_pressed,
                    alignment=omni.ui.Alignment.LEFT_CENTER,
                    style=isaacsim.gui.components.ui_utils.get_style(),
                )

                omni.ui.CheckBox(model=self._autoscale_model, tooltip="", width=4)

            omni.ui.Line(
                style={"color": 0x338A8777},
                width=omni.ui.Fraction(1),
                alignment=omni.ui.Alignment.CENTER,
            )

    def _build_filter_frame(self):
        """Build the frame containing the filter controls.

        This is an internal function to build the frame containing the filter controls. This function
        should only be called from within the build function of a frame.

        The built widget has the following layout:
        +-------------------------------------------------------+
        |                   filter_frame                        |
        ||+---------------------------------------------------+||
        |||                                                   |||
        |||                                                   |||
        |||                                                   |||
        |||+-------------------------------------------------+|||
        |+-----------------------------------------------------+|
        +-------------------------------------------------------+
        """
        with omni.ui.VStack():
            with omni.ui.HStack():

                def _filter_changed(value):
                    self.clear()
                    self._filter_mode = value

                isaacsim.gui.components.ui_utils.dropdown_builder(
                    label="Filter",
                    type="dropdown",
                    items=["None", "Lowpass", "Integrate", "Derivative"],
                    tooltip="Select a filter",
                    on_clicked_fn=_filter_changed,
                )

                def _toggle_paused():
                    self._is_paused = not self._is_paused

                # Button
                omni.ui.Button(
                    "Play/Pause",
                    width=isaacsim.gui.components.ui_utils.BUTTON_WIDTH,
                    clicked_fn=_toggle_paused,
                    alignment=omni.ui.Alignment.LEFT_CENTER,
                    style=isaacsim.gui.components.ui_utils.get_style(),
                )

    def _create_ui_widget(self):
        """Create the full UI widget."""

        def _build_widget():
            self._is_built = False
            with omni.ui.VStack():
                self._main_plot_frame = omni.ui.Frame(build_fn=self._build_stacked_plots)
                omni.ui.Spacer(height=8)
                self._legends_frame = omni.ui.Frame(build_fn=self._build_legends_frame)
                omni.ui.Spacer(height=8)
                self._limits_frame = omni.ui.Frame(build_fn=self._build_limits_frame)
                omni.ui.Spacer(height=8)
                self._filter_frame = omni.ui.Frame(build_fn=self._build_filter_frame)
            self._is_built = True

        containing_frame = omni.ui.Frame(build_fn=_build_widget)

        return containing_frame

    """ UI Actions Listener Functions """

    def _change_plot_visibility(self, idx: int, visible: bool):
        """Change the visibility of a plot at position idx."""
        self._series_visible[idx] = visible
        self._plot_frames[idx].visible = visible
        # self._main_plot_frame.rebuild()

    def _set_y_min(self, val: float):
        """Update the y-axis minimum."""
        self._y_min = val
        self.lower_limit_drag.model.set_value(val)
        self._main_plot_frame.rebuild()

    def _set_y_max(self, val: float):
        """Update the y-axis maximum."""
        self._y_max = val
        self.upper_limit_drag.model.set_value(val)
        self._main_plot_frame.rebuild()

    def _rescale_btn_pressed(self):
        """Autoscale the y-axis to the current data."""
        if any(self._series_visible):
            y_min = np.round(
                min([min(y) for idx, y in enumerate(self._y_data) if self._series_visible[idx]]),
                4,
            )
            y_max = np.round(
                max([max(y) for idx, y in enumerate(self._y_data) if self._series_visible[idx]]),
                4,
            )
            if y_min == y_max:
                y_max += 1e-4  # Make sure axes don't collapse

            self._y_max = y_max
            self._y_min = y_min

        if hasattr(self, "lower_limit_drag") and hasattr(self, "upper_limit_drag"):
            self.lower_limit_drag.model.set_value(self._y_min)
            self.upper_limit_drag.model.set_value(self._y_max)

        self._main_plot_frame.rebuild()

    """ Helper Functions """

    def _get_distinct_hex_colors(self, num_colors) -> list[int]:
        """
        This function returns a list of distinct colors for plotting.

        Args:
            num_colors (int): the number of colors to generate

        Returns:
            List[int]: a list of distinct colors in hexadecimal format 0xFFBBGGRR
        """
        # Generate equally spaced colors in HSV space
        rgb_colors = [
            colorsys.hsv_to_rgb(hue / num_colors, 0.75, 1) for hue in np.linspace(0, num_colors - 1, num_colors)
        ]
        # Convert to 0-255 RGB values
        rgb_colors = [[int(c * 255) for c in rgb] for rgb in rgb_colors]
        # Convert to 0xFFBBGGRR format
        hex_colors = [0xFF * 16**6 + c[2] * 16**4 + c[1] * 16**2 + c[0] for c in rgb_colors]
        return hex_colors
