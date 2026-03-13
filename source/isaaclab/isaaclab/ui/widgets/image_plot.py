# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import cm

import omni

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils

from .ui_widget_wrapper import UIWidgetWrapper

if TYPE_CHECKING:
    import isaacsim.gui.components
    import omni.ui

# import logger
logger = logging.getLogger(__name__)


class ImagePlot(UIWidgetWrapper):
    """An image plot widget to display live data.

    It has the following Layout where the mode frame is only useful for depth images:
    +-------------------------------------------------------+
    |                  containing_frame                     |
    |+-----------------------------------------------------+|
    |                   main_plot_frame                     |
    ||+---------------------------------------------------+||
    |||                    plot_frames                    |||
    |||                                                   |||
    |||                                                   |||
    |||               (Image Plot Data)                   |||
    |||                                                   |||
    |||                                                   |||
    |||+-------------------------------------------------+|||
    |||                   mode_frame                      |||
    |||                                                   |||
    |||    [Dropdown: Mode Selection]                     |||
    |||    [Collapsible: Manual Normalization Options]    |||
    ||+---------------------------------------------------+||
    |+-----------------------------------------------------+|
    +-------------------------------------------------------+

    """

    def __init__(
        self,
        image: np.ndarray | None = None,
        label: str = "",
        widget_height: int = 200,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        """Create an XY plot UI Widget with axis scaling, legends, and support for multiple plots.

        Overlapping data is most accurately plotted when centered in the frame with reasonable axis scaling.
        Pressing down the mouse gives the x and y values of each function at an x coordinate.

        Args:
            image: Image to display
            label: Short descriptive text to the left of the plot
            widget_height: Height of the plot in pixels
            min_value: Minimum value for manual normalization/colorization. Defaults to 0.0.
            max_value: Maximum value for manual normalization/colorization. Defaults to 1.0.
        """

        self._curr_mode = "None"

        self._has_built = False

        self._enabled = True

        self._byte_provider = omni.ui.ByteImageProvider()
        if image is None:
            logger.warning("image is NONE")
            image = np.ones((480, 640, 3), dtype=np.uint8) * 255
            image[:, :, 0] = 0
            image[:, :240, 1] = 0

        # if image is channel first, convert to channel last
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = np.moveaxis(image, 0, -1)

        self._aspect_ratio = image.shape[1] / image.shape[0]
        self._widget_height = widget_height
        self._label = label
        self.update_image(image)

        plot_frame = self._create_ui_widget()

        super().__init__(plot_frame)

    def setEnabled(self, enabled: bool):
        self._enabled = enabled

    def update_image(self, image: np.ndarray):
        if not self._enabled:
            return

        # if image is channel first, convert to channel last
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = np.moveaxis(image, 0, -1)

        height, width = image.shape[:2]

        if self._curr_mode == "Normalization":
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
        elif self._curr_mode == "Colorization":
            if image.ndim == 3 and image.shape[2] == 3:
                logger.warning("Colorization mode is only available for single channel images")
            else:
                image = (image - image.min()) / (image.max() - image.min())
                colormap = cm.get_cmap("jet")
                if image.ndim == 3 and image.shape[2] == 1:
                    image = (colormap(image).squeeze(2) * 255).astype(np.uint8)
                else:
                    image = (colormap(image) * 255).astype(np.uint8)

        # convert image to 4-channel RGBA
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = np.dstack((image, image, image, np.full((height, width, 1), 255, dtype=np.uint8)))

        elif image.ndim == 3 and image.shape[2] == 3:
            image = np.dstack((image, np.full((height, width, 1), 255, dtype=np.uint8)))

        self._byte_provider.set_bytes_data(image.flatten().data, [width, height])

    def update_min_max(self, image: np.ndarray):
        if self._show_min_max and hasattr(self, "_min_max_label"):
            non_inf = image[np.isfinite(image)].flatten()
            if len(non_inf) > 0:
                self._min_max_label.text = self._get_unit_description(
                    np.min(non_inf), np.max(non_inf), np.median(non_inf)
                )
            else:
                self._min_max_label.text = self._get_unit_description(0, 0)

    def _create_ui_widget(self):
        containing_frame = omni.ui.Frame(build_fn=self._build_widget)
        return containing_frame

    def _get_unit_description(self, min_value: float, max_value: float, median_value: float = None):
        return (
            f"Min: {min_value * self._unit_scale:.2f} {self._unit_name} Max:"
            f" {max_value * self._unit_scale:.2f} {self._unit_name}"
            + (f" Median: {median_value * self._unit_scale:.2f} {self._unit_name}" if median_value is not None else "")
        )

    def _build_widget(self):
        with omni.ui.VStack(spacing=3):
            with omni.ui.HStack():
                # Write the leftmost label for what this plot is
                omni.ui.Label(
                    self._label,
                    width=isaacsim.gui.components.ui_utils.LABEL_WIDTH,
                    alignment=omni.ui.Alignment.LEFT_TOP,
                )
                with omni.ui.Frame(width=self._aspect_ratio * self._widget_height, height=self._widget_height):
                    self._base_plot = omni.ui.ImageWithProvider(self._byte_provider)

            if self._show_min_max:
                self._min_max_label = omni.ui.Label(self._get_unit_description(0, 0))

            omni.ui.Spacer(height=8)
            self._mode_frame = omni.ui.Frame(build_fn=self._build_mode_frame)

            omni.ui.Spacer(width=5)
        self._has_built = True

    def _build_mode_frame(self):
        """Build the frame containing the mode selection for the plots.

        This is an internal function to build the frame containing the mode selection for the plots. This function
        should only be called from within the build function of a frame.

        The built widget has the following layout:
        +-------------------------------------------------------+
        |                   mode_frame                          |
        ||+---------------------------------------------------+||
        |||    [Dropdown: Mode Selection]                     |||
        |||    [Collapsible: Manual Normalization Options]    |||
        |||+-------------------------------------------------+|||
        |+-----------------------------------------------------+|
        +-------------------------------------------------------+
        """
        with omni.ui.VStack(spacing=5, style=isaacsim.gui.components.ui_utils.get_style()):

            def _change_mode(value):
                self._curr_mode = value
                # Update visibility of collapsible frame
                show_options = value in ["Normalization", "Colorization"]
                if hasattr(self, "_options_collapsable"):
                    self._options_collapsable.visible = show_options
                    if show_options:
                        self._options_collapsable.title = f"{value} Options"

            # Mode dropdown
            isaacsim.gui.components.ui_utils.dropdown_builder(
                label="Mode",
                type="dropdown",
                items=["Original", "Normalization", "Colorization"],
                tooltip="Select a mode",
                on_clicked_fn=_change_mode,
            )

            # Collapsible frame for options (initially hidden)
            self._options_collapsable = omni.ui.CollapsableFrame(
                "Normalization Options",
                height=0,
                collapsed=False,
                visible=False,
                style=isaacsim.gui.components.ui_utils.get_style(),
                style_type_name_override="CollapsableFrame",
            )

            with self._options_collapsable:
                with omni.ui.VStack(spacing=5, style=isaacsim.gui.components.ui_utils.get_style()):

                    def _on_manual_changed(enabled):
                        self._enabled_min_max = enabled
                        # Enable/disable the float fields
                        if hasattr(self, "_min_model"):
                            self._min_field.enabled = enabled
                        if hasattr(self, "_max_model"):
                            self._max_field.enabled = enabled

                    def _on_min_changed(model):
                        self._min_value = model.as_float

                    def _on_max_changed(model):
                        self._max_value = model.as_float

                    # Manual values checkbox
                    isaacsim.gui.components.ui_utils.cb_builder(
                        label="Use Manual Values",
                        type="checkbox",
                        default_val=self._enabled_min_max,
                        tooltip="Enable manual min/max values",
                        on_clicked_fn=_on_manual_changed,
                    )

                    # Min value with reset button
                    with omni.ui.HStack():
                        omni.ui.Label(
                            "Min",
                            width=isaacsim.gui.components.ui_utils.LABEL_WIDTH,
                            alignment=omni.ui.Alignment.LEFT_CENTER,
                        )
                        self._min_field = omni.ui.FloatDrag(
                            name="FloatField",
                            width=omni.ui.Fraction(1),
                            height=0,
                            alignment=omni.ui.Alignment.LEFT_CENTER,
                            enabled=self._enabled_min_max,
                        )
                        self._min_model = self._min_field.model
                        self._min_model.set_value(self._min_value)
                        self._min_model.add_value_changed_fn(_on_min_changed)

                        omni.ui.Spacer(width=5)
                        omni.ui.Button(
                            "0",
                            width=20,
                            height=20,
                            clicked_fn=lambda: self._min_model.set_value(0.0),
                            tooltip="Reset to 0.0",
                            style=isaacsim.gui.components.ui_utils.get_style(),
                        )
                        isaacsim.gui.components.ui_utils.add_line_rect_flourish(False)

                    # Max value with reset button
                    with omni.ui.HStack():
                        omni.ui.Label(
                            "Max",
                            width=isaacsim.gui.components.ui_utils.LABEL_WIDTH,
                            alignment=omni.ui.Alignment.LEFT_CENTER,
                        )
                        self._max_field = omni.ui.FloatDrag(
                            name="FloatField",
                            width=omni.ui.Fraction(1),
                            height=0,
                            alignment=omni.ui.Alignment.LEFT_CENTER,
                            enabled=self._enabled_min_max,
                        )
                        self._max_model = self._max_field.model
                        self._max_model.set_value(self._max_value)
                        self._max_model.add_value_changed_fn(_on_max_changed)

                        omni.ui.Spacer(width=5)
                        omni.ui.Button(
                            "1",
                            width=20,
                            height=20,
                            clicked_fn=lambda: self._max_model.set_value(1.0),
                            tooltip="Reset to 1.0",
                            style=isaacsim.gui.components.ui_utils.get_style(),
                        )
                        isaacsim.gui.components.ui_utils.add_line_rect_flourish(False)
