# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import cm
from typing import TYPE_CHECKING, Optional

import carb
import omni
import omni.log

from .ui_widget_wrapper import UIWidgetWrapper

if TYPE_CHECKING:
    import isaacsim.gui.components
    import omni.ui


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
    |||    [x][Absolute] [x][Grayscaled] [ ][Colorized]   |||
    |+-----------------------------------------------------+|
    +-------------------------------------------------------+

    """

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        label: str = "",
        widget_height: int = 200,
        show_min_max: bool = True,
        unit: tuple[float, str] = (1, ""),
    ):
        """Create an XY plot UI Widget with axis scaling, legends, and support for multiple plots.

        Overlapping data is most accurately plotted when centered in the frame with reasonable axis scaling.
        Pressing down the mouse gives the x and y values of each function at an x coordinate.

        Args:
            image: Image to display
            label: Short descriptive text to the left of the plot
            widget_height: Height of the plot in pixels
            show_min_max: Whether to show the min and max values of the image
            unit: Tuple of (scale, name) for the unit of the image
        """
        self._show_min_max = show_min_max
        self._unit_scale = unit[0]
        self._unit_name = unit[1]

        self._curr_mode = "None"

        self._has_built = False

        self._enabled = True

        self._byte_provider = omni.ui.ByteImageProvider()
        if image is None:
            carb.log_warn("image is NONE")
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
                omni.log.warn("Colorization mode is only available for single channel images")
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
        |                   legends_frame                       |
        ||+---------------------------------------------------+||
        |||                                                   |||
        |||    [x][Series 1] [x][Series 2] [ ][Series 3]      |||
        |||                                                   |||
        |||+-------------------------------------------------+|||
        |+-----------------------------------------------------+|
        +-------------------------------------------------------+
        """
        with omni.ui.HStack():
            with omni.ui.HStack():

                def _change_mode(value):
                    self._curr_mode = value

                isaacsim.gui.components.ui_utils.dropdown_builder(
                    label="Mode",
                    type="dropdown",
                    items=["Original", "Normalization", "Colorization"],
                    tooltip="Select a mode",
                    on_clicked_fn=_change_mode,
                )
