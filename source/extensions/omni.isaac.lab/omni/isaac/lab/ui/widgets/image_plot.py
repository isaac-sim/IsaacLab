# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import omni.ui as ui
from omni.isaac.ui.element_wrappers.base_ui_element_wrappers import UIWidgetWrapper
from omni.kit.window.property.templates import LABEL_WIDTH


class ImagePlot(UIWidgetWrapper):
    def __init__(self, image: np.ndarray = None, label: str = "", widget_height=200, show_min_max=True, unit=(1, "")):
        """Create an XY plot UI Widget with axis scaling, legends, and support for multiple plots.
        Overlapping data is most accurately plotted when centered in the frame with reasonable axis scaling.
        Pressing down the mouse gives the x and y values of each function at an x coordinate.
        Args:
            image (np.ndarray): Image to display
            label (str): Short descriptive text to the left of the plot
            widget_height (int): Height of the plot in pixels
            show_min_max (bool): Whether to show the min and max values of the image
            unit (tuple): Tuple of (scale, name) for the unit of the image
        """

        self._show_min_max = show_min_max
        self._unit_scale = unit[0]
        self._unit_name = unit[1]

        self._has_built = False

        self._enabled = False

        self._byte_provider = ui.ByteImageProvider()
        if image is None:
            image = np.ones((480, 640, 3), dtype=np.uint8) * 255
            image[:, :, 0] = 0
            image[:, :240, 1] = 0

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
        height, width = image.shape[:2]

        # convert image to 4-channel RGBA
        if image.ndim == 2:
            image = np.dstack((image, image, image, np.full((height, width, 1), 255, dtype=np.uint8)))
        elif image.ndim == 3:
            if image.shape[2] == 3:
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
        containing_frame = ui.Frame(build_fn=self._build_widget)
        return containing_frame

    def _get_unit_description(self, min_value: float, max_value: float, median_value: float = None):
        return (
            f"Min: {min_value * self._unit_scale:.2f} {self._unit_name} Max:"
            f" {max_value * self._unit_scale:.2f} {self._unit_name}"
            + (f" Median: {median_value * self._unit_scale:.2f} {self._unit_name}" if median_value is not None else "")
        )

    def _build_widget(self):

        with ui.VStack(spacing=3):
            with ui.HStack():
                # Write the leftmost label for what this plot is
                ui.Label(self._label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP)
                with ui.Frame(width=self._aspect_ratio * self._widget_height, height=self._widget_height):
                    self._base_plot = ui.ImageWithProvider(self._byte_provider)

            if self._show_min_max:
                self._min_max_label = ui.Label(self._get_unit_description(0, 0))
            ui.Spacer(width=5)
        self._has_built = True
