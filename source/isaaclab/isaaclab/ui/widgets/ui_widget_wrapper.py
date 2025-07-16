# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This file has been adapted from _isaac_sim/exts/isaacsim.gui.components/isaacsim/gui/components/element_wrappers/base_ui_element_wrappers.py

from __future__ import annotations

from typing import TYPE_CHECKING

import omni

if TYPE_CHECKING:
    import omni.ui


class UIWidgetWrapper:
    """
    Base class for creating wrappers around any subclass of omni.ui.Widget in order to provide an easy interface
    for creating and managing specific types of widgets such as state buttons or file pickers.
    """

    def __init__(self, container_frame: omni.ui.Frame):
        self._container_frame = container_frame

    @property
    def container_frame(self) -> omni.ui.Frame:
        return self._container_frame

    @property
    def enabled(self) -> bool:
        return self.container_frame.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self.container_frame.enabled = value

    @property
    def visible(self) -> bool:
        return self.container_frame.visible

    @visible.setter
    def visible(self, value: bool):
        self.container_frame.visible = value

    def cleanup(self):
        """
        Perform any necessary cleanup
        """
        pass
