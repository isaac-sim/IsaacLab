# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import omni.ui


class UiVisualizerBase:
    """Base Class for components that support debug visualizations that requires access to some UI elements.

    This class provides a set of functions that can be used to assign ui interfaces.

    The following functions are provided:

    * :func:`set_debug_vis`: Assigns a debug visualization interface. This function is called by the main UI
        when the checkbox for debug visualization is toggled.
    * :func:`set_vis_frame`: Assigns a small frame within the isaac lab tab that can be used to visualize debug
        information. Such as e.g. plots or images. It is called by the main UI on startup to create the frame.
    * :func:`set_window`: Assigns the main window that is used by the main UI. This allows the user
        to have full controller over all UI elements. But be warned, with great power comes great responsibility.
    """

    """
    Exposed Properties
    """

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        return self._has_implementation(self._set_debug_vis_impl)

    @property
    def has_vis_frame_implementation(self) -> bool:
        """Whether the component has a debug visualization frame implemented."""
        return self._has_implementation(self._set_vis_frame_impl)

    @property
    def has_window_implementation(self) -> bool:
        """Whether the component has a window implementation."""
        return self._has_implementation(self._set_window_impl)

    @property
    def has_env_selection_implementation(self) -> bool:
        """Whether the component has an environment selection implementation."""
        return self._has_implementation(self._set_env_selection_impl)

    """
    Exposed Setters
    """

    def set_env_selection(self, env_selection: int) -> bool:
        """Sets the selected environment id.

        This function is called by the main UI when the user selects a different environment.

        Args:
            env_selection: The currently selected environment id.

        Returns:
            Whether the environment selection was successfully set. False if the component
            does not support environment selection.
        """
        if not self.has_env_selection_implementation:
            return False
        self._set_env_selection_impl(env_selection)
        return True

    def set_window(self, window: omni.ui.Window) -> bool:
        """Sets the current main ui window.

        This function is called by the main UI when the window is created. It allows the component
        to add custom UI elements to the window or to control the window and its elements.

        Args:
            window: The ui window.

        Returns:
            Whether the window was successfully set. False if the component
            does not support this functionality.
        """
        if not self.has_window_implementation:
            return False
        self._set_window_impl(window)
        return True

    def set_vis_frame(self, vis_frame: omni.ui.Frame) -> bool:
        """Sets the debug visualization frame.

        This function is called by the main UI when the window is created. It allows the component
        to modify a small frame within the orbit tab that can be used to visualize debug information.

        Args:
            vis_frame: The debug visualization frame.

        Returns:
            Whether the debug visualization frame was successfully set. False if the component
            does not support debug visualization.
        """
        if not self.has_vis_frame_implementation:
            return False
        self._set_vis_frame_impl(vis_frame)
        return True

    """
    Internal Implementation
    """

    @staticmethod
    def _has_implementation(method) -> bool:
        """Whether ``method`` is overridden, i.e. its source does not raise :class:`NotImplementedError`."""
        return "NotImplementedError" not in inspect.getsource(method)

    def _set_env_selection_impl(self, env_idx: int):
        """Set the environment selection."""
        raise NotImplementedError(f"Environment selection is not implemented for {self.__class__.__name__}.")

    def _set_window_impl(self, window: omni.ui.Window):
        """Set the window."""
        raise NotImplementedError(f"Window is not implemented for {self.__class__.__name__}.")

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization state."""
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _set_vis_frame_impl(self, vis_frame: omni.ui.Frame):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")
