# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import omni.kit.app

if TYPE_CHECKING:
    import omni.ui

    from ..manager_based_env import ManagerBasedEnv


class EmptyWindow:
    """
    Creates an empty UI window that can be docked in the Omniverse Kit environment.

    The class initializes a dockable UI window and provides a main frame with a vertical stack.
    You can add custom UI elements to this vertical stack.

    Example for adding a UI element from the standalone execution script:
        >>> with env.window.ui_window_elements["main_vstack"]:
        >>>     ui.Label("My UI element")

    """

    def __init__(self, env: ManagerBasedEnv, window_name: str):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window.
        """
        # store environment
        self.env = env

        # create window for UI
        self.ui_window = omni.ui.Window(
            window_name, width=400, height=500, visible=True, dock_preference=omni.ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        asyncio.ensure_future(self._dock_window(window_title=self.ui_window.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_window_elements = dict()
        # create main frame
        self.ui_window_elements["main_frame"] = self.ui_window.frame
        with self.ui_window_elements["main_frame"]:
            # create main vstack
            self.ui_window_elements["main_vstack"] = omni.ui.VStack(spacing=5, height=0)

    def __del__(self):
        """Destructor for the window."""
        # destroy the window
        if self.ui_window is not None:
            self.ui_window.visible = False
            self.ui_window.destroy()
            self.ui_window = None

    async def _dock_window(self, window_title: str):
        """Docks the custom UI window to the property window."""
        # wait for the window to be created
        for _ in range(5):
            if omni.ui.Workspace.get_window(window_title):
                break
            await self.env.sim.app.next_update_async()

        # dock next to properties window
        custom_window = omni.ui.Workspace.get_window(window_title)
        property_window = omni.ui.Workspace.get_window("Property")
        if custom_window and property_window:
            custom_window.dock_in(property_window, omni.ui.DockPosition.SAME, 1.0)
            custom_window.focus()
