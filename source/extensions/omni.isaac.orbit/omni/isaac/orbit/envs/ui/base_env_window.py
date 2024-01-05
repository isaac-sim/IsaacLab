# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import numpy as np
import weakref
from typing import TYPE_CHECKING

import omni.isaac.ui.ui_utils as ui_utils
import omni.ui
from omni.kit.window.extensions import SimpleCheckBox

if TYPE_CHECKING:
    from ..base_env import BaseEnv


class BaseEnvWindow:
    """Window manager for the basic environment.

    This class creates a window that is used to control the environment. The window
    contains controls for rendering, debug visualization, and other environment-specific
    UI elements.

    Users can add their own UI elements to the window by using the `with` context manager.
    This can be done either be inheriting the class or by using the `env.window` object
    directly from the standalone execution script.

    Example for adding a UI element from the standalone execution script:
        >>> with env.window.ui_window_elements["main_vstack"]:
        >>>     ui.Label("My UI element")

    """

    def __init__(self, env: BaseEnv, window_name: str = "Orbit"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "Orbit".
        """
        # store inputs
        self.env = env
        # store the viewer related variables
        # -- locations of the camera
        self._viewer_eye = np.array(self.env.cfg.viewer.eye)
        self._viewer_lookat = np.array(self.env.cfg.viewer.lookat)
        # -- which environment to lookat
        self._viewer_env_index = 0
        self._viewer_origin = self.env.scene.env_origins[self._viewer_env_index].detach().cpu().numpy()
        # -- whether to follow an asset and which one
        self._viewer_follow_enabled = False
        self._viewer_assets_options = [
            *self.env.scene.rigid_objects.keys(),
            *self.env.scene.articulations.keys(),
        ]
        self._viewer_asset_name = self._viewer_assets_options[0]
        # create a handle to the camera callback
        self._viewer_follow_cam_handle = None

        print("Creating window for environment.")
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
            # create main stack
            self.ui_window_elements["main_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["main_vstack"]:
                # create collapsable frame for simulation
                self._build_sim_frame()
                # create collapsable frame for viewer
                self._build_viewer_frame()
                # create collapsable frame for debug visualization
                self._build_debug_vis_frame()

    def __del__(self):
        """Destructor for the window."""
        # destroy the window
        if self.ui_window is not None:
            self.ui_window.visible = False
            self.ui_window.destroy()
            self.ui_window = None

    """
    Build sub-sections of the UI.
    """

    def _build_sim_frame(self):
        """Builds the sim-related controls frame for the UI."""
        # create collapsable frame for controls
        self.ui_window_elements["sim_frame"] = omni.ui.CollapsableFrame(
            title="Simulation Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["sim_frame"]:
            # create stack for controls
            self.ui_window_elements["sim_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["sim_vstack"]:
                # create rendering mode dropdown
                render_mode_cfg = {
                    "label": "Rendering Mode",
                    "type": "dropdown",
                    "default_val": self.env.sim.render_mode.value,
                    "items": [member.name for member in self.env.sim.RenderMode if member.value >= 0],
                    "tooltip": "Select a rendering mode\n" + self.env.sim.RenderMode.__doc__,
                    "on_clicked_fn": lambda value: self.env.sim.set_render_mode(self.env.sim.RenderMode[value]),
                }
                self.ui_window_elements["render_dropdown"] = ui_utils.dropdown_builder(**render_mode_cfg)

    def _build_viewer_frame(self):
        """Build the viewer-related control frame for the UI."""
        # create collapsable frame for viewer
        self.ui_window_elements["viewer_frame"] = omni.ui.CollapsableFrame(
            title="Viewer Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["viewer_frame"]:
            # create stack for controls
            self.ui_window_elements["viewer_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["viewer_vstack"]:
                # create a number slider to move to environment origin
                viewport_origin_cfg = {
                    "label": "Environment Index",
                    "type": "button",
                    "default_val": self._viewer_env_index + 1,
                    "min": 1,
                    "max": self.env.num_envs,
                    "tooltip": "Move the viewport to the origin of the environment",
                }
                self.ui_window_elements["viewer_env_index"] = ui_utils.int_builder(**viewport_origin_cfg)
                # create a number slider to move to environment origin
                self.ui_window_elements["viewer_env_index"].add_value_changed_fn(self._set_viewer_env_index_fn)

                # create a tracker for the camera location
                viewer_follow_cfg = {
                    "label": "Follow Mode",
                    "type": "checkbox_dropdown",
                    "default_val": [False, 0],
                    "items": [name.replace("_", " ").title() for name in self._viewer_assets_options],
                    "tooltip": "Follow an asset in the scene.",
                    "on_clicked_fn": [self._toggle_viewer_follow_fn, self._set_viewer_follow_asset_fn],
                }
                self.ui_window_elements["viewer_follow"] = ui_utils.combo_cb_dropdown_builder(**viewer_follow_cfg)

                # add viewer default eye and lookat locations
                self.ui_window_elements["viewer_eye"] = ui_utils.xyz_builder(
                    label="Camera Eye",
                    tooltip="Modify the XYZ location of the viewer eye",
                    default_val=self.env.cfg.viewer.eye,
                    step=0.1,
                    on_value_changed_fn=[self._set_viewer_location_fn] * 3,
                )
                self.ui_window_elements["viewer_lookat"] = ui_utils.xyz_builder(
                    label="Camera Target",
                    tooltip="Modify the XYZ location of the viewer target",
                    default_val=self.env.cfg.viewer.lookat,
                    step=0.1,
                    on_value_changed_fn=[self._set_viewer_location_fn] * 3,
                )

    def _build_debug_vis_frame(self):
        """Builds the debug visualization frame for various scene elements.

        This function inquires the scene for all elements that have a debug visualization
        implemented and creates a checkbox to toggle the debug visualization for each element
        that has it implemented. If the element does not have a debug visualization implemented,
        a label is created instead.
        """
        # create collapsable frame for debug visualization
        self.ui_window_elements["debug_frame"] = omni.ui.CollapsableFrame(
            title="Scene Debug Visualization",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["debug_frame"]:
            # create stack for debug visualization
            self.ui_window_elements["debug_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["debug_vstack"]:
                elements = [
                    self.env.scene.terrain,
                    *self.env.scene.rigid_objects.values(),
                    *self.env.scene.articulations.values(),
                    *self.env.scene.sensors.values(),
                ]
                names = [
                    "terrain",
                    *self.env.scene.rigid_objects.keys(),
                    *self.env.scene.articulations.keys(),
                    *self.env.scene.sensors.keys(),
                ]
                # create one for the terrain
                for elem, name in zip(elements, names):
                    if elem is not None:
                        self._create_debug_vis_ui_element(name, elem)

    """
    Custom callbacks for UI elements.
    """

    def _toggle_viewer_follow_fn(self, value: bool):
        """Toggles the viewer follow mode."""
        # store the desired env index
        self._viewer_follow_enabled = value
        # register the camera callback for the follow mode
        if self._viewer_follow_enabled:
            # create a subscriber for the post update event if it doesn't exist
            if self._viewer_follow_cam_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._viewer_follow_cam_handle = (
                    app_interface.get_post_update_event_stream().create_subscription_to_pop(
                        lambda event, obj=weakref.proxy(self): obj._viewer_follow_asset_callback(event)
                    )
                )
        else:
            # remove the subscriber if it exists
            if self._viewer_follow_cam_handle is not None:
                self._viewer_follow_cam_handle.unsubscribe()
                self._viewer_follow_cam_handle = None
        # update the camera view
        self._update_camera_view()

    def _set_viewer_follow_asset_fn(self, value: str):
        """Sets the asset to follow."""
        # find which index the asset is
        fancy_names = [name.replace("_", " ").title() for name in self._viewer_assets_options]
        # store the desired env index
        self._viewer_asset_name = self._viewer_assets_options[fancy_names.index(value)]
        # update the camera view
        self._update_camera_view()

    def _set_viewer_location_fn(self, model: omni.ui.SimpleFloatModel):
        """Sets the viewer location based on the UI."""
        # obtain the camera locations
        for i in range(3):
            self._viewer_eye[i] = self.ui_window_elements["viewer_eye"][i].get_value_as_float()
            self._viewer_lookat[i] = self.ui_window_elements["viewer_lookat"][i].get_value_as_float()
        # update the camera view
        self._update_camera_view()

    def _set_viewer_env_index_fn(self, model: omni.ui.SimpleIntModel):
        """Moves the viewport to the origin of the environment."""
        # store the desired env index
        self._viewer_env_index = model.as_int - 1
        # obtain the origin of the environment
        if self._viewer_follow_enabled:
            self._viewer_origin = self.env.scene[self._viewer_asset_name].data.root_pos_w[self._viewer_env_index]
        else:
            self._viewer_origin = self.env.scene.env_origins[self._viewer_env_index]
        # origin
        self._viewer_origin = self._viewer_origin.detach().cpu().numpy()
        # update the camera view
        self._update_camera_view()

    def _viewer_follow_asset_callback(self, event):
        # update the camera origins
        self._viewer_origin = self.env.scene[self._viewer_asset_name].data.root_pos_w[self._viewer_env_index]
        # origin
        self._viewer_origin = self._viewer_origin.detach().cpu().numpy()
        # update the camera view
        self._update_camera_view()

    """
    Helper functions - UI updates.
    """

    def _update_camera_view(self, event=None):
        """Updates the camera view based on the current settings."""
        # set the camera locations
        cam_eye = self._viewer_origin + self._viewer_eye
        cam_target = self._viewer_origin + self._viewer_lookat
        # set the camera view
        self.env.sim.set_camera_view(eye=cam_eye, target=cam_target)

    """
    Helper functions - UI building.
    """

    def _create_debug_vis_ui_element(self, name: str, elem: object):
        """Create a checkbox for toggling debug visualization for the given element."""
        with omni.ui.HStack():
            # create the UI element
            text = (
                "Toggle debug visualization."
                if elem.has_debug_vis_implementation
                else "Debug visualization not implemented."
            )
            omni.ui.Label(
                name.replace("_", " ").title(),
                width=ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip=text,
            )
            self.ui_window_elements[f"{name}_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=elem.has_debug_vis_implementation,
                checked=elem.cfg.debug_vis,
                on_checked_fn=lambda value, e=weakref.proxy(elem): e.set_debug_vis(value),
            )
            ui_utils.add_line_rect_flourish()

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
