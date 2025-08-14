# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import os
import weakref
from datetime import datetime
from typing import TYPE_CHECKING

import isaacsim
import omni.kit.app
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.stage import get_current_stage
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.ui.widgets import ManagerLiveVisualizer

if TYPE_CHECKING:
    import omni.ui

    from ..manager_based_env import ManagerBasedEnv


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

    def __init__(self, env: ManagerBasedEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # store inputs
        self.env = env
        # prepare the list of assets that can be followed by the viewport camera
        # note that the first two options are "World" and "Env" which are special cases
        self._viewer_assets_options = [
            "World",
            "Env",
            *self.env.scene.rigid_objects.keys(),
            *self.env.scene.articulations.keys(),
        ]

        # get stage handle
        self.stage = get_current_stage()

        # Listeners for environment selection changes
        self._ui_listeners: list[ManagerLiveVisualizer] = []

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
                with self.ui_window_elements["debug_frame"]:
                    with self.ui_window_elements["debug_vstack"]:
                        self._visualize_manager(title="Actions", class_name="action_manager")
                        self._visualize_manager(title="Observations", class_name="observation_manager")

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
            style=isaacsim.gui.components.ui_utils.get_style(),
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
                self.ui_window_elements["render_dropdown"] = isaacsim.gui.components.ui_utils.dropdown_builder(
                    **render_mode_cfg
                )

                # create animation recording box
                record_animate_cfg = {
                    "label": "Record Animation",
                    "type": "state_button",
                    "a_text": "START",
                    "b_text": "STOP",
                    "tooltip": "Record the animation of the scene. Only effective if fabric is disabled.",
                    "on_clicked_fn": lambda value: self._toggle_recording_animation_fn(value),
                }
                self.ui_window_elements["record_animation"] = isaacsim.gui.components.ui_utils.state_btn_builder(
                    **record_animate_cfg
                )
                # disable the button if fabric is not enabled
                self.ui_window_elements["record_animation"].enabled = not self.env.sim.is_fabric_enabled()

    def _build_viewer_frame(self):
        """Build the viewer-related control frame for the UI."""
        # create collapsable frame for viewer
        self.ui_window_elements["viewer_frame"] = omni.ui.CollapsableFrame(
            title="Viewer Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=isaacsim.gui.components.ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["viewer_frame"]:
            # create stack for controls
            self.ui_window_elements["viewer_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["viewer_vstack"]:
                # create a number slider to move to environment origin
                # NOTE: slider is 1-indexed, whereas the env index is 0-indexed
                viewport_origin_cfg = {
                    "label": "Environment Index",
                    "type": "button",
                    "default_val": self.env.cfg.viewer.env_index + 1,
                    "min": 1,
                    "max": self.env.num_envs,
                    "tooltip": "The environment index to follow. Only effective if follow mode is not 'World'.",
                }
                self.ui_window_elements["viewer_env_index"] = isaacsim.gui.components.ui_utils.int_builder(
                    **viewport_origin_cfg
                )
                # create a number slider to move to environment origin
                self.ui_window_elements["viewer_env_index"].add_value_changed_fn(self._set_viewer_env_index_fn)

                # create a tracker for the camera location
                viewer_follow_cfg = {
                    "label": "Follow Mode",
                    "type": "dropdown",
                    "default_val": 0,
                    "items": [name.replace("_", " ").title() for name in self._viewer_assets_options],
                    "tooltip": "Select the viewport camera following mode.",
                    "on_clicked_fn": self._set_viewer_origin_type_fn,
                }
                self.ui_window_elements["viewer_follow"] = isaacsim.gui.components.ui_utils.dropdown_builder(
                    **viewer_follow_cfg
                )

                # add viewer default eye and lookat locations
                self.ui_window_elements["viewer_eye"] = isaacsim.gui.components.ui_utils.xyz_builder(
                    label="Camera Eye",
                    tooltip="Modify the XYZ location of the viewer eye.",
                    default_val=self.env.cfg.viewer.eye,
                    step=0.1,
                    on_value_changed_fn=[self._set_viewer_location_fn] * 3,
                )
                self.ui_window_elements["viewer_lookat"] = isaacsim.gui.components.ui_utils.xyz_builder(
                    label="Camera Target",
                    tooltip="Modify the XYZ location of the viewer target.",
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
            style=isaacsim.gui.components.ui_utils.get_style(),
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

    def _visualize_manager(self, title: str, class_name: str) -> None:
        """Checks if the attribute with the name 'class_name' can be visualized. If yes, create vis interface.

        Args:
            title: The title of the manager visualization frame.
            class_name: The name of the manager to visualize.
        """

        if hasattr(self.env, class_name) and class_name in self.env.manager_visualizers:
            manager = self.env.manager_visualizers[class_name]
            if hasattr(manager, "has_debug_vis_implementation"):
                self._create_debug_vis_ui_element(title, manager)
            else:
                print(
                    f"ManagerLiveVisualizer cannot be created for manager: {class_name}, has_debug_vis_implementation"
                    " does not exist"
                )
        else:
            print(f"ManagerLiveVisualizer cannot be created for manager: {class_name}, Manager does not exist")

    """
    Custom callbacks for UI elements.
    """

    def _toggle_recording_animation_fn(self, value: bool):
        """Toggles the animation recording."""
        if value:
            # log directory to save the recording
            if not hasattr(self, "animation_log_dir"):
                # create a new log directory
                log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.animation_log_dir = os.path.join(os.getcwd(), "recordings", log_dir)
            # start the recording
            _ = omni.kit.commands.execute(
                "StartRecording",
                target_paths=[("/World", True)],
                live_mode=True,
                use_frame_range=False,
                start_frame=0,
                end_frame=0,
                use_preroll=False,
                preroll_frame=0,
                record_to="FILE",
                fps=0,
                apply_root_anim=False,
                increment_name=True,
                record_folder=self.animation_log_dir,
                take_name="TimeSample",
            )
        else:
            # stop the recording
            _ = omni.kit.commands.execute("StopRecording")
            # save the current stage
            source_layer = self.stage.GetRootLayer()
            # output the stage to a file
            stage_usd_path = os.path.join(self.animation_log_dir, "Stage.usd")
            source_prim_path = "/"
            # creates empty anon layer
            temp_layer = Sdf.Find(stage_usd_path)
            if temp_layer is None:
                temp_layer = Sdf.Layer.CreateNew(stage_usd_path)
            temp_stage = Usd.Stage.Open(temp_layer)
            # update stage data
            UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.GetStageUpAxis(self.stage))
            UsdGeom.SetStageMetersPerUnit(temp_stage, UsdGeom.GetStageMetersPerUnit(self.stage))
            # copy the prim
            Sdf.CreatePrimInLayer(temp_layer, source_prim_path)
            Sdf.CopySpec(source_layer, source_prim_path, temp_layer, source_prim_path)
            # set the default prim
            temp_layer.defaultPrim = Sdf.Path(source_prim_path).name
            # remove all physics from the stage
            for prim in temp_stage.TraverseAll():
                # skip if the prim is an instance
                if prim.IsInstanceable():
                    continue
                # if prim has articulation then disable it
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                # if prim has rigid body then disable it
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                # if prim is a joint type then disable it
                if prim.IsA(UsdPhysics.Joint):
                    prim.GetAttribute("physics:jointEnabled").Set(False)
            # resolve all paths relative to layer path
            omni.usd.resolve_paths(source_layer.identifier, temp_layer.identifier)
            # save the stage
            temp_layer.Save()
            # print the path to the saved stage
            print("Recording completed.")
            print(f"\tSaved recorded stage to    : {stage_usd_path}")
            print(f"\tSaved recorded animation to: {os.path.join(self.animation_log_dir, 'TimeSample_tk001.usd')}")
            print("\nTo play the animation, check the instructions in the following link:")
            print(
                "\thttps://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html#using-the-captured-timesamples"
            )
            print("\n")
            # reset the log directory
            self.animation_log_dir = None

    def _set_viewer_origin_type_fn(self, value: str):
        """Sets the origin of the viewport's camera. This is based on the drop-down menu in the UI."""
        # Extract the viewport camera controller from environment
        vcc = self.env.viewport_camera_controller
        if vcc is None:
            raise ValueError("Viewport camera controller is not initialized! Please check the rendering mode.")

        # Based on origin type, update the camera view
        if value == "World":
            vcc.update_view_to_world()
        elif value == "Env":
            vcc.update_view_to_env()
        else:
            # find which index the asset is
            fancy_names = [name.replace("_", " ").title() for name in self._viewer_assets_options]
            # store the desired env index
            viewer_asset_name = self._viewer_assets_options[fancy_names.index(value)]
            # update the camera view
            vcc.update_view_to_asset_root(viewer_asset_name)

    def _set_viewer_location_fn(self, model: omni.ui.SimpleFloatModel):
        """Sets the viewport camera location based on the UI."""
        # access the viewport camera controller (for brevity)
        vcc = self.env.viewport_camera_controller
        if vcc is None:
            raise ValueError("Viewport camera controller is not initialized! Please check the rendering mode.")
        # obtain the camera locations and set them in the viewpoint camera controller
        eye = [self.ui_window_elements["viewer_eye"][i].get_value_as_float() for i in range(3)]
        lookat = [self.ui_window_elements["viewer_lookat"][i].get_value_as_float() for i in range(3)]
        # update the camera view
        vcc.update_view_location(eye, lookat)

    def _set_viewer_env_index_fn(self, model: omni.ui.SimpleIntModel):
        """Sets the environment index and updates the camera if in 'env' origin mode."""
        # access the viewport camera controller (for brevity)
        vcc = self.env.viewport_camera_controller
        if vcc is None:
            raise ValueError("Viewport camera controller is not initialized! Please check the rendering mode.")
        # store the desired env index, UI is 1-indexed
        vcc.set_view_env_index(model.as_int - 1)
        # notify additional listeners
        for listener in self._ui_listeners:
            listener.set_env_selection(model.as_int - 1)

    """
    Helper functions - UI building.
    """

    def _create_debug_vis_ui_element(self, name: str, elem: object):
        """Create a checkbox for toggling debug visualization for the given element."""
        from omni.kit.window.extensions import SimpleCheckBox

        with omni.ui.HStack():
            # create the UI element
            text = (
                "Toggle debug visualization."
                if elem.has_debug_vis_implementation
                else "Debug visualization not implemented."
            )
            omni.ui.Label(
                name.replace("_", " ").title(),
                width=isaacsim.gui.components.ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip=text,
            )
            has_cfg = hasattr(elem, "cfg") and elem.cfg is not None
            is_checked = False
            if has_cfg:
                is_checked = (hasattr(elem.cfg, "debug_vis") and elem.cfg.debug_vis) or (
                    hasattr(elem, "debug_vis") and elem.debug_vis
                )
            self.ui_window_elements[f"{name}_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=elem.has_debug_vis_implementation,
                checked=is_checked,
                on_checked_fn=lambda value, e=weakref.proxy(elem): e.set_debug_vis(value),
            )
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()

        # Create a panel for the debug visualization
        if isinstance(elem, ManagerLiveVisualizer):
            self.ui_window_elements[f"{name}_panel"] = omni.ui.Frame(width=omni.ui.Fraction(1))
            if not elem.set_vis_frame(self.ui_window_elements[f"{name}_panel"]):
                print(f"Frame failed to set for ManagerLiveVisualizer: {name}")

        # Add listener for environment selection changes
        if isinstance(elem, ManagerLiveVisualizer):
            self._ui_listeners.append(elem)

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
