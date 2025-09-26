# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training-aware viewer that adds a separate pause for simulation/training while keeping rendering at full rate.

This class subclasses Newton's ViewerGL and introduces a second pause mode:
- Rendering pause: identical to the base viewer's pause (space key / Pause checkbox)
- Training pause: stops simulation/training steps but keeps rendering running

The training pause can be toggled from the UI via a button and optionally via the 'T' key.
"""

from __future__ import annotations

import newton as nt
from newton.viewer import ViewerGL


class NewtonViewerGL(ViewerGL):
    def __init__(self, *args, train_mode: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused_training: bool = False
        self._paused_rendering: bool = False
        self._fallback_draw_controls: bool = False
        self._is_train_mode: bool = train_mode  # Convert train_mode to play_mode

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def is_training_paused(self) -> bool:
        return self._paused_training

    def is_paused(self) -> bool:
        return self._paused_rendering

    # UI callback rendered inside the "Example Options" panel of the left sidebar
    def _render_training_controls(self, imgui):
        imgui.separator()

        # Use simple flag to adjust labels
        if self._is_train_mode:
            imgui.text("IsaacLab Training Controls")
            pause_label = "Resume Training" if self._paused_training else "Pause Training"
        else:
            imgui.text("IsaacLab Playback Controls")
            pause_label = "Resume Playing" if self._paused_training else "Pause Playing"

        if imgui.button(pause_label):
            self._paused_training = not self._paused_training

        # Only show rendering controls when in training mode
        if self._is_train_mode:
            rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
            if imgui.button(rendering_label):
                self._paused_rendering = not self._paused_rendering

        # Import NewtonManager locally to avoid circular imports
        from .newton_manager import NewtonManager  # noqa: PLC0415

        imgui.text("Visualizer Update Frequency")

        current_frequency = NewtonManager._visualizer_update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            NewtonManager._visualizer_update_frequency = new_frequency

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls visualizer update frequency\nlower values-> more responsive visualizer but slower"
                " training\nhigher values-> less responsive visualizer but faster training"
            )

    # Override only SPACE key to use rendering pause, preserve all other shortcuts
    def on_key_press(self, symbol, modifiers):
        if self.ui.is_capturing():
            return

        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return

        if symbol == pyglet.window.key.SPACE:
            # Override SPACE to pause rendering instead of base pause
            self._paused_rendering = not self._paused_rendering
            return

        # For all other keys, call base implementation to preserve functionality
        super().on_key_press(symbol, modifiers)

    def _render_ui(self):
        if not self._fallback_draw_controls:
            return super()._render_ui()

        # Render base UI first
        super()._render_ui()

        # Then render a small floating window with training controls
        imgui = self.ui.imgui
        # Place near left panel but offset
        from contextlib import suppress

        with suppress(Exception):
            imgui.set_next_window_pos(imgui.ImVec2(320, 10))

        flags = 0
        if imgui.begin("Training Controls", flags=flags):
            self._render_training_controls(imgui)
        imgui.end()
        return None

    def _render_left_panel(self):
        """Override the left panel to remove the base pause checkbox."""
        imgui = self.ui.imgui

        # Use theme colors directly
        nav_highlight_color = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

        # Position the window on the left side
        io = self.ui.io
        imgui.set_next_window_pos(imgui.ImVec2(10, 10))
        imgui.set_next_window_size(imgui.ImVec2(300, io.display_size[1] - 20))

        # Main control panel window - use safe flag values
        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()

            header_flags = 0

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("IsaacLab Options"):
                # Render UI callbacks for side panel
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            # Model Information section
            if self.model is not None:
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information", flags=header_flags):
                    imgui.separator()
                    imgui.text(f"Environments: {self.model.num_envs}")
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[self.model.up_axis]}")
                    gravity = self.model.gravity
                    gravity_text = f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})"
                    imgui.text(gravity_text)

                # Visualization Controls section
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization", flags=header_flags):
                    imgui.separator()

                    # Joint visualization
                    show_joints = self.show_joints
                    changed, self.show_joints = imgui.checkbox("Show Joints", show_joints)

                    # Contact visualization
                    show_contacts = self.show_contacts
                    changed, self.show_contacts = imgui.checkbox("Show Contacts", show_contacts)

                    # Spring visualization
                    show_springs = self.show_springs
                    changed, self.show_springs = imgui.checkbox("Show Springs", show_springs)

                    # Center of mass visualization
                    show_com = self.show_com
                    changed, self.show_com = imgui.checkbox("Show Center of Mass", show_com)

            # Rendering Options section
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()

                # Sky rendering
                changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)

                # Shadow rendering
                changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)

                # Wireframe mode
                changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

                # Light color
                changed, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)
                # Sky color
                changed, self.renderer.sky_upper = imgui.color_edit3("Sky Color", self.renderer.sky_upper)
                # Ground color
                changed, self.renderer.sky_lower = imgui.color_edit3("Ground Color", self.renderer.sky_lower)

            # Camera Information section
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Camera"):
                imgui.separator()

                pos = self.camera.pos
                pos_text = f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                imgui.text(pos_text)
                imgui.text(f"FOV: {self.camera.fov:.1f}°")
                imgui.text(f"Yaw: {self.camera.yaw:.1f}°")
                imgui.text(f"Pitch: {self.camera.pitch:.1f}°")

                # Camera controls hint - update to reflect new controls
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("Left Click - Look around")
                imgui.text("Right Click - Pick objects")
                imgui.text("Scroll - Zoom")
                imgui.text("Space - Pause/Resume Rendering")
                imgui.text("H - Toggle UI")
                imgui.text("ESC/Q - Exit")

            # NOTE: Removed selection API section for now. In the future, we can add single env control through this section.
            # Selection API section
            # self._render_selection_panel()

        imgui.end()
        return
