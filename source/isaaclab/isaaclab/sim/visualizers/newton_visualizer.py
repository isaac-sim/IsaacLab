# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import omni.log
import warp as wp
from newton.viewer import ViewerGL

from .newton_visualizer_cfg import NewtonVisualizerCfg
from .visualizer import Visualizer


class NewtonViewerGL(ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls.
    
    Adds two pause modes:
    - Training pause: Stops physics simulation, continues rendering
    - Rendering pause: Stops rendering updates, continues physics (SPACE key)
    """

    def __init__(self, *args, train_mode: bool = True, metadata: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._paused_rendering = False
        self._is_train_mode = train_mode
        self._metadata = metadata or {}
        self._fallback_draw_controls = False

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def is_training_paused(self) -> bool:
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        return self._paused_rendering
    
    def is_paused(self) -> bool:
        # duplicate to above for now
        return self._paused_rendering

    def _render_training_controls(self, imgui):
        imgui.separator()
        
        if self._is_train_mode:
            imgui.text("IsaacLab Training Controls")
            pause_label = "Resume Training" if self._paused_training else "Pause Training"
        else:
            imgui.text("IsaacLab Playback Controls")
            pause_label = "Resume Playing" if self._paused_training else "Pause Playing"
        
        if imgui.button(pause_label):
            self._paused_training = not self._paused_training

        if self._is_train_mode:
            rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
            if imgui.button(rendering_label):
                self._paused_rendering = not self._paused_rendering
                self._paused = self._paused_rendering

    def on_key_press(self, symbol, modifiers):
        if self.ui.is_capturing():
            return

        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return

        if symbol == pyglet.window.key.SPACE:
            self._paused_rendering = not self._paused_rendering
            self._paused = self._paused_rendering
            return

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
        import newton as nt

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
                    num_envs = self._metadata.get("num_envs", 0)
                    imgui.text(f"Environments: {num_envs}")
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[self.model.up_axis]}")
                    gravity = wp.to_torch(self.model.gravity)[0]
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


class NewtonVisualizer(Visualizer):
    """Newton OpenGL visualizer for Isaac Lab.
    
    Lightweight OpenGL-based visualization with training/rendering pause controls.
    """

    def __init__(self, cfg: NewtonVisualizerCfg):
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._model = None
        self._state = None

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize visualizer with scene data."""
        if self._is_initialized:
            return

        metadata = {}
        if scene_data is not None:
            self._model = scene_data.get("model")
            self._state = scene_data.get("state")
            metadata = scene_data.get("metadata", {})
        
        physics_backend = metadata.get("physics_backend", "unknown")
        if physics_backend != "newton" and physics_backend != "unknown":
            raise RuntimeError(
                f"Newton visualizer requires Newton physics backend, got '{physics_backend}'. "
                f"Use OVVisualizer for other backends."
            )
        
        if self._model is None:
            raise RuntimeError(
                "Newton visualizer requires Newton Model in scene_data['model']. "
                "Ensure Newton physics is initialized first."
            )

        # Create the viewer with train_mode and metadata
        self._viewer = NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            train_mode=self.cfg.train_mode,
            metadata=metadata,
        )

        # Set the model
        self._viewer.set_model(self._model)

        # Configure camera
        self._viewer.camera.pos = wp.vec3(*self.cfg.camera_position)
        self._viewer.up_axis = ["X", "Y", "Z"].index(self.cfg.up_axis)
        self._viewer.scaling = 1.0
        self._viewer._paused = False

        # Configure visualization options
        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_com = self.cfg.show_com

        # Configure rendering options
        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_sky = self.cfg.enable_sky
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe

        # Configure colors
        self._viewer.renderer.sky_upper = self.cfg.background_color
        self._viewer.renderer.sky_lower = self.cfg.ground_color
        self._viewer.renderer._light_color = self.cfg.light_color

        # Setup partial visualization (env_ids_to_viz filtering)
        if self.cfg.env_ids_to_viz is not None:
            num_envs = metadata.get("num_envs", 0)
            self._setup_env_filtering(num_envs)
            omni.log.info(
                f"[NewtonVisualizer] Partial visualization: {len(self.cfg.env_ids_to_viz)}/{num_envs} environments"
            )

        self._is_initialized = True

    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualizer for one step."""
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        if state is not None:
            self._state = state

        try:
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                if self._state is not None:
                    self._viewer.log_state(self._state)
                self._viewer.end_frame()
            else:
                self._viewer._update()
        except Exception:
            pass

    def close(self) -> None:
        """Close visualizer and clean up resources."""
        if self._is_closed:
            return
        if self._viewer is not None:
            self._viewer = None
        self._is_closed = True

    def is_running(self) -> bool:
        """Check if visualizer window is still open."""
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return False
        return self._viewer.is_running()
    
    def supports_markers(self) -> bool:
        """Newton visualizer does not support USD-based markers."""
        return False
    
    def supports_live_plots(self) -> bool:
        """Newton visualizer supports ImGui-based plots."""
        return True

    def is_training_paused(self) -> bool:
        """Check if training is paused."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()
    
    def _setup_env_filtering(self, num_envs: int) -> None:
        """Setup environment filtering using world offsets.
        
        WIP: Moves non-visualized environments far away (10000 units) to hide them.
        Future: Could use more sophisticated filtering or culling.
        
        Args:
            num_envs: Total number of environments.
        """
        import warp as wp
        
        # Create world offsets array (use viewer's device, not renderer's)
        offsets = wp.zeros(num_envs, dtype=wp.vec3, device=self._viewer.device)
        offsets_np = offsets.numpy()
        
        # Move non-visualized environments far away
        visualized_set = set(self.cfg.env_ids_to_viz)
        for world_idx in range(num_envs):
            if world_idx not in visualized_set:
                offsets_np[world_idx] = (10000.0, 10000.0, 10000.0)
        
        offsets.assign(offsets_np)
        self._viewer.world_offsets = offsets

