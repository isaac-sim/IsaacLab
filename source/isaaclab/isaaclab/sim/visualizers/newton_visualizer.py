# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp
from newton.viewer import ViewerGL

from .newton_visualizer_cfg import NewtonVisualizerCfg
from .visualizer import Visualizer

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


class NewtonViewerGL(ViewerGL):
    """Training-aware viewer that adds a separate pause for simulation/training.
    
    This class subclasses Newton's ViewerGL and introduces a second pause mode:
    - Rendering pause: identical to the base viewer's pause (space key / Pause checkbox)
    - Training pause: stops simulation/training steps but keeps rendering running
    
    The training pause can be toggled from the UI via a button and optionally via the 'T' key.
    """

    def __init__(self, *args, train_mode: bool = True, metadata: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused_training: bool = False
        self._paused_rendering: bool = False
        self._fallback_draw_controls: bool = False
        self._is_train_mode: bool = train_mode
        self._visualizer_update_frequency: int = 1
        self._metadata = metadata or {}

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def is_training_paused(self) -> bool:
        """Check if training is paused."""
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused."""
        return self._paused_rendering

    def set_visualizer_update_frequency(self, frequency: int) -> None:
        """Set the visualizer update frequency.
        
        Args:
            frequency: Number of simulation steps between visualizer updates.
        """
        self._visualizer_update_frequency = max(1, frequency)

    def get_visualizer_update_frequency(self) -> int:
        """Get the current visualizer update frequency.
        
        Returns:
            Number of simulation steps between visualizer updates.
        """
        return self._visualizer_update_frequency

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

        imgui.text("Visualizer Update Frequency")

        current_frequency = self._visualizer_update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            self._visualizer_update_frequency = new_frequency

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
    """Newton OpenGL Visualizer for Isaac Lab.
    
    This visualizer uses Newton's OpenGL-based viewer to provide lightweight,
    fast visualization of simulations. It includes IsaacLab-specific features
    like training controls, rendering pause, and update frequency adjustment.
    
    This class is registered with the visualizer registry as "newton" and can be
    instantiated via NewtonVisualizerCfg.create_visualizer().
    
    Args:
        cfg: Configuration for the Newton visualizer.
    """

    def __init__(self, cfg: NewtonVisualizerCfg):
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time: float = 0.0
        self._step_counter: int = 0
        self._model = None
        self._state = None

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize the Newton visualizer with the simulation scene.
        
        Args:
            scene_data: Optional dictionary containing initial scene data. Expected keys:
                       - "model": Newton Model object (required)
                       - "state": Newton State object (optional)
                       - "metadata": Scene metadata (contains physics_backend)
        
        Raises:
            RuntimeError: If Newton model is not available or if physics backend is incompatible.
        """
        if self._is_initialized:
            return

        # Extract model, state, and metadata from scene data
        metadata = {}
        if scene_data is not None:
            self._model = scene_data.get("model")
            self._state = scene_data.get("state")
            metadata = scene_data.get("metadata", {})
        
        # Validate physics backend compatibility
        physics_backend = metadata.get("physics_backend", "unknown")
        if physics_backend != "newton" and physics_backend != "unknown":
            raise RuntimeError(
                f"Newton visualizer requires Newton physics backend, but '{physics_backend}' is running. "
                f"Please use a compatible visualizer for {physics_backend} physics (e.g., OVVisualizer)."
            )
        
        # Validate required data
        if self._model is None:
            raise RuntimeError(
                "Newton visualizer requires a Newton Model in scene_data['model']. "
                "Make sure Newton physics is initialized before creating the visualizer."
            )

        # Create the viewer with metadata
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

        # Set update frequency
        self._viewer.set_visualizer_update_frequency(self.cfg.update_frequency)

        self._is_initialized = True

    def step(self, dt: float, scene_provider: SceneDataProvider | None = None) -> None:
        """Update the visualizer for one simulation step.
        
        Args:
            dt: Time step in seconds since last visualization update.
            scene_provider: Provider for accessing current scene data. The visualizer
                           will pull the latest Newton state from this provider.
        
        Note:
            Pause handling (training and rendering) is managed by SimulationContext.
            This method only performs the actual rendering when called.
            The visualizer MUST be called every frame to maintain proper ImGui state,
            even if we skip rendering some frames based on update_frequency.
        """
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        # Update simulation time
        self._sim_time += dt

        # Get the latest state from the scene provider
        if scene_provider is not None:
            self._state = scene_provider.get_state()

        # Render the current frame
        # Note: We always call begin_frame/end_frame to maintain ImGui state
        # The update frequency is handled internally by the viewer
        try:
            self._viewer.begin_frame(self._sim_time)
            try:
                if self._state is not None:
                    self._viewer.log_state(self._state)
            finally:
                # Always call end_frame if begin_frame succeeded
                self._viewer.end_frame()
        except Exception as e:
            # Handle any rendering errors gracefully
            # Frame lifecycle is now properly handled by try-finally
            pass  # Silently ignore to avoid log spam - the viewer will recover

    def close(self) -> None:
        """Close the visualizer and clean up resources."""
        if self._is_closed:
            return

        if self._viewer is not None:
            try:
                # Newton viewer doesn't have an explicit close method,
                # but we can clean up our reference
                self._viewer = None
            except Exception as e:
                print(f"[Warning] Error closing Newton visualizer: {e}")

        self._is_closed = True

    def is_running(self) -> bool:
        """Check if the visualizer is still running.
        
        Returns:
            True if the visualizer window is open and running.
        """
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return False

        return self._viewer.is_running()
    
    def supports_markers(self) -> bool:
        """Check if Newton visualizer supports visualization markers.
        
        Returns:
            False - Newton visualizer currently does not support VisualizationMarkers
            (they are USD-based and Newton uses its own rendering).
        """
        return False
    
    def supports_live_plots(self) -> bool:
        """Check if Newton visualizer supports live plots.
        
        Returns:
            True - Newton visualizer supports live plots via ImGui integration.
        """
        return True

    def is_training_paused(self) -> bool:
        """Check if training is paused by the visualizer.
        
        Returns:
            True if the user has paused training via the visualizer controls.
        """
        if not self._is_initialized or self._viewer is None:
            return False

        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused by the visualizer.
        
        Returns:
            True if rendering is paused via the visualizer controls.
        """
        if not self._is_initialized or self._viewer is None:
            return False

        return self._viewer.is_rendering_paused()

    def update_state(self, state) -> None:
        """Update the simulation state for visualization.
        
        This method should be called before step() to provide the latest simulation state.
        
        Args:
            state: The Newton State object to visualize.
        """
        self._state = state

