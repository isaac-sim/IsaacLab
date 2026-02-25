# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, Any

import warp as wp
from newton.viewer import ViewerGL

from .newton_visualizer_cfg import NewtonVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


class NewtonViewerGL(ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls."""

    def __init__(
        self,
        *args,
        metadata: dict | None = None,
        update_frequency: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._paused_rendering = False
        self._metadata = metadata or {}
        self._fallback_draw_controls = False
        self._update_frequency = update_frequency

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def is_training_paused(self) -> bool:
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        return self._paused_rendering

    def _render_training_controls(self, imgui):
        imgui.separator()
        imgui.text("IsaacLab Controls")

        pause_label = "Resume Training" if self._paused_training else "Pause Training"
        if imgui.button(pause_label):
            self._paused_training = not self._paused_training

        rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
        if imgui.button(rendering_label):
            self._paused_rendering = not self._paused_rendering
            self._paused = self._paused_rendering

        imgui.text("Visualizer Update Frequency")
        current_frequency = self._update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            self._update_frequency = new_frequency

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls visualizer update frequency\nlower values -> more responsive visualizer but slower"
                " training\nhigher values -> less responsive visualizer but faster training"
            )

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

        super()._render_ui()
        imgui = self.ui.imgui
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
        nav_highlight_color = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

        io = self.ui.io
        imgui.set_next_window_pos(imgui.ImVec2(10, 10))
        imgui.set_next_window_size(imgui.ImVec2(300, io.display_size[1] - 20))

        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()

            header_flags = 0

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("IsaacLab Options"):
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

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

                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization", flags=header_flags):
                    imgui.separator()

                    show_joints = self.show_joints
                    changed, self.show_joints = imgui.checkbox("Show Joints", show_joints)

                    show_contacts = self.show_contacts
                    changed, self.show_contacts = imgui.checkbox("Show Contacts", show_contacts)

                    show_springs = self.show_springs
                    changed, self.show_springs = imgui.checkbox("Show Springs", show_springs)

                    show_com = self.show_com
                    changed, self.show_com = imgui.checkbox("Show Center of Mass", show_com)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()

                changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)
                changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)
                changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

                changed, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)
                changed, self.renderer.sky_upper = imgui.color_edit3("Upper Sky Color", self.renderer.sky_upper)
                changed, self.renderer.sky_lower = imgui.color_edit3("Lower Sky Color", self.renderer.sky_lower)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Camera"):
                imgui.separator()

                pos = self.camera.pos
                pos_text = f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                imgui.text(pos_text)
                imgui.text(f"FOV: {self.camera.fov:.1f}°")
                imgui.text(f"Yaw: {self.camera.yaw:.1f}°")
                imgui.text(f"Pitch: {self.camera.pitch:.1f}°")

                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Forward/Left/Back/Right")
                imgui.text("QE - Down/Up")
                imgui.text("Left Click - Look around")
                imgui.text("Scroll - Zoom")
                imgui.text("Space - Pause/Resume Rendering")
                imgui.text("H - Toggle UI")
                imgui.text("ESC - Exit")

        imgui.end()
        return


class NewtonVisualizer(Visualizer):
    """Newton OpenGL visualizer for Isaac Lab."""

    def __init__(self, cfg: NewtonVisualizerCfg):
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._model = None
        self._state = None
        self._update_frequency = cfg.update_frequency
        self._scene_data_provider = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[NewtonVisualizer] initialize() called while already initialized.")
            return
        if scene_data_provider is None:
            raise RuntimeError("Newton visualizer requires a scene_data_provider.")

        self._scene_data_provider = scene_data_provider
        metadata = scene_data_provider.get_metadata()
        self._env_ids = self._compute_visualized_env_ids()
        if self._env_ids:
            get_filtered_model = getattr(scene_data_provider, "get_newton_model_for_env_ids", None)
            if callable(get_filtered_model):
                self._model = get_filtered_model(self._env_ids)
            else:
                self._model = scene_data_provider.get_newton_model()
        else:
            self._model = scene_data_provider.get_newton_model()
        self._state = scene_data_provider.get_newton_state(self._env_ids)

        self._viewer = NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

        self._viewer.set_model(self._model)
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        self._apply_camera_pose(self._resolve_initial_camera_pose())
        self._viewer.up_axis = 2  # Z-up

        self._viewer.scaling = 1.0
        self._viewer._paused = False

        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_com = self.cfg.show_com

        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_sky = self.cfg.enable_sky
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe

        self._viewer.renderer.sky_upper = self.cfg.sky_upper_color
        self._viewer.renderer.sky_lower = self.cfg.sky_lower_color
        self._viewer.renderer._light_color = self.cfg.light_color

        logger.info(
            "[NewtonVisualizer] initialized | camera_pos=%s camera_target=%s",
            self._viewer.camera.pos,
            self._last_camera_pose[1] if self._last_camera_pose else self.cfg.camera_target,
        )
        self._is_initialized = True

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1

        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()

        self._state = self._scene_data_provider.get_newton_state(self._env_ids)

        contacts = None
        if self._viewer.show_contacts:
            contacts_data = self._scene_data_provider.get_contacts()
            if isinstance(contacts_data, dict):
                contacts = contacts_data.get("contacts", contacts_data)
            else:
                contacts = contacts_data

        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        try:
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                if self._state is not None:
                    body_q = getattr(self._state, "body_q", None)
                    if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                        self._viewer.end_frame()
                        return
                    self._viewer.log_state(self._state)
                    if contacts is not None and hasattr(self._viewer, "log_contacts"):
                        try:
                            self._viewer.log_contacts(contacts, self._state)
                        except RuntimeError as exc:
                            logger.debug(f"[NewtonVisualizer] Failed to log contacts: {exc}")
                self._viewer.end_frame()
            else:
                self._viewer._update()
        except RuntimeError as exc:
            logger.debug("[NewtonVisualizer] Viewer update failed: %s", exc)

    def close(self) -> None:
        if self._is_closed:
            return
        if self._viewer is not None:
            self._viewer = None
        self._is_closed = True

    def is_running(self) -> bool:
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return False
        return self._viewer.is_running()

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.cfg.camera_source == "usd_path":
            pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
            if pose is not None:
                return pose
            logger.warning(
                "[NewtonVisualizer] camera_usd_path '%s' not found; using configured camera.",
                self.cfg.camera_usd_path,
            )
        return self.cfg.camera_position, self.cfg.camera_target

    def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        self._viewer.camera.pos = wp.vec3(*cam_pos)
        cam_pos_np = np.array(cam_pos, dtype=np.float32)
        cam_target_np = np.array(cam_target, dtype=np.float32)
        direction = cam_target_np - cam_pos_np
        yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        horizontal_dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        pitch = np.degrees(np.arctan2(direction[2], horizontal_dist))
        self._viewer.camera.yaw = float(yaw)
        self._viewer.camera.pitch = float(pitch)
        self._last_camera_pose = (cam_pos, cam_target)

    def _update_camera_from_usd_path(self) -> None:
        pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
        if pose is None:
            return
        if self._last_camera_pose == pose:
            return
        self._apply_camera_pose(pose)

    def supports_markers(self) -> bool:
        return False

    def supports_live_plots(self) -> bool:
        return False

    def is_training_paused(self) -> bool:
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()
