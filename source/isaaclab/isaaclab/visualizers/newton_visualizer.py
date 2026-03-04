# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import warp as wp
from newton.viewer import ViewerGL

from .newton_visualizer_cfg import NewtonVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
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

        # Camera view switching state
        self._camera_views: OrderedDict[str, tuple[tuple, tuple]] = OrderedDict()
        self._selected_camera_idx: int = 0  # 0 = Free Camera
        self._locked_camera_pose: tuple[tuple, tuple] | None = None
        self._saved_free_camera: tuple[object, float, float] | None = None  # (pos, yaw, pitch)

        # Multi-tile camera view state
        self._multi_tile_active: bool = False
        self._tile_selections: dict[str, bool] = {}  # which cameras are checked (keyed by name)
        self._saved_free_camera_for_multi: tuple[object, float, float] | None = None

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def register_camera_view(self, name: str, pos: tuple, target: tuple) -> None:
        """Register a named camera viewpoint for the dropdown selector."""
        self._camera_views[name] = (pos, target)

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

    def _is_camera_locked(self) -> bool:
        if self._multi_tile_active:
            return True
        return self._selected_camera_idx > 0 and self._locked_camera_pose is not None

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

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._is_camera_locked():
            return
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self._is_camera_locked():
            return
        super().on_mouse_scroll(x, y, scroll_x, scroll_y)

    def _update_camera(self, dt: float):
        if self._is_camera_locked():
            return
        super()._update_camera(dt)

    def _render_fallback_controls(self):
        """Render fallback training controls window when side callback registration failed."""
        imgui = self.ui.imgui
        from contextlib import suppress

        with suppress(Exception):
            imgui.set_next_window_pos(imgui.ImVec2(320, 10))

        flags = 0
        if imgui.begin("Training Controls", flags=flags):
            self._render_training_controls(imgui)
        imgui.end()

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

                    show_collision = self.show_collision
                    changed, self.show_collision = imgui.checkbox("Show Collision", show_collision)

                    show_visual = self.show_visual
                    changed, self.show_visual = imgui.checkbox("Show Visual", show_visual)

                    show_inertia_boxes = self.show_inertia_boxes
                    changed, self.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", show_inertia_boxes)

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

                # Camera view selector dropdown
                if self._camera_views:
                    camera_names = ["Free Camera"] + list(self._camera_views.keys()) + ["Multi-Tile"]
                    multi_tile_idx = len(camera_names) - 1

                    # Determine the current combo index
                    if self._multi_tile_active:
                        current_combo_idx = multi_tile_idx
                    else:
                        current_combo_idx = self._selected_camera_idx

                    changed, new_idx = imgui.combo(
                        "##camera_select", current_combo_idx, camera_names
                    )
                    if changed:
                        if new_idx == multi_tile_idx:
                            # Entering multi-tile mode
                            if not self._multi_tile_active:
                                # Save free camera state
                                if self._selected_camera_idx == 0:
                                    self._saved_free_camera_for_multi = (
                                        self.camera.pos, self.camera.yaw, self.camera.pitch
                                    )
                                elif self._saved_free_camera is not None:
                                    self._saved_free_camera_for_multi = self._saved_free_camera
                                    self._saved_free_camera = None
                                # Default all cameras checked on first activation
                                if not self._tile_selections:
                                    self._tile_selections["Free Camera"] = True
                                    for name in self._camera_views:
                                        self._tile_selections[name] = True
                                self._multi_tile_active = True
                                self._locked_camera_pose = None
                        else:
                            # Leaving multi-tile mode (if was active)
                            was_multi = self._multi_tile_active
                            self._multi_tile_active = False

                            was_locked = self._selected_camera_idx > 0
                            self._selected_camera_idx = new_idx
                            if new_idx > 0:
                                # Save free camera state before locking
                                if was_multi and self._saved_free_camera_for_multi is not None:
                                    self._saved_free_camera = self._saved_free_camera_for_multi
                                    self._saved_free_camera_for_multi = None
                                elif not was_locked:
                                    self._saved_free_camera = (
                                        self.camera.pos, self.camera.yaw, self.camera.pitch
                                    )
                                name = camera_names[new_idx]
                                pose = self._camera_views[name]
                                self._locked_camera_pose = pose
                                cam_pos, cam_target = pose
                                self.camera.pos = wp.vec3(*cam_pos)
                                d = np.array(cam_target, dtype=np.float32) - np.array(
                                    cam_pos, dtype=np.float32
                                )
                                self.camera.yaw = float(np.degrees(np.arctan2(d[1], d[0])))
                                self.camera.pitch = float(
                                    np.degrees(np.arctan2(d[2], np.sqrt(d[0] ** 2 + d[1] ** 2)))
                                )
                            else:
                                self._locked_camera_pose = None
                                # Restore free camera state
                                saved = self._saved_free_camera_for_multi or self._saved_free_camera
                                if saved is not None:
                                    pos, yaw, pitch = saved
                                    self.camera.pos = pos
                                    self.camera.yaw = yaw
                                    self.camera.pitch = pitch
                                self._saved_free_camera = None
                                self._saved_free_camera_for_multi = None

                    # Show checkboxes when multi-tile is active
                    if self._multi_tile_active:
                        imgui.separator()
                        imgui.text("Select cameras:")
                        all_names = ["Free Camera"] + list(self._camera_views.keys())
                        for cam_name in all_names:
                            checked = self._tile_selections.get(cam_name, True)
                            changed_cb, new_val = imgui.checkbox(cam_name, checked)
                            if changed_cb:
                                self._tile_selections[cam_name] = new_val

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

    def _render_multi_tile(self):
        """Render multiple camera views tiled into the existing FBO."""
        from newton._src.viewer.camera import Camera
        from newton._src.viewer.gl.opengl import RendererGL

        gl = RendererGL.gl
        renderer = self.renderer

        # Gather selected cameras in order
        selected: list[str] = []
        all_names = ["Free Camera"] + list(self._camera_views.keys())
        for name in all_names:
            if self._tile_selections.get(name, True):
                selected.append(name)

        if not selected:
            # Nothing selected — fall back to normal render
            renderer.render(self.camera, self.objects, self.lines)
            return

        n = len(selected)
        sw = renderer._screen_width
        sh = renderer._screen_height

        # ---- 1. Shadow pass (once, using free camera position for light) ----
        gl.glViewport(0, 0, renderer._shadow_width, renderer._shadow_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, renderer._shadow_fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        if renderer.draw_shadows:
            # Use the free camera for shadow map center
            renderer.camera = self.camera
            renderer._view_matrix = self.camera.get_view_matrix()
            renderer._projection_matrix = self.camera.get_projection_matrix()
            renderer._render_shadow_map(self.objects)

        # ---- 2. Bind scene FBO and clear ----
        gl.glViewport(0, 0, sw, sh)
        target_fbo = renderer._frame_msaa_fbo if getattr(renderer, "msaa_samples", 0) > 0 else renderer._frame_fbo
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, target_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glClearColor(*renderer.sky_upper, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # ---- 3. Compute tile grid ----
        cols = min(n, 2)
        rows = math.ceil(n / cols)
        tile_w = sw // cols
        tile_h = sh // rows

        # ---- 4. Render each selected camera into its tile ----
        gl.glEnable(gl.GL_SCISSOR_TEST)

        for i, cam_name in enumerate(selected):
            col = i % cols
            row = rows - 1 - (i // cols)  # OpenGL y=0 is bottom
            x = col * tile_w
            y = row * tile_h

            gl.glViewport(x, y, tile_w, tile_h)
            gl.glScissor(x, y, tile_w, tile_h)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

            # Create a temporary camera for this tile
            tile_cam = Camera(
                fov=self.camera.fov,
                near=self.camera.near,
                far=self.camera.far,
                width=tile_w,
                height=tile_h,
                pos=(0.0, 0.0, 0.0),
                up_axis=self.camera.up_axis,
            )

            if cam_name == "Free Camera":
                # Use the frozen free camera state
                if self._saved_free_camera_for_multi is not None:
                    pos, yaw, pitch = self._saved_free_camera_for_multi
                    tile_cam.pos = pos
                    tile_cam.yaw = yaw
                    tile_cam.pitch = pitch
                else:
                    tile_cam.pos = self.camera.pos
                    tile_cam.yaw = self.camera.yaw
                    tile_cam.pitch = self.camera.pitch
            else:
                # Use registered camera view (pos, target)
                cam_pos, cam_target = self._camera_views[cam_name]
                tile_cam.pos = wp.vec3(*cam_pos)
                d = np.array(cam_target, dtype=np.float32) - np.array(cam_pos, dtype=np.float32)
                tile_cam.yaw = float(np.degrees(np.arctan2(d[1], d[0])))
                tile_cam.pitch = float(
                    np.degrees(np.arctan2(d[2], np.sqrt(d[0] ** 2 + d[1] ** 2)))
                )

            # Set renderer state for this tile
            renderer.camera = tile_cam
            renderer._view_matrix = tile_cam.get_view_matrix()
            renderer._projection_matrix = tile_cam.get_projection_matrix()

            renderer._render_scene(self.objects)
            if self.lines:
                renderer._render_lines(self.lines)

        gl.glDisable(gl.GL_SCISSOR_TEST)

        # ---- 5. MSAA resolve (full FBO blit) ----
        if getattr(renderer, "msaa_samples", 0) > 0 and renderer._frame_msaa_fbo is not None:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, renderer._frame_msaa_fbo)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, renderer._frame_fbo)
            gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glBlitFramebuffer(
                0, 0, sw, sh, 0, 0, sw, sh,
                gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT,
                gl.GL_NEAREST,
            )

        # ---- 6. Composite to screen (same as normal path) ----
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, sw, sh)

        if renderer._frame_fbo is not None:
            with renderer._frame_shader:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._frame_texture)
                renderer._frame_shader.update(0)
                gl.glBindVertexArray(renderer._frame_vao)
                gl.glDrawElements(gl.GL_TRIANGLES, len(renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
                gl.glBindVertexArray(0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Restore the main camera on the renderer
        renderer.camera = self.camera

        # ---- 7. Store tile layout for UI label overlay ----
        self._tile_layout = []
        for i, cam_name in enumerate(selected):
            col = i % cols
            row = i // cols
            # Screen coordinates (top-left origin for imgui)
            sx = col * tile_w
            sy = row * tile_h
            self._tile_layout.append((cam_name, sx, sy, tile_w, tile_h))

    def _render_ui(self):
        """Override to handle fallback controls and draw tile labels when multi-tile is active."""
        super()._render_ui()

        if self._fallback_draw_controls:
            self._render_fallback_controls()

        if not self._multi_tile_active or not hasattr(self, "_tile_layout"):
            return

        imgui = self.ui.imgui
        draw_list = imgui.get_foreground_draw_list()

        label_color = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.9))
        bg_color = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.5))
        padding = 4.0

        for cam_name, sx, sy, tw, th in self._tile_layout:
            # Draw background rect behind text
            text_size = imgui.calc_text_size(cam_name)
            x0 = float(sx) + padding
            y0 = float(sy) + padding
            draw_list.add_rect_filled(
                imgui.ImVec2(x0, y0),
                imgui.ImVec2(x0 + text_size.x + padding * 2, y0 + text_size.y + padding * 2),
                bg_color,
            )
            draw_list.add_text(imgui.ImVec2(x0 + padding, y0 + padding), label_color, cam_name)

    def _update(self):
        """Override to use multi-tile rendering when active."""
        self.renderer.update()

        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)

        self.wind.update(dt)

        if self.renderer.has_exit():
            return

        if self._multi_tile_active:
            # Set up GL state that render() normally sets
            from newton._src.viewer.gl.opengl import RendererGL

            gl = RendererGL.gl
            renderer = self.renderer
            renderer._make_current()
            gl.glClearColor(*renderer.sky_upper, 1)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(True)
            gl.glDepthRange(0.0, 1.0)

            # Lazy-load environment map
            if renderer._env_path is not None and renderer._env_texture is None:
                try:
                    renderer.set_environment_map(renderer._env_path)
                except Exception:
                    pass
                renderer._env_path = None

            self._render_multi_tile()
        else:
            self.renderer.render(self.camera, self.objects, self.lines)

        self._update_fps()

        if self.ui and self.ui.is_available and self.show_ui:
            self.ui.begin_frame()
            self._render_ui()
            self.ui.end_frame()
            self.ui.render()

        self.renderer.present()


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
        self._scene: InteractiveScene | None = None
        self._scene_camera_names: list[str] = []

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

    def register_scene_cameras(self, scene: InteractiveScene) -> None:
        """Discover TiledCamera sensors in the scene and register them as selectable viewpoints."""
        from isaaclab.sensors import TiledCamera

        self._scene = scene
        self._scene_camera_names = []
        if self._viewer is None:
            return
        for name, sensor in scene.sensors.items():
            if isinstance(sensor, TiledCamera):
                pos, target = self._extract_camera_pose(sensor)
                self._viewer.register_camera_view(name, pos, target)
                self._scene_camera_names.append(name)
                logger.info("[NewtonVisualizer] Registered scene camera '%s'", name)

    @staticmethod
    def _extract_camera_pose(sensor) -> tuple[tuple, tuple]:
        """Extract (position, target) from a TiledCamera sensor's current data."""
        import torch

        from isaaclab.utils.math import quat_apply

        pos_w = sensor.data.pos_w[0].cpu()  # (3,)
        quat_wxyz = sensor.data.quat_w_world[0].cpu()  # (w, x, y, z)
        # quat_apply expects (x, y, z, w)
        quat_xyzw = torch.tensor(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=torch.float32
        ).unsqueeze(0)
        # quat_w_world convention: forward = +X
        forward = quat_apply(quat_xyzw, torch.tensor([[1.0, 0.0, 0.0]]))[0]
        target = pos_w + forward
        return tuple(pos_w.tolist()), tuple(target.tolist())

    def _update_locked_camera(self) -> None:
        """If a sensor camera is selected, update the viewport to match its live pose.

        When multi-tile is active, update ALL registered sensor camera poses so
        every tile stays current with body-mounted cameras.
        """
        if self._viewer is None or self._scene is None:
            return

        if self._viewer._multi_tile_active:
            # Update all registered sensor cameras for multi-tile rendering
            for camera_name in self._scene_camera_names:
                sensor = self._scene.sensors.get(camera_name)
                if sensor is None:
                    continue
                pos, target = self._extract_camera_pose(sensor)
                self._viewer._camera_views[camera_name] = (pos, target)
            return

        idx = self._viewer._selected_camera_idx
        if idx <= 0 or idx > len(self._scene_camera_names):
            return
        camera_name = self._scene_camera_names[idx - 1]
        sensor = self._scene.sensors.get(camera_name)
        if sensor is None:
            return
        pos, target = self._extract_camera_pose(sensor)
        # Update the stored view and apply the pose
        self._viewer._camera_views[camera_name] = (pos, target)
        self._viewer._locked_camera_pose = (pos, target)
        self._apply_camera_pose((pos, target))

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1

        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()

        self._update_locked_camera()

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
