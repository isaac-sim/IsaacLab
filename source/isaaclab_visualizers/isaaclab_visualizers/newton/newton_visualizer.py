# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.viewer import ViewerGL
from pyglet.math import Vec3 as PygletVec3

from isaaclab.envs.utils.camera_view import (
    VISUALIZER_TILED_CAMERA_MAX_TILES,
    apply_camera_target_positions,
    camera_rgb_batch,
    compute_tile_resolution,
    create_visualizer_camera,
    find_camera_by_prim_path,
    prim_world_positions,
    remove_generated_prims,
    resolve_tiled_env_indices,
)
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import apply_viewer_visible_worlds, resolve_visible_env_indices

from .newton_visualizer_cfg import NewtonVisualizerCfg

logger = logging.getLogger(__name__)

CONTACT_ARROW_PATH = "/contacts"
"""Viewer path used for native and synthesized contact arrows."""

CONTACT_ARROW_COLOR = (0.0, 1.0, 0.0)
"""Color used by Newton's native contact visualization."""

CONTACT_ARROW_LENGTH = 0.1
"""Length of synthesized contact arrows in meters."""

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider


class NewtonViewerGL(ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls."""

    def __init__(
        self,
        *args,
        metadata: dict | None = None,
        update_frequency: int = 1,
        **kwargs,
    ):
        """Initialize Newton viewer wrapper state.

        Args:
            *args: Positional arguments forwarded to ``ViewerGL``.
            metadata: Optional metadata shown in viewer panels.
            update_frequency: Viewer refresh cadence in simulation frames.
            **kwargs: Keyword arguments forwarded to ``ViewerGL``.
        """
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._paused_rendering = False
        self._metadata = metadata or {}
        self._fallback_draw_controls = False
        self._update_frequency = update_frequency
        self._color_edit3_prefers_sequence: bool | None = None
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_count = 0
        self._particle_color_buffer_value: tuple[float, float, float] | None = None

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
        except AttributeError:
            self._fallback_draw_controls = True

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused by viewer controls."""
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls."""
        return self._paused_rendering

    def _render_training_controls(self, imgui):
        """Render Isaac Lab-specific control widgets in the Newton viewer UI."""
        imgui.separator()
        imgui.text("IsaacLab Controls")

        pause_label = "Resume Simulation" if self._paused_training else "Pause Simulation"
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
        """Forward key presses unless UI is currently capturing input."""
        if self.ui.is_capturing():
            return
        super().on_key_press(symbol, modifiers)

    def _render_ui(self):
        """Render default UI and fallback control window when callback hooks are unavailable."""
        if not self._fallback_draw_controls:
            return super()._render_ui()

        super()._render_ui()
        imgui = self.ui.imgui
        from contextlib import suppress

        with suppress(Exception):
            imgui.set_next_window_pos(imgui.ImVec2(320, 10))

        flags = 0
        if imgui.begin("Simulation Controls", flags=flags):
            self._render_training_controls(imgui)
        imgui.end()
        return None

    def _coerce_color3(self, color) -> tuple[float, float, float]:
        """Normalize color values from imgui/renderer into an RGB tuple."""
        if hasattr(color, "x") and hasattr(color, "y") and hasattr(color, "z"):
            return (float(color.x), float(color.y), float(color.z))
        return (float(color[0]), float(color[1]), float(color[2]))

    def _particle_color_array(self, count: int) -> wp.array:
        """Return a cached Warp color array for Newton's particle point batch."""
        color = self._coerce_color3(self.particle_color)
        if (
            self._particle_color_buffer is None
            or self._particle_color_buffer_count != count
            or self._particle_color_buffer_value != color
        ):
            self._particle_color_buffer = wp.full(
                shape=count,
                value=wp.vec3(*color),
                dtype=wp.vec3,
                device=self.device,
            )
            self._particle_color_buffer_count = count
            self._particle_color_buffer_value = color
        return self._particle_color_buffer

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """Apply configured model-particle appearance while preserving Newton's point logging.

        The configured particle color only applies to Newton's canonical
        ``/model/particles`` point batch. User-defined point clouds retain the
        colors provided by their own ``log_points`` calls.
        """
        if name != "/model/particles" or points is None or self.particle_color is None:
            return super().log_points(name, points, radii, colors, hidden)

        colors = self._particle_color_array(len(points))
        return super().log_points(name, points, radii, colors, hidden)

    def _color_edit3_compat(self, imgui, label: str, color):
        """
        # Handle imgui.color_edit3 API differences between bindings.
        # Some require vector-like objects, others require a Sequence[float].
        # This method tries both approaches, caching the one that works to avoid repeated exceptions.
        # NOTE: This is a compatibility workaround, perhaps we can address the issue more directly.
        """
        color_tuple = self._coerce_color3(color)
        sequence_color = [color_tuple[0], color_tuple[1], color_tuple[2]]
        if self._color_edit3_prefers_sequence is not True:
            try:
                imvec4 = imgui.ImVec4(sequence_color[0], sequence_color[1], sequence_color[2], 1.0)
                changed, edited = imgui.color_edit3(label, imvec4)
                self._color_edit3_prefers_sequence = False
                return changed, self._coerce_color3(edited)
            except Exception:
                self._color_edit3_prefers_sequence = True

        try:
            changed, edited = imgui.color_edit3(label, sequence_color)
            return changed, self._coerce_color3(edited)
        except Exception as exc:
            logger.debug("[NewtonVisualizer] color_edit3 failed for '%s': %s", label, exc)
            return False, color_tuple

    def _render_left_panel(self):
        """Override the left panel to remove the base pause checkbox."""
        import newton as nt

        imgui = self.ui.imgui

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

                    show_collision = self.show_collision
                    changed, self.show_collision = imgui.checkbox("Show Collision", show_collision)

                    show_springs = self.show_springs
                    changed, self.show_springs = imgui.checkbox("Show Springs", show_springs)

                    show_inertia_boxes = self.show_inertia_boxes
                    changed, self.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", show_inertia_boxes)

                    show_com = self.show_com
                    changed, self.show_com = imgui.checkbox("Show Center of Mass", show_com)

                    show_particles = self.show_particles
                    changed, self.show_particles = imgui.checkbox("Show Particles", show_particles)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()

                changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)
                changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)
                changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

                try:
                    changed, self.renderer._light_color = self._color_edit3_compat(
                        imgui, "Light Color", self.renderer._light_color
                    )
                    changed, self.renderer.sky_upper = self._color_edit3_compat(
                        imgui, "Upper Sky Color", self.renderer.sky_upper
                    )
                    changed, self.renderer.sky_lower = self._color_edit3_compat(
                        imgui, "Lower Sky Color", self.renderer.sky_lower
                    )
                except Exception as exc:
                    logger.debug("[NewtonVisualizer] Rendering color controls failed: %s", exc)

            # Newton's ImageLogger owns camera-output image windows. Since Isaac Lab overrides
            # ViewerGL's left panel, explicitly keep the logged-image selector and draw path.
            if self._image_logger is not None:
                self._draw_tiled_camera_view_controls()

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
                imgui.text("WASD - Forward/Left/Back/Right")
                imgui.text("QE - Down/Up")
                imgui.text("Left Click - Look around")
                imgui.text("Scroll - Zoom")
                imgui.text("H - Toggle UI")
                imgui.text("ESC - Exit")

        imgui.end()
        if self._image_logger is not None:
            self._prime_image_logger_window_layout()
            self._image_logger.draw()
        return

    def _draw_tiled_camera_view_controls(self) -> None:
        """Render Newton ImageLogger controls with Isaac Lab-specific naming."""
        image_logger = self._image_logger
        if image_logger is None or not image_logger._images:
            return

        imgui = self.ui.imgui
        if not imgui.collapsing_header("Tiled Camera View", imgui.TreeNodeFlags_.default_open.value):
            return

        names = list(image_logger._images.keys())
        items = ["Hide", *names]
        if image_logger._selected is not None and image_logger._selected in names:
            current = names.index(image_logger._selected) + 1
        else:
            current = 0

        changed, new_idx = imgui.combo("##tiled_camera_view", current, items)
        if changed:
            image_logger._selected = None if new_idx == 0 else names[new_idx - 1]

    def _prime_image_logger_window_layout(self) -> None:
        """Make first-open image windows use the available viewer space.

        Newton's ImageLogger initializes windows from a fixed per-tile size, so
        small tiled batches otherwise open as small windows. Prime the next
        ImGui window size once, then let users move/resize it normally.
        """
        image_logger = self._image_logger
        if image_logger is None:
            return
        selected = image_logger._selected
        if selected is None:
            return
        entry = image_logger._images.get(selected)
        if entry is None or entry.window_initialized:
            return

        imgui = self.ui.imgui
        viewport = imgui.get_main_viewport()
        sidebar_width = float(image_logger._sidebar_width_px)
        margin = 20.0
        available_w = max(320.0, viewport.work_size.x - sidebar_width - 2.0 * margin)
        available_h = max(240.0, viewport.work_size.y - 2.0 * margin)

        n_tiles = max(1, int(entry.n))
        tile_aspect = float(entry.tile_aspect)
        cols = max(1, math.ceil(math.sqrt(n_tiles)))
        rows = math.ceil(n_tiles / cols)
        grid_aspect = (rows * tile_aspect) / cols
        title_and_padding_h = 40.0

        window_w = available_w
        window_h = min(available_h, max(240.0, window_w * grid_aspect + title_and_padding_h))
        pos_x = sidebar_width + margin
        pos_y = margin

        imgui.set_next_window_pos(imgui.ImVec2(float(pos_x), float(pos_y)), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(float(window_w), float(window_h)), imgui.Cond_.once)
        entry.window_initialized = True


class NewtonVisualizer(BaseVisualizer):
    """Newton OpenGL visualizer for Isaac Lab."""

    def __init__(self, cfg: NewtonVisualizerCfg):
        """Initialize Newton visualizer state.

        Args:
            cfg: Newton visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._model = None
        self._state = None
        self._update_frequency = cfg.update_frequency
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._headless_no_viewer = False
        self._resolved_visible_env_ids: list[int] | None = None
        self._camera_sensor = None
        self._camera_sensor_indices: list[int] = []
        self._camera_env_indices: list[int] = []
        self._camera_is_owned = False
        self._generated_camera_prim_paths: list[str] = []

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
        """
        from isaaclab_newton.physics import NewtonManager

        if self._is_initialized:
            logger.debug("[NewtonVisualizer] initialize() called while already initialized.")
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        num_envs = scene_data_provider.num_envs
        metadata = {"num_envs": num_envs}
        self._env_ids = self._compute_visualized_env_ids()
        self._model = NewtonManager.get_model()
        self._state = NewtonManager.get_state(self._scene_data_provider)

        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )

        # Use pyglet's EGL headless backend when requested or when no Linux X display is available.
        # This must run before the first ``pyglet.window`` import so ``Window`` resolves to
        # :class:`~pyglet.window.headless.HeadlessWindow`.
        if runtime_headless:
            import pyglet

            pyglet.options["headless"] = True

        self._viewer = NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

        if self._viewer is not None:
            self._viewer.set_model(self._model)
            apply_viewer_visible_worlds(
                self._viewer,
                env_ids=self._env_ids,
                max_visible_envs=self.cfg.max_visible_envs,
                num_envs=num_envs,
            )
            self._viewer.set_world_offsets((0.0, 0.0, 0.0))
            self._apply_camera_focal_length()
            initial_pose = self._resolve_initial_camera_pose()
            self._apply_camera_pose(initial_pose)
            self._viewer.up_axis = 2  # Z-up

            self._viewer.scaling = 1.0
            self._viewer._paused = False

            self._viewer.show_joints = self.cfg.show_joints
            self._viewer.show_contacts = self.cfg.show_contacts
            self._viewer.show_collision = self.cfg.show_collision
            self._viewer.show_springs = self.cfg.show_springs
            self._viewer.show_inertia_boxes = self.cfg.show_inertia_boxes
            self._viewer.show_com = self.cfg.show_com
            self._viewer.show_particles = self.cfg.show_particles
            self._viewer.particle_color = self.cfg.particle_color

            self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
            self._viewer.renderer.draw_sky = self.cfg.enable_sky
            self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe

            # Accept list/tuple/array-like config colors and provide a stable tuple for nanobind conversion.
            self._viewer.renderer.sky_upper = self._viewer._coerce_color3(self.cfg.sky_upper_color)
            self._viewer.renderer.sky_lower = self._viewer._coerce_color3(self.cfg.sky_lower_color)
            self._viewer.renderer._light_color = self._viewer._coerce_color3(self.cfg.light_color)

        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self._setup_camera_sensor_view(num_envs)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        self._log_initialization_table(
            logger=logger,
            title="NewtonVisualizer Configuration",
            rows=[
                (
                    "eye",
                    tuple(float(x) for x in self._viewer.camera.pos) if self._viewer is not None else self.cfg.eye,
                ),
                ("lookat", self._last_camera_pose[1] if self._last_camera_pose else self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("tiled_cam_view", self.cfg.tiled_cam_view),
                ("tiled_cam_num", self.cfg.tiled_cam_num),
                ("num_visualized_envs", num_visualized_envs),
                ("headless", self.cfg.headless),
                ("show_particles", self.cfg.show_particles),
                ("particle_color", self.cfg.particle_color),
            ],
        )
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed:
            return

        self._sim_time += dt
        self._step_counter += 1

        from isaaclab_newton.physics import NewtonManager

        if self._viewer is None:
            self._state = NewtonManager.get_state(self._scene_data_provider)
            return

        self._state = NewtonManager.get_state(self._scene_data_provider)

        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        num_envs = NewtonManager.get_num_envs()

        try:
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                try:
                    if self._state is not None:
                        body_q = getattr(self._state, "body_q", None)
                        if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                            return
                        self._viewer.log_state(self._state)
                        contacts = NewtonManager.get_contacts()
                        if contacts is not None:
                            self._viewer.log_contacts(contacts, self._state)
                        else:
                            self._log_scene_contact_sensor_arrows(num_envs)
                        if self.cfg.enable_markers:
                            render_newton_visualization_markers(
                                self._viewer, self._resolved_visible_env_ids, num_envs=num_envs
                            )
                        self._log_camera_sensor_image()
                finally:
                    self._viewer.end_frame()
            else:
                self._viewer._update()
        except Exception:
            logger.exception("[NewtonVisualizer] Viewer update failed.")

    def close(self) -> None:
        """Release viewer resources."""
        if self._is_closed:
            return
        if self._viewer is not None:
            self._viewer = None
        if self._camera_sensor is not None and self._camera_is_owned:
            remove_generated_prims(self._generated_camera_prim_paths)
        self._camera_sensor = None
        self._is_closed = True

    def _log_scene_contact_sensor_arrows(self, num_envs: int) -> None:
        """Render contact sensor data as Newton-style arrows when native contacts are unavailable."""
        if self._viewer is None:
            return
        if not self._viewer.show_contacts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return
        contact_sensors = (
            self._scene_data_provider.get_contact_sensors() if self._scene_data_provider is not None else {}
        )
        if not contact_sensors:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return

        starts: list[torch.Tensor] = []
        ends: list[torch.Tensor] = []
        for sensor in contact_sensors.values():
            sensor_starts, sensor_ends = self._contact_sensor_arrow_tensors(sensor, num_envs)
            if sensor_starts is not None and sensor_ends is not None:
                starts.append(sensor_starts)
                ends.append(sensor_ends)

        if not starts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return

        starts_t = torch.cat(starts, dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()
        ends_t = torch.cat(ends, dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()
        self._viewer.log_arrows(
            CONTACT_ARROW_PATH,
            wp.array(starts_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            wp.array(ends_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            CONTACT_ARROW_COLOR,
        )

    def _contact_sensor_arrow_tensors(self, sensor, num_envs: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build Newton-style arrow starts/ends from an Isaac Lab contact sensor."""
        try:
            data = sensor.data
            net_forces_proxy = data.net_forces_w
            net_forces = net_forces_proxy.torch if net_forces_proxy is not None else None
        except (AttributeError, NotImplementedError, RuntimeError):
            return None, None

        if net_forces is None or net_forces.numel() == 0:
            return None, None
        net_forces = self._filter_visible_env_tensor(net_forces, num_envs)

        force_threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if force_threshold is None:
            force_threshold = 0.0

        try:
            contact_pos = getattr(data, "contact_pos_w", None)
            force_matrix = getattr(data, "force_matrix_w", None)
        except NotImplementedError:
            contact_pos = None
            force_matrix = None
        if contact_pos is not None and force_matrix is not None:
            contact_pos_t = self._filter_visible_env_tensor(contact_pos.torch, num_envs)
            force_matrix_t = self._filter_visible_env_tensor(force_matrix.torch, num_envs)
            if contact_pos_t.numel() != 0 and force_matrix_t.numel() != 0:
                force_norm = torch.linalg.norm(force_matrix_t, dim=-1)
                finite_pos = torch.isfinite(contact_pos_t).all(dim=-1)
                active = (force_norm > force_threshold) & finite_pos
                if torch.any(active):
                    starts = contact_pos_t[active]
                    directions = torch.nn.functional.normalize(force_matrix_t[active], dim=-1)
                    return starts, starts + directions * CONTACT_ARROW_LENGTH

        origins = self._contact_sensor_origin_positions(sensor, data, net_forces)
        if origins is None:
            return None, None
        origins = self._filter_visible_env_tensor(origins, num_envs)

        force_norm = torch.linalg.norm(net_forces, dim=-1)
        active = force_norm > force_threshold
        if not torch.any(active):
            return None, None

        starts = origins[active]
        directions = torch.nn.functional.normalize(net_forces[active], dim=-1)
        return starts, starts + directions * CONTACT_ARROW_LENGTH

    def _contact_sensor_origin_positions(self, sensor, data, net_forces: torch.Tensor) -> torch.Tensor | None:
        """Return per-sensor origins for contact arrow starts."""
        try:
            pos_w = getattr(data, "pos_w", None)
        except NotImplementedError:
            pos_w = None
        if pos_w is not None:
            return pos_w.torch

        body_physx_view = getattr(sensor, "body_physx_view", None)
        if body_physx_view is None:
            return None
        try:
            pose = body_physx_view.get_transforms()
        except RuntimeError:
            return None
        return wp.to_torch(pose).view(*net_forces.shape[:-1], 7)[..., :3]

    def _filter_visible_env_tensor(self, tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Apply Newton visualizer visible-world filtering to a sensor tensor."""
        if self._resolved_visible_env_ids is None or tensor.ndim == 0 or tensor.shape[0] != num_envs:
            return tensor
        ids = torch.as_tensor(self._resolved_visible_env_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, ids)

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping.

        Returns:
            ``True`` while the visualizer is active, otherwise ``False``.
        """
        if not self._is_initialized or self._is_closed:
            return False
        if self._headless_no_viewer and self._viewer is None:
            return True
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config or USD camera path.

        Returns:
            Camera eye and target tuples.
        """
        return self._resolve_cfg_camera_pose("NewtonVisualizer")

    def _uses_camera_sensor_view(self) -> bool:
        """Return whether the visualizer displays camera sensor images instead of interactive camera controls."""
        return bool(self.cfg.tiled_cam_view)

    def _setup_camera_sensor_view(self, num_envs: int) -> None:
        """Resolve or create the camera sensor used by non-interactive image views."""
        if not self._uses_camera_sensor_view():
            return
        env_ids = resolve_tiled_env_indices(
            num_envs,
            self.cfg.tiled_cam_num,
            self.cfg.tiled_cam_env_indices,
            max_tiles=VISUALIZER_TILED_CAMERA_MAX_TILES,
            sample_from=self._resolved_visible_env_ids,
        )
        self._camera_env_indices = env_ids
        if self.cfg.tiled_cam_prim_path is not None:
            logger.debug(
                "[NewtonVisualizer] tiled_cam_prim_path uses existing camera sensor output; "
                "generated tiled camera pose fields are ignored."
            )
            cameras = self._scene_data_provider.get_camera_sensors()
            self._camera_sensor = find_camera_by_prim_path(cameras, self.cfg.tiled_cam_prim_path, env_ids)
            self._camera_sensor_indices = env_ids
            return

        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        count = max(1, len(env_ids))
        tile_w, tile_h = compute_tile_resolution(self.cfg.window_width, self.cfg.window_height, count)
        self._camera_sensor, self._generated_camera_prim_paths = create_visualizer_camera(
            num_envs=num_envs,
            width=tile_w,
            height=tile_h,
            renderer_cfg=NewtonWarpRendererCfg(),
        )
        self._camera_sensor_indices = env_ids
        self._camera_is_owned = True
        self._update_owned_camera_poses()

    def _update_owned_camera_poses(self) -> None:
        """Update generated camera poses from env origins or follow prims."""
        if self._camera_sensor is None or not self._camera_is_owned:
            return
        target_positions = prim_world_positions(
            self._scene_data_provider.get_usd_stage(),
            self.cfg.tiled_cam_target_prim_path,
            self._camera_env_indices,
            scene=self._scene_data_provider.get_interactive_scene(),
        )
        eyes, targets = apply_camera_target_positions(
            self._camera_sensor, target_positions, self.cfg.tiled_cam_eye, self._camera_env_indices
        )

    def _log_camera_sensor_image(self) -> None:
        """Log the selected camera sensor RGB output into Newton's image panel."""
        if self._viewer is None or self._camera_sensor is None:
            return
        if self._camera_is_owned:
            self._update_owned_camera_poses()
        if self._camera_is_owned:
            self._camera_sensor.update(dt=0.0, force_recompute=True)
        rgb = camera_rgb_batch(self._camera_sensor, self._camera_sensor_indices).contiguous()
        self._viewer.log_image("Visualizer Tiled Camera", wp.from_torch(rgb))

    def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """Apply camera eye/target pose to the Newton viewer.

        Args:
            pose: Camera eye and target tuples.
        """
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        # Match Newton's Camera native pos type: PyVec3, not wp.vec3.
        self._viewer.camera.pos = PygletVec3(*cam_pos)
        self._viewer.camera.look_at(cam_target)
        self._last_camera_pose = (cam_pos, cam_target)

    def _apply_camera_focal_length(self) -> None:
        """Apply cfg focal length to Newton's vertical-FOV camera."""
        if self._viewer is None:
            return
        self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set active viewer camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self.cfg.eye = eye_t
        self.cfg.lookat = target_t
        self._apply_camera_pose((eye_t, target_t))

    def supports_markers(self) -> bool:
        """Newton OpenGL viewer supports Isaac Lab markers through viewer-side meshes and lines."""
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        """Newton OpenGL viewer does not provide live-plot panels."""
        return False

    def is_training_paused(self) -> bool:
        """Return whether training is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()
