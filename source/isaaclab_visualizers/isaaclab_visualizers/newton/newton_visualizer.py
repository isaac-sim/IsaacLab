# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL visualization of the simulation model and scene-camera images."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import newton
import numpy as np  # noqa: F401 — used in type hints and colorization helpers
import torch
import warp as wp
from isaaclab_newton.physics import NewtonBackendCfg, NewtonManager

from pxr import Usd

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sensors.camera import Camera
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers

from .newton_viewer import NewtonViewerGL, _NewtonCameraControls
from .newton_visualizer_cfg import NewtonGLVisualizerCfg

logger = logging.getLogger(__name__)


def _newton_scalar_base_name(name: str) -> str:
    """Strip a trailing ``[N]`` component index from a scalar name to get the term base name."""
    if name.endswith("]") and "[" in name:
        bracket = name.rfind("[")
        if name[bracket + 1 : -1].isdigit():
            return name[:bracket]
    return name


CONTACT_ARROW_PATH = "/contacts"
"""Viewer path used for native and synthesized contact arrows."""

CONTACT_ARROW_COLOR = (0.0, 1.0, 0.0)
"""Color used by Newton's native contact visualization."""

CONTACT_ARROW_LENGTH = 0.1
"""Length of synthesized contact arrows in meters."""


@dataclass(frozen=True)
class _MeshSubmission:
    """Mesh data staged for the next Newton viewer frame."""

    name: str
    points: wp.array
    indices: wp.array
    normals: wp.array | None
    uvs: wp.array | None
    texture: np.ndarray | str | None
    hidden: bool
    backface_culling: bool
    color: tuple[float, float, float] | None
    roughness: float | None
    metallic: float | None
    dynamic: bool
    opacity: float | None


_NEWTON_ICON_DIR = Path(newton.__file__).parent / "_src" / "viewer" / "gl"


if TYPE_CHECKING:
    from newton import State

    from isaaclab.scene_data import SceneDataProvider


# ---------------------------------------------------------------------------
# Shared base visualizer
# ---------------------------------------------------------------------------


class NewtonGLVisualizer(_NewtonCameraControls, BaseVisualizer):
    """Newton model rendering, picking, and contact overlays used by the GL visualizer."""

    @property
    def visual_material_writer(self):
        """Return the shared Newton model color-writer factory."""
        return self.backend.create_visual_material_writer

    class _ViewerPickingBinding:
        """Stable Newton-manager callback for viewer picking.

        CUDA graphs record picking arrays by address, so closing the window
        neutralizes and retains them until the captured graph is gone.
        """

        def __init__(self) -> None:
            self._viewer: NewtonViewerGL | None = None
            self._retained_picking = None

        def bind(self, viewer: NewtonViewerGL) -> None:
            """Bind picking to the current viewer model."""
            self._viewer = viewer
            self._retained_picking = None

        def apply(self, state: State) -> None:
            """Apply picking while the viewer is active."""
            if self._viewer is None:
                # Host callbacks do not run during graph replay, so reaching
                # this branch means captured inputs are no longer needed.
                self._retained_picking = None
                return
            self._viewer.apply_forces(state)

        def deactivate(self) -> None:
            """Make captured picking inert while preserving its inputs."""
            viewer = self._viewer
            if viewer is None:
                return

            picking = getattr(viewer, "picking", None)
            if picking is not None:
                viewer.picking_enabled = False
                picking.release()

            self._retained_picking = picking
            self._viewer = None

    def __init__(self, cfg: NewtonGLVisualizerCfg):
        """Initialize the GL scene renderer and window.

        Args:
            cfg: Newton visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonGLVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._step_counter = 0
        self._runtime_headless: bool = False
        self.backend = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._headless_no_viewer = False
        self._viewer_picking_binding = self._ViewerPickingBinding()
        self._picking_enabled = False
        self._live_plots_manager_visible: dict[str, bool] = {}
        self._pending_mesh_submissions: dict[str, _MeshSubmission] = {}

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        scene_data_provider: SceneDataProvider,
        *,
        cameras: list[PerspectiveCameraCfg | Camera],
        stage: Usd.Stage | None = None,
    ) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
            stage: Authored scene stage, when available.
        """

        if self._is_initialized:
            logger.debug("[%s] initialize() called while already initialized.", type(self).__name__)
            return

        super().initialize(scene_data_provider, cameras=cameras, stage=stage)
        newton_backend_active = self.physics_backend == "newton"
        sim = SimulationContext.instance()
        physics_manager = sim.physics_manager
        picking_supported = newton_backend_active and bool(
            getattr(physics_manager, "_supports_rigid_body_force_input", False)
        )
        num_envs = scene_data_provider.num_envs
        metadata = {"num_envs": num_envs, "physics_backend": self.physics_backend, "gravity": sim.cfg.gravity}
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = scene_data_provider.create_mapping(list(self.backend.model.body_label))

        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        if runtime_headless and not self.cfg.headless:
            logger.warning(
                "[NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )
        self._runtime_headless = runtime_headless

        # Use pyglet's EGL headless backend when requested or when no Linux X display is available.
        # NOTE: this call is only effective when ``DISPLAY`` is unset on Linux.  When a display
        # is present, ``from newton.viewer import ViewerGL`` at module-import time
        # already initialised pyglet (and resolved the ``Window`` class), so setting
        # ``pyglet.options["headless"]`` here is a no-op.  In that situation ``cfg.headless=True``
        # has no effect and a real windowed viewer is created.  To guarantee headless behaviour
        # when a display is present, unset DISPLAY before importing this module.
        if runtime_headless:
            import pyglet

            pyglet.options["headless"] = True

        self._picking_enabled = self.cfg.enable_picking and picking_supported and not runtime_headless
        self._viewer = self._create_viewer(runtime_headless, metadata)

        if self._viewer is not None:
            self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
            self._viewer.set_model(self.backend.model)
            if self._picking_enabled:
                # Keep Newton's public force path scoped to picking for this integration.
                self._viewer.wind = None
            self._viewer.set_visible_worlds(self._env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_camera_focal_length()
            self._apply_camera_pose((self.cfg.eye, self.cfg.lookat))
            self._viewer._paused = False

            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled

            self._apply_viewer_post_init()

        self._setup_streaming_view(
            num_envs,
            visible_env_ids=self._env_ids,
            target_aspect=self.cfg.window_width / self.cfg.window_height,
        )

        if self._viewer is not None:
            self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
            self._select_camera(self._camera_index)

        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else num_envs
        try:
            current_eye = tuple(float(x) for x in self._viewer.camera.pos) if self._viewer is not None else self.cfg.eye
        except AttributeError:
            current_eye = self.cfg.eye
        self._log_initialization_table(
            logger=logger,
            title=f"{type(self).__name__} Configuration",
            rows=[
                ("eye", current_eye),
                ("lookat", self._last_camera_pose[1] if self._last_camera_pose else self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("background_color", self.cfg.background_color),
                ("streaming_view", self.cfg.streaming_view),
                ("streaming_gt_types", list(self.cfg.streaming_gt_types)),
                ("num_visualized_envs", num_visualized_envs),
                ("headless", self.cfg.headless),
                ("show_particles", self.cfg.show_particles),
                ("enable_picking", self._picking_enabled),
            ],
        )
        if self._viewer is not None and self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)
            NewtonManager.register_state_force_callback(self._viewer_picking_binding.apply)
        if self._viewer is not None and self.cfg.enable_picking and not picking_supported:
            logger.info(
                "[NewtonVisualizer] Object dragging is disabled because the active physics solver does not support"
                " rigid-body force input."
            )
        self._is_initialized = True
        # Inform the viewer whether contact data is available so the UI can grey
        # out "Show Contacts" when neither native Newton contacts nor a ContactSensor
        # exists in the scene.
        if self._viewer is not None:
            contact_sensors = self._scene_data_provider.get_contact_sensors() if self._scene_data_provider else {}
            self._viewer._contacts_available = newton_backend_active or bool(contact_sensors)

    def _apply_model_visualization_options(self) -> None:
        """Apply configured options reset by Newton model changes."""
        if self._viewer is None:
            return
        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_collision = self.cfg.show_collision
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_inertia_boxes = self.cfg.show_inertia_boxes
        self._viewer.show_com = self.cfg.show_com
        self._viewer.show_particles = self.cfg.show_particles

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed:
            return

        self._sim_time += dt
        self._step_counter += 1

        # Headless capture requests current SDP arrays in render_rgb_array().
        if self._runtime_headless or self._viewer is None:
            return

        if self._step_counter % self._viewer._update_frequency != 0:
            return
        self._navigate_scene_camera()

        num_envs = self.backend.model.num_envs

        try:
            if not self._viewer.is_paused():
                self._viewer.begin_frame(self._sim_time)
                try:
                    if self._uses_streaming_view():
                        self._log_streaming_image()
                    else:
                        backend, provider = self.backend, self._scene_data_provider
                        state, count = backend.state_0, backend.model.body_count
                        poses = SceneDataFormat.Transform()
                        if provider.get_transforms(poses, mapping=self._transform_mapping, count=count):
                            state.body_q = poses.transforms
                        if backend.geometry_offsets:
                            provider.get_geometry_points(output=state.particle_q, offsets=backend.geometry_offsets)
                        if state.body_q is not None and state.body_q.shape[0] == 0:
                            self._log_pending_meshes()
                            return
                        self._viewer.log_state(state)
                        contacts = NewtonManager.get_contacts()
                        if contacts is not None:
                            self._viewer.log_contacts(contacts, state)
                        else:
                            self._log_scene_contact_sensor_arrows(num_envs)
                        if self.cfg.enable_markers:
                            render_newton_visualization_markers(self._viewer, self._env_ids, num_envs=num_envs)
                        self._log_pending_meshes()
                    self._render_live_plots()
                finally:
                    self._viewer.end_frame()
                    if not self._viewer.is_running():
                        self._viewer_picking_binding.deactivate()
            else:
                self._pump_paused()
                if not self._viewer.is_running():
                    self._viewer_picking_binding.deactivate()
        except Exception:
            logger.exception("[%s] Viewer update failed.", type(self).__name__)

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested via the viewer UI."""
        if self._viewer is not None:
            return self._viewer.is_reset_requested()
        return False

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        if self._viewer is not None:
            return self._viewer.consume_reset_request()
        return False

    def reset(self, soft: bool = False) -> None:
        """Rebind viewer resources after a hard Newton model reset."""
        super().reset(soft)
        if soft or not self._is_initialized or self._is_closed:
            return

        sim = SimulationContext.instance()
        backend = sim.get_or_create_backend(self.newton_cfg)
        if backend is self.backend:
            return
        self.backend = backend
        self._transform_mapping = self._scene_data_provider.create_mapping(list(backend.model.body_label))
        if self._viewer is not None:
            self._viewer.set_model(self.backend.model)
            if self._picking_enabled:
                self._viewer.wind = None
            self._viewer.register_ui_callback(self._viewer._render_training_controls, position="side")
            self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
            self._viewer.set_visible_worlds(self._env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_viewer_post_init()
            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled
            if self._picking_enabled:
                self._viewer_picking_binding.bind(self._viewer)

    def _release_viewer(self) -> None:
        """Release picking and the viewer reference without destroying Kit's shared GL context."""
        viewer = self._viewer
        if viewer is None:
            return
        try:
            if self._picking_enabled:
                # Keep the stable callback registered: captured graphs replay
                # its now-neutral device inputs without retaining the viewer.
                self._viewer_picking_binding.deactivate()
        finally:
            self._viewer = None

    def close(self) -> None:
        """Release viewer resources."""
        if self._is_closed:
            return
        try:
            self._release_viewer()
        finally:
            self._pending_mesh_submissions.clear()
            self.backend = self._transform_mapping = None
            super().close()

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping."""
        if not self._is_initialized or self._is_closed:
            return False
        if self._headless_no_viewer and self._viewer is None:
            return True
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def supports_markers(self) -> bool:
        """Newton viewers support Isaac Lab markers through viewer-side meshes and lines."""
        return bool(self.cfg.enable_markers)

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

    def is_key_down(self, key: str) -> bool:
        """Return whether a key is held in the viewer window.

        Args:
            key: Key name, such as ``"i"`` or ``"space"``.

        Returns:
            True if the viewer is open and the key is held, False otherwise.
        """
        if not self._is_initialized or self._viewer is None:
            return False
        return bool(self._viewer.is_key_down(key))

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

    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        dynamic: bool = False,
        opacity: float | None = None,
    ) -> None:
        """Stage a mesh registration or update for the next Newton viewer frame.

        Newton viewers require geometry updates between ``begin_frame()`` and
        ``end_frame()``. Staging here keeps that lifecycle internal to the
        visualizer and lets callers submit meshes before :meth:`step`.

        Args:
            name: Unique viewer path for the mesh.
            points: Vertex positions [m].
            indices: Flattened triangle vertex indices.
            normals: Optional per-vertex normals.
            uvs: Optional per-vertex texture coordinates.
            texture: Optional texture path or image.
            hidden: Whether to hide the mesh.
            backface_culling: Whether to cull back-facing triangles.
            color: Optional RGB base color in ``[0, 1]``.
            roughness: Optional surface roughness in ``[0, 1]``.
            metallic: Optional surface metallic value in ``[0, 1]``.
            dynamic: Whether the mesh topology may change between updates.
            opacity: Optional surface opacity in ``[0, 1]``.

        Raises:
            RuntimeError: If the Newton viewer has not been initialized.
        """
        if not self._is_initialized or self._viewer is None:
            raise RuntimeError("Newton visualizer must be initialized before logging meshes.")
        self._pending_mesh_submissions[name] = _MeshSubmission(
            name=name,
            points=points,
            indices=indices,
            normals=normals,
            uvs=uvs,
            texture=texture,
            hidden=hidden,
            backface_culling=backface_culling,
            color=color,
            roughness=roughness,
            metallic=metallic,
            dynamic=dynamic,
            opacity=opacity,
        )

    def _log_pending_meshes(self) -> None:
        """Publish staged meshes inside the active viewer frame."""
        if self._viewer is None or not self._pending_mesh_submissions:
            return

        submissions = tuple(self._pending_mesh_submissions.values())
        self._pending_mesh_submissions.clear()
        for mesh in submissions:
            self._viewer.log_mesh(
                mesh.name,
                mesh.points,
                mesh.indices,
                normals=mesh.normals,
                uvs=mesh.uvs,
                texture=mesh.texture,
                hidden=mesh.hidden,
                backface_culling=mesh.backface_culling,
                color=mesh.color,
                roughness=mesh.roughness,
                metallic=mesh.metallic,
                dynamic=mesh.dynamic,
                opacity=mesh.opacity,
            )

    # ------------------------------------------------------------------
    # Hook methods — override in subclasses
    # ------------------------------------------------------------------

    def _pump_paused(self) -> None:
        """Keep the event loop alive while simulation is paused without advancing state."""
        self._viewer.begin_frame(self._sim_time)
        try:
            if self._uses_streaming_view():
                self._log_streaming_image()
            else:
                self._log_pending_meshes()
        finally:
            self._viewer.end_frame()

    def _render_headless_frame(self) -> None:
        """Render on demand, borrowing current SDP arrays and preserving paused frames."""
        if not self._runtime_headless or self._viewer.is_paused():
            return
        if self._uses_streaming_view():
            self._viewer.begin_frame(self._sim_time)
            try:
                self._log_streaming_image()
            finally:
                self._viewer.end_frame()
            return
        backend, provider = self.backend, self._scene_data_provider
        poses = SceneDataFormat.Transform()
        if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
            backend.state_0.body_q = poses.transforms
        if backend.geometry_offsets:
            provider.get_geometry_points(output=backend.state_0.particle_q, offsets=backend.geometry_offsets)
        self._viewer.begin_frame(self._sim_time)
        try:
            self._viewer.log_state(backend.state_0)
            if self.cfg.enable_markers:
                render_newton_visualization_markers(self._viewer, self._env_ids, num_envs=backend.model.num_envs)
            self._log_pending_meshes()
        finally:
            self._viewer.end_frame()

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

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
            net_forces_proxy = data.net_normal_forces_w
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
            force_matrix = getattr(data, "normal_force_matrix_w", None)
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
        num_envs, num_bodies = net_forces.shape[0], net_forces.shape[1]
        return wp.to_torch(pose).view(num_bodies, num_envs, 7).transpose(0, 1)[..., :3]

    def _filter_visible_env_tensor(self, tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Apply Newton visualizer visible-world filtering to a sensor tensor."""
        if self._env_ids is None or tensor.ndim == 0 or tensor.shape[0] != num_envs:
            return tensor
        ids = torch.as_tensor(self._env_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, ids)

    def _uses_streaming_view(self) -> bool:
        return self._camera_sensor is not None

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> NewtonViewerGL:
        if not runtime_headless:
            # pyglet sets WM_CLASS from the window caption, which ViewerGL defaults to "Newton".
            write_desktop_entry("isaaclab-newton-gl-viewer", "Newton", "Newton", _NEWTON_ICON_DIR / "icon_64.png")
        return NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

    def register_ui_callback(self, callback: Callable[[Any], None], position: str = "side") -> None:
        """Add a panel to the viewer's ImGui interface.

        Args:
            callback: Callable invoked with the ``imgui`` module every UI frame.
            position: Newton viewer UI slot, such as ``"side"`` or ``"panel"``.
        """
        if self._viewer is not None:
            self._viewer.register_ui_callback(callback, position=position)

    def request_close(self) -> None:
        """Close the viewer window once the current frame ends.

        Safe to call from a UI callback, where closing immediately would destroy the GL context
        mid-frame.
        """
        if self._viewer is not None:
            self._viewer.request_close()

    def supports_live_plots(self) -> bool:
        """Newton GL supports live scalar/array plots via the ImGui sidebar."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting and add per-manager sidebar toggles.

        Calls the base implementation to populate :attr:`_live_plot_sources`, then registers
        the Live Plots collapsing section in the Newton viewer sidebar.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if not self._live_plot_sources or self._viewer is None:
            return
        self._live_plots_manager_visible = {source.manager_name: True for source in self._live_plot_sources}
        self._viewer._live_plots_callback = self._live_plots_panel_imgui

    def _live_plots_panel_imgui(self, imgui) -> None:
        """Render a Live Plots collapsing section in the Newton GL sidebar."""
        if not self._live_plot_sources or self._viewer is None:
            return
        viewer = self._viewer
        plots = viewer._plot_logger
        scalar_buffers = plots._scalar_buffers
        array_buffers = plots._array_buffers
        if not scalar_buffers and not array_buffers:
            return

        _ip = getattr(viewer, "_implot", None)
        scalar_arrays = plots._scalar_arrays
        n = plots._plot_history_size
        s = viewer.gui.ui.dpi_scale
        plot_h = 180 * s

        groups: dict[str, list[str]] = {}
        for name in scalar_buffers or {}:
            base = _newton_scalar_base_name(name)
            groups.setdefault(base, []).append(name)

        episode_keys = [k for k in groups if k.startswith("episode/")]
        other_keys = [k for k in groups if not k.startswith("episode/")]
        groups = {k: groups[k] for k in episode_keys + other_keys}

        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Live Plots"):
            return
        imgui.separator()

        for base_name, names in groups.items():
            term_label = base_name.rsplit("/", 1)[-1]
            if not imgui.collapsing_header(term_label):
                continue
            for name in names:
                buf = scalar_buffers.get(name, [])
                arr = scalar_arrays.get(name)
                if arr is None:
                    arr = np.full(n, np.nan, dtype=np.float32)
                    arr[n - len(buf) :] = np.array(buf, dtype=np.float32)
                    scalar_arrays[name] = arr
            if _ip is not None and _ip.begin_plot(f"##{base_name}", imgui.ImVec2(-1, plot_h)):
                _auto = _ip.AxisFlags_.auto_fit.value
                _ip.setup_axes("", "", _auto, _auto)
                _ip.setup_finish()
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        suffix = name[len(base_name) :]
                        label = suffix if suffix else term_label
                        _ip.plot_line(label, arr)
                _ip.end_plot()
            else:
                graph_size = imgui.ImVec2(-1, 80 * s)
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        buf = scalar_buffers.get(name, [])
                        overlay = f"{buf[-1]:.4g}" if buf else ""
                        imgui.plot_lines(f"##{name}", arr, graph_size=graph_size, overlay_text=overlay)

        panel_width = imgui.get_content_region_avail().x
        for name, array in array_buffers.items():
            if imgui.collapsing_header(name):
                plots._render_array_heatmap(imgui, name, array, panel_width - 20.0 * s, dpi_scale=s)

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Newton viewer's built-in plot panel."""
        if self._viewer is None or not self._live_plot_sources:
            return
        if getattr(self, "_runtime_headless", False):
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, getattr(self.cfg, "live_plots_update_interval", 10)) != 0:
            return
        for source in self._live_plot_sources:
            if not self._live_plots_manager_visible.get(source.manager_name, True):
                continue
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def _apply_viewer_post_init(self) -> None:
        """Apply GL-specific renderer settings after viewer construction."""
        self._viewer.up_axis = 2  # Z-up
        self._viewer.scaling = 1.0
        self._viewer.particle_color = self.cfg.particle_color
        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe
        # Accept list/tuple/array-like config colors; provide a stable tuple for nanobind conversion.
        if self.cfg.background_color is None:
            self._viewer.renderer.draw_sky = self.cfg.enable_sky
            upper_color = self.cfg.sky_upper_color
            lower_color = self.cfg.sky_lower_color
        else:
            self._viewer.renderer.draw_sky = False
            upper_color = lower_color = self.cfg.background_color
        self._viewer.renderer.sky_upper = tuple(upper_color)
        self._viewer.renderer.sky_lower = tuple(lower_color)
        self._viewer.renderer._light_color = tuple(self.cfg.light_color)

    def render_rgb_array(self) -> np.ndarray:
        """Return the latest RGB frame rendered by the Newton GL viewer.

        In headless mode, current transforms and geometry are requested from SDP
        only when a frame is captured.

        Returns:
            The latest viewer framebuffer as a uint8 array with shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If the visualizer has not been initialized.
        """
        if self._viewer is None:
            raise RuntimeError("NewtonGLVisualizer must be initialized before capturing an RGB frame.")
        self._render_headless_frame()
        return self._viewer.get_frame().numpy()

    def _log_streaming_image(self) -> None:
        """Present sensor pixels directly; paused rendering retains the last displayed image."""
        composite = self._streaming_frame.data
        if composite is None or self._streaming_frame.timestamp < 0 or not self._viewer.is_paused():
            composite = self.render_tiled_rgba()
        if composite is None:
            raise ValueError("Scene-camera display requires at least one selected camera frame.")
        self._viewer.log_image("Streaming View", composite, fullscreen=True)
