# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive RTX images from the simulation-owned renderer."""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from pxr import Gf

from isaaclab.renderers import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp
from isaaclab.visualizers import PerspectiveCameraCfg
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .newton_viewer import NewtonViewerGL, _NewtonCameraControls
from .newton_visualizer_cfg import NewtonRTXVisualizerCfg

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.renderers import BaseRenderer
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.sensors.camera import Camera


class NewtonRTXVisualizer(_NewtonCameraControls, BaseVisualizer):
    """Present shared OVRTX render products in a Newton window.

    The renderer owns scene import, scene synchronization, and native products. The window owns
    controls and presentation. Perspective views have their own resizable product; scene-camera
    selections borrow published sensor pixels without changing sensor resolution or lifetime.
    """

    def __init__(self, cfg: NewtonRTXVisualizerCfg, *, renderer: BaseRenderer | None = None):
        """Create a window consumer with the renderer supplied by the simulation registry."""
        super().__init__(cfg)
        self._renderer = renderer
        self._viewer: NewtonViewerGL | None = None
        self._runtime_headless = cfg.headless
        self._step_counter = 0
        self._render_data = None
        self._camera_data: CameraData | None = None
        self._capture_frame: ProxyArray | None = None
        self._intrinsic_parameters: wp.array | None = None
        self._projection: tuple | None = None
        self._view_matrix: np.ndarray | None = None
        self._display_image: wp.array | None = None

    def initialize(
        self,
        scene_data_provider: SceneDataProvider,
        *,
        cameras: list[PerspectiveCameraCfg | Camera],
        stage: Usd.Stage | None = None,
    ) -> None:
        """Bind scene cameras and create the presentation window after scene preparation."""
        if self._is_initialized:
            return
        if self.physics_backend in ("physx", "isaacsim_physx"):
            raise RuntimeError(
                "Newton RTX is kitless and cannot share a process with Kit physics; use Newton or OVPhysX."
            )
        if self._renderer is None:
            raise ValueError("Newton RTX requires a renderer supplied by SimulationContext.")
        super().initialize(scene_data_provider, cameras=cameras, stage=stage)
        cfg = self.cfg
        self._runtime_headless = cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        self._viewer = NewtonViewerGL(
            width=cfg.window_width,
            height=cfg.window_height,
            headless=self._runtime_headless,
            metadata={"num_envs": scene_data_provider.num_envs, "physics_backend": self.physics_backend},
            update_frequency=cfg.update_frequency,
        )
        self._viewer.renderer.set_title("Isaac Lab RTX")
        self._setup_streaming_view(
            scene_data_provider.num_envs,
            visible_env_ids=self._env_ids,
            target_aspect=cfg.window_width / cfg.window_height,
            select_camera=False,
        )
        self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
        self._select_camera(self._camera_index)
        if any(isinstance(source, PerspectiveCameraCfg) for source in self._camera_choices):
            self._update_render_product()
        self._is_initialized = True

    def _select_camera(self, index: int) -> None:
        super()._select_camera(index)
        self._display_image = None

    def _update_render_product(self) -> None:
        """Allocate the product before material writers are finalized, then update it on resize."""
        viewer, renderer = self._viewer, self._renderer
        width, height = (max(1, size) for size in viewer.renderer.window.get_framebuffer_size())
        if self._camera_data is None or self._camera_data.image_shape != (height, width):
            if self._render_data is None:
                camera_cfg = CameraCfg(
                    prim_path="/Render/Perspective",
                    width=width,
                    height=height,
                    data_types=["rgba"],
                    background_color=self.cfg.background_color,
                    renderer_cfg=self.cfg.renderer_cfg,
                )
                settings = {"omni:rtx:quality": ("Int", 0), **self.cfg.render_settings}
                spec = CameraRenderSpec(camera_cfg, str(viewer.device), 1, (), 1, render_settings=settings)
                self._render_data = renderer.create_render_data(spec)
            else:
                renderer.resize_render_product(self._render_data, width=width, height=height)
            self._camera_data = CameraData.allocate(
                ["rgba"], height, width, 1, str(viewer.device), renderer.supported_output_types()
            )
            self._camera_data.create_buffers(1, str(viewer.device))
            renderer.set_outputs(self._render_data, self._camera_data.output)
            self._capture_frame = ProxyArray(wp.zeros(1, dtype=wp.int64, device=viewer.device))
            self._intrinsic_parameters = wp.empty((5, 1), dtype=wp.float32, device=viewer.device)
            self._projection = self._view_matrix = None

    def _render_perspective(self) -> wp.array:
        """Resolve camera controls, then render directly into this product's device output."""
        self._update_render_product()
        viewer, renderer = self._viewer, self._renderer
        data, render_data = self._camera_data, self._render_data
        height, width = data.image_shape
        # Host camera controls are resolved before uploading the small pose/calibration inputs.
        view = viewer.camera.get_view_matrix().reshape(4, 4).T
        position = orientation = None
        if self._view_matrix is None or not np.array_equal(view, self._view_matrix):
            world = np.linalg.inv(view)
            quaternion = Gf.Matrix3d(world[:3, :3].T.tolist()).ExtractRotation().GetQuat()
            position = np.asarray(world[:3, 3], dtype=np.float32).reshape(1, 3)
            orientation = np.array([[*quaternion.GetImaginary(), quaternion.GetReal()]], dtype=np.float32)
            self._view_matrix = view.copy()
        projection = (width, height, viewer.camera.fov)
        if projection != self._projection:
            vertical = 48.0 * math.tan(math.radians(viewer.camera.fov) / 2)
            horizontal = vertical * width / height
            parameters = np.array([[24.0], [horizontal], [vertical], [0.0], [0.0]], dtype=np.float32)
            focal_px = height * 24.0 / vertical
            intrinsics = np.array([[[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]]], dtype=np.float32)
            self._intrinsic_parameters.assign(parameters)
            data.intrinsic_matrices.warp.assign(intrinsics)
            renderer.update_camera_intrinsics(render_data, data.intrinsic_matrices.warp, self._intrinsic_parameters)
            self._projection = projection
        if position is not None:
            data.pos_w.warp.assign(position)
            data.quat_w_world.warp.assign(orientation)
            convert_camera_frame_orientation_convention_wp(
                data.quat_w_world.warp, data.quat_w_world.warp, "opengl", "world", device=str(viewer.device)
            )
            renderer.update_camera(render_data, data.pos_w, data.quat_w_world, data.intrinsic_matrices)
        renderer.update_transforms()
        renderer.update_geometries()
        frame = self._capture_frame.warp
        wp.map(wp.add, frame, wp.int64(1), out=frame)
        renderer.prepare_capture(render_data, data, self._capture_frame)
        renderer.render(render_data)
        renderer.read_output(render_data, data)
        return data.output["rgba"].warp.reshape((height, width, 4))

    def step(self, dt: float) -> None:
        """Resolve controls, render on the GPU, and present one image."""
        if not self._is_initialized or self._is_closed:
            return
        self._sim_time += dt
        self._step_counter += 1
        viewer = self._viewer
        if self._runtime_headless or self._step_counter % viewer._update_frequency:
            return
        self._navigate_scene_camera()
        if not viewer.is_paused() or self._display_image is None:
            self._display_image = (
                self.render_tiled_rgba() if self._camera_sensor is not None else self._render_perspective()
            )
        viewer.begin_frame(self._sim_time)
        try:
            if self._display_image is not None:
                viewer.log_image("RTX View", self._display_image, fullscreen=True)
        finally:
            viewer.end_frame()

    def render_rgb_array(self) -> np.ndarray | None:
        """Capture the selected view and explicitly download an RGB image for recording."""
        if self._viewer is None:
            return None
        self._navigate_scene_camera()
        if not self._viewer.is_paused() or self._display_image is None:
            self._display_image = (
                self.render_tiled_rgba() if self._camera_sensor is not None else self._render_perspective()
            )
        image = self._display_image
        return None if image is None else np.ascontiguousarray(image.numpy()[..., :3])

    def is_running(self) -> bool:
        """Return whether the presentation window is open."""
        return self._viewer is not None and not self._is_closed and self._viewer.is_running()

    def is_training_paused(self) -> bool:
        """Return the window's training pause state."""
        return self._viewer is not None and self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Return the window's rendering pause state."""
        return self._viewer is not None and self._viewer.is_rendering_paused()

    def is_reset_requested(self) -> bool:
        """Return whether the user requested an episode reset."""
        return self._viewer is not None and self._viewer.is_reset_requested()

    def consume_reset_request(self) -> bool:
        """Consume an episode reset request from the window."""
        return self._viewer is not None and self._viewer.consume_reset_request()

    def is_key_down(self, key: str) -> bool:
        """Return whether a window key is held."""
        return self._viewer is not None and self._viewer.is_key_down(key)

    def reset(self, soft: bool = False) -> None:
        """Reset this product's capture state while preserving shared renderer resources."""
        super().reset(soft)
        if self._render_data is not None:
            self._renderer.reset(self._render_data)
            self._capture_frame.warp.zero_()
        self._display_image = None

    def close(self) -> None:
        """Release this view's product and window; the simulation retains the shared renderer."""
        if self._is_closed:
            return
        try:
            if self._render_data is not None:
                self._renderer.cleanup(self._render_data)
        finally:
            try:
                if self._viewer is not None:
                    self._viewer.close()
            finally:
                self._viewer = self._renderer = self._render_data = self._camera_data = None
                self._display_image = self._capture_frame = self._intrinsic_parameters = None
                super().close()
