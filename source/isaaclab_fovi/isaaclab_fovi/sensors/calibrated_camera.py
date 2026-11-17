# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense calibrated cameras that render the physical lens's distorted raster."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from fovi.sensing.projection import CameraCalibration, CameraModel
from isaaclab import sim as sim_utils
from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.configclass import configclass

from .dense_calibrated_geometry import apply_lens_schema, dense_camera_rays
from .generalized_projection import (
    apply_generalized_projection_to_prim,
    build_dense_calibrated_textures,
    save_generalized_projection_textures,
)

if TYPE_CHECKING:
    from pxr import Usd


class CalibratedTiledCamera(Camera):
    """Dense OpenCV pinhole/fisheye camera for Isaac RTX, OVRTX and Newton/Warp.

    Lens parameters describe the raw rendered orientation. The raster is never
    rectified; each Newton ray and RTX primary ray follows the physical lens.
    ``data.intrinsic_matrices`` contains K but cannot alone describe this lens.
    """

    @property
    def supports_batch_update(self) -> bool:
        return True

    @staticmethod
    def _update_buffers_batch_impl(sensors: Sequence[CalibratedTiledCamera]) -> None:
        Camera._update_buffers_batch_impl(sensors)
        if sim_utils.SimulationContext.instance() is None:
            return  # The base implementation already invoked scalar capture.
        for sensor in sensors:
            sensor.process_rendered_frame()

    @property
    def renderer(self) -> BaseRenderer:
        """Renderer shared with task-owned visual animation bindings."""
        return self._renderer

    def __init__(self, cfg: CalibratedTiledCameraCfg) -> None:
        if cfg.camera_calibration is None:
            raise ValueError("CalibratedTiledCamera requires camera_calibration")
        self.camera_model = CameraModel.from_config(cfg.camera_calibration).resized((cfg.height, cfg.width))
        self._calibrated_rays = None
        self._invalid_pixels = None
        self._lut_directory: tempfile.TemporaryDirectory[str] | None = None
        if cfg.renderer_cfg.renderer_type not in ("isaac_rtx", "ovrtx", "newton_warp"):
            raise ValueError("Calibrated cameras support isaac_rtx, ovrtx, and newton_warp")
        super().__init__(cfg)
        if cfg.renderer_cfg.renderer_type == "ovrtx":
            self._author_lens()

    def _author_lens(self) -> None:
        if self.cfg.renderer_cfg.renderer_type in ("isaac_rtx", "ovrtx"):
            prims = sim_utils.find_matching_prims(self.cfg.prim_path, self.stage)
            if self.camera_model.model == "fisheye":
                # RTX's native OpenCV fisheye clips to an implicit image circle,
                # even when the calibrated rectangular sensor has valid corners.
                # Odd inverse dimensions put octahedral axis seams on texel centers.
                direction, inverse = build_dense_calibrated_textures(
                    self.camera_model, exit_size=max(128, 4 * max(self.camera_model.image_size)) + 1
                )
                self._lut_directory = tempfile.TemporaryDirectory(prefix="rtx_calibrated_luts_")
                direction_path, inverse_path = save_generalized_projection_textures(
                    direction, inverse, self._lut_directory.name, "calibrated"
                )
                for prim in prims:
                    apply_generalized_projection_to_prim(prim, direction_path, inverse_path)
            elif self.camera_model.model == "pinhole":
                for prim in prims:
                    apply_lens_schema(prim, self.camera_model)
            else:
                raise ValueError(f"Unsupported lens model {self.camera_model.model!r}")

    def _initialize_impl(self) -> None:
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            self._author_lens()
        super()._initialize_impl()
        directions, valid = dense_camera_rays(self.camera_model, str(self._device))
        self._invalid_pixels = (~valid)[None, ..., None]
        if self.cfg.renderer_cfg.renderer_type == "newton_warp":
            height, width = self.camera_model.image_size
            rays = torch.zeros((1, height, width, 2, 3), device=self._device)
            rays[0, :, :, 1] = directions
            self._calibrated_rays = wp.from_torch(rays, dtype=wp.vec3f)
            self._render_data.camera_rays = self._calibrated_rays

    def _read_authored_opencv_intrinsics(
        self, prim: Usd.Prim, width: int, height: int, env_id: int
    ) -> tuple[float, float, float, float]:
        # LUT schemas carry rays, not fx/fy/cx/cy attributes. The calibration
        # remains the source of public K metadata, in OpenCV pixel centers.
        return self.camera_model.intrinsics

    def _initialize_intrinsics(self) -> None:
        super()._initialize_intrinsics()
        # Newton's explicit-ray camera has no USD lens schema, so its base
        # initialization otherwise reconstructs a nominal pinhole K.
        fx, fy, cx, cy = self.camera_model.intrinsics
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        matrices = wp.array(
            np.repeat(matrix[None], self._view.count, axis=0),
            dtype=wp.mat33f,
            device=self._device,
        )
        wp.copy(self._data.intrinsic_matrices.warp, matrices)

    def set_intrinsic_matrices(
        self,
        matrices: torch.Tensor,
        focal_length: float = 1.0,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        raise NotImplementedError("Construct a new calibrated camera to change its lens; K alone is insufficient")

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        if not self._env_mask_has_any(env_mask):
            return
        super()._update_buffers_impl(env_mask)
        self.process_rendered_frame()

    def process_rendered_frame(self) -> None:
        """Apply calibrated range and validity conventions after a render."""
        if not self._env_mask_has_any(self._is_outdated):
            return
        if self.cfg.renderer_cfg.renderer_type == "ovrtx" and "distance_to_camera" in self._data.output:
            # OVRTX measures range from the near-clipped ray origin.
            self._data.output["distance_to_camera"].torch.add_(self.cfg.spawn.clipping_range[0])
        for name, output in self._data.output.items():
            fill = (
                self.cfg.spawn.clipping_range[1]
                if name in ("depth", "distance_to_image_plane", "distance_to_camera")
                else 0
            )
            output.torch.masked_fill_(self._invalid_pixels, fill)


@configclass
class CalibratedTiledCameraCfg(CameraCfg):
    """Opt-in dense calibrated lens; existing CameraCfg remains pinhole."""

    class_type: type = CalibratedTiledCamera
    camera_calibration: CameraCalibration | None = None
