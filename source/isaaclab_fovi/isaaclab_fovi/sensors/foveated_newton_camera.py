# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton foveated camera: Newton's Warp renderer casts one ray per retinal sample."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.warp import ProxyArray

from .foveated_camera import OPENGL_TO_ROS_AXES, FoveatedCamera
from .native_grid import (
    GRID_STYLE,
    allocate_grid_outputs,
    copy_grid_outputs,
    copy_updated_rows,
    grid_output_shape,
    grid_sample_indices,
    mask_grid_samples,
    proxy_outputs,
)
from .pooling import build_highres_coords, build_lowres_coords, build_pooler

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg

logger = logging.getLogger(__name__)


def explicit_rays(directions: torch.Tensor, height: int, width: int) -> wp.array:
    """Newton's ray array, shape (height, width, 1, 2): a zero origin and a unit direction per ray, shared by every
    camera.

    Args:
        directions: Unit ray directions in the OpenGL camera frame, in sample order, shape (N, 3).
        height: Rendered rows.
        width: Rendered columns; ``height * width`` must cover the N rays. Padding repeats the last ray.
    """
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"Ray directions must have shape (N, 3), got {tuple(directions.shape)}.")
    if height * width < len(directions):
        raise ValueError(f"A {height}x{width} render cannot hold {len(directions)} rays.")
    index = torch.arange(height * width, device=directions.device).clamp_max(len(directions) - 1)
    rays = directions.new_zeros(height, width, 1, 2, 3, dtype=torch.float32)
    rays[..., 1, :] = directions.float()[index].reshape(height, width, 1, 3)
    # The Warp view keeps the tensor alive.
    return wp.from_torch(rays, dtype=wp.vec3f)


class FoveatedNewtonCamera(FoveatedCamera):
    """Newton camera that renders with foveated (CMF-distributed) rays.

    Instead of a uniform pixel grid, this sensor shoots *N* rays whose
    directions are derived from :class:`fovi.sensing.coords.SamplingCoords`
    and injected into Newton's renderer, with Warp or OptiX traversal. Vector sensors publish
    ``(B, N, 1, C)``; native grid sensors publish upright ``(B, H, W, C)``.

    The samples cover the retinal window of the physical camera; the gaze turns the rendered camera pose.
    """

    cfg: FoveatedCameraCfg

    def __init__(self, cfg: FoveatedCameraCfg):
        # The parent's initialisation updates poses before the rays exist.
        self._foveated_rays: wp.array | None = None
        super().__init__(cfg)

    def _initialize_impl(self):
        # The pooler's KNN tensors are plain attributes that nn.Module.to() does not move, so the coordinates start on
        # the CPU and the pooler is built on the sensor device after the parent initializes.
        self._sampling_coords, matched_res = build_lowres_coords(self.cfg, device="cpu")
        self._native_grid = self.cfg.foveated_style == GRID_STYLE
        self._sampling_coords_highres = build_highres_coords(self.cfg, self._sampling_coords, matched_res)
        self._num_valid_samples = len(self._sampling_coords)
        self._num_ray_samples = len(self._sampling_coords_highres or self._sampling_coords)
        logger.info(
            "FoveatedNewtonCamera: %d samples (%d rendered), style=%s, cmf_a=%.4g deg, field of view=%.1f deg",
            self._num_valid_samples,
            self._num_ray_samples,
            self.cfg.foveated_style,
            self._sampling_coords.cmf_a,
            self._sampling_coords.fov,
        )

        # The parent allocates render buffers for the rendered rays: the high-resolution ones when pooling.
        ray_coords = self._sampling_coords_highres or self._sampling_coords
        self.cfg.height, self.cfg.width = grid_output_shape(ray_coords)
        renderer = self.cfg.renderer_cfg
        if renderer.render_order == "tiled":
            tile_width = renderer.tile_rendering_width
            tile_height = renderer.tile_rendering_height
            if tile_width <= 0 or tile_height <= 0:
                raise ValueError("Newton render tile dimensions must be positive.")
            if self._native_grid:
                self._validate_native_render_layout(tile_width, tile_height)
            else:
                # Pack consecutive vector rays into complete tiles. Padding is
                # private renderer storage and never enters masking or pooling.
                self.cfg.width = tile_width
                self.cfg.height = (
                    (self._num_ray_samples + tile_width * tile_height - 1) // (tile_width * tile_height) * tile_height
                )

        super()._initialize_impl()

        self._sampling_coords = self._sampling_coords.to(device=self._device)
        if self._sampling_coords_highres is not None:
            self._sampling_coords_highres = self._sampling_coords_highres.to(device=self._device)
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._initialize_gaze()
        self._pooler, self._pool_k = build_pooler(
            self.cfg, self._sampling_coords, self._sampling_coords_highres, device=str(self._device)
        )

        # The gaze geometry's rendered rays, in canonical sample order and rolled by image_rotation_deg, back in the
        # OpenGL frame Newton traces.
        self._ray_directions = self._gaze_rays * self._gaze_rays.new_tensor(OPENGL_TO_ROS_AXES)
        self._foveated_rays = explicit_rays(self._ray_directions, self.cfg.height, self.cfg.width)
        render_samples = self.cfg.height * self.cfg.width
        # Renderer buffers remain whole tiles; zero-copy canonical views hide
        # trailing padding and preserve every output's dtype and sample order.
        self._render_output = dict(self._data.output)
        self._flat_render_output = {}
        for name, value in self._render_output.items():
            tensor = value.torch
            flat = tensor.view(tensor.shape[0], render_samples, 1, tensor.shape[-1])
            self._flat_render_output[name] = ProxyArray(wp.from_torch(flat[:, : self._num_ray_samples]))
        self._data.output.clear()
        self._data.output.update(self._flat_render_output)

        self._allocate_pooled_buffers()
        source = self._pooled_output if self._pooler is not None else self._flat_render_output
        self._sample_output = None
        if self._pooler is None and not self._native_grid:
            # Newton redraws every environment, so sample sets publish a copy that changes only on update.
            self._sample_output = {
                name: torch.zeros_like(value.torch)
                for name, value in self._flat_render_output.items()
                if not (name == "rgb" and "rgba" in self._flat_render_output)
            }
            if "rgba" in self._sample_output and "rgb" in self._flat_render_output:
                self._sample_output["rgb"] = self._sample_output["rgba"][..., :3]
            source = self._sample_output
        if self._native_grid:
            self._grid_output = allocate_grid_outputs(source, self._sampling_coords)
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            source = self._grid_output
        # The buffers the camera publishes: stable for the camera's lifetime and refilled each frame.
        self._public_output = proxy_outputs(source)
        self._data.output.clear()
        self._data.output.update(self._public_output)
        self._set_published_shape(self._public_output)
        if self._native_grid and self._pooler is None:
            self._prepare_grid_publish(
                {name: value.torch.squeeze(2) for name, value in self._flat_render_output.items()}
            )
        self._render_data.camera_rays = self._foveated_rays

    def _validate_native_render_layout(self, tile_width: int, tile_height: int) -> None:
        """Reject a native grid that Newton's tiles cannot cover, before the parent allocates render buffers."""
        if self.cfg.width % tile_width or self.cfg.height % tile_height:
            raise ValueError(
                f"Native-grid Newton render size {self.cfg.width}x{self.cfg.height} "
                f"must be divisible by renderer tile size {tile_width}x{tile_height}. "
                "Choose foveated_res and pool_res_mult so the rendered grid satisfies "
                "these dimensions, or select a non-tiled Newton render_order."
            )

    def _allocate_pooled_buffers(self) -> None:
        """Allocate pooled output tensors; the renderer keeps writing the high-resolution buffers."""
        if self._pooler is None:
            return
        self._highres_output: dict[str, ProxyArray] = dict(self._data.output)
        self._pooled_output: dict[str, torch.Tensor] = {}
        for name, tex in self._highres_output.items():
            if name == "rgb" and "rgba" in self._highres_output:
                continue  # an alias of rgba, below
            tensor = tex.torch
            B, _, _, C = tensor.shape
            self._pooled_output[name] = torch.zeros(
                (B, self._num_valid_samples, 1, C), dtype=tensor.dtype, device=tensor.device
            )
        # The renderer publishes rgb as a view of rgba whenever either is requested; so does the pooled output.
        if "rgb" in self._highres_output and "rgba" in self._pooled_output:
            self._pooled_output["rgb"] = self._pooled_output["rgba"][..., :3]
        self._prepare_segmentation_pooling(self._highres_output)
        dtypes_summary = {n: str(t.dtype).removeprefix("torch.") for n, t in self._highres_output.items()}
        logger.info(
            "FoveatedNewtonCamera: %s pooling (mode=%s k=%s sample_cortex=%s N_hi=%d -> N_lo=%d) outputs=%s",
            type(self._pooler).__name__,
            self.cfg.pool_mode,
            self._pool_k,
            self.cfg.pool_sample_cortex,
            self._num_ray_samples,
            self._num_valid_samples,
            dtypes_summary,
        )

    def _update_poses(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        env_mask: wp.array | None = None,
        frame_op: int = 0,
    ) -> None:
        """Read base poses, turn them by the gaze, update the renderer once, and inject the foveated rays.

        The scene pose is read even without a gaze: the camera may inherit
        motion from a parent body.
        """
        # The base update turns the poses by the gaze in _update_camera_state, then hands them to the renderer.
        Camera._update_poses(self, env_ids, env_mask=env_mask, frame_op=frame_op)
        if self._foveated_rays is not None:
            self._render_data.camera_rays = self._foveated_rays

    def _update_buffers_impl(self, env_mask: wp.array):
        """Render, then mask and publish the updated environments, pooling high-resolution samples if enabled.

        Newton redraws every environment's render buffer, each at the pose and gaze it last received. The sensor mask,
        its visibility record and the published buffers, which the camera owns, change only for the updated
        environments.
        """
        self._data.output.clear()
        self._data.output.update(self._render_output)
        try:
            super()._update_buffers_impl(env_mask)
        finally:
            self._data.output.clear()
            self._data.output.update(self._public_output)
        if not self._env_mask_has_any(env_mask):
            return
        if self._pooler is None:
            if self._grid_publish:
                self._publish_grid(env_mask)
            else:
                self._apply_sensor_fov_mask(self._flat_render_output, env_mask)
                for name, samples in self._sample_output.items():
                    if name == "rgb" and "rgba" in self._sample_output:
                        continue  # rgba alias
                    copy_updated_rows(self._flat_render_output[name].torch, samples, env_mask)
            return
        mask_grid_samples(self._highres_output, self._ray_coords)
        self._apply_sensor_fov_mask(self._highres_output, env_mask)
        for name, hi in self._highres_output.items():
            if name == "rgb" and "rgba" in self._highres_output:
                continue  # rgba alias
            # Pooling covers every environment; only the updated ones are published.
            copy_updated_rows(self._pool_output(name, hi.torch), self._pooled_output[name], env_mask)
        if self._native_grid:
            copy_grid_outputs(self._pooled_output, self._grid_output, self._grid_indices)

    @staticmethod
    def _pose_update_turns_gaze(cfg: FoveatedCameraCfg) -> bool:
        return True
