# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton foveated tiled camera sensor.

Subclasses :class:`isaaclab.sensors.Camera` to replace uniform pinhole
rays with CMF-distributed foveated rays computed from
:class:`fovi.sensing.coords.SamplingCoords`.  Uses Newton's Warp renderer
to shoot custom ray arrays.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.warp import ProxyArray

from .calibrated_geometry import retinal_directions
from .foveated_camera import FoveatedCamera
from .foveated_rays import compute_explicit_ray_array, compute_foveated_ray_array
from .native_grid import (
    GRID_STYLE,
    allocate_grid_outputs,
    copy_grid_outputs,
    grid_output_shape,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
    proxy_outputs,
)
from .pooling import (
    build_highres_coords,
    build_lowres_coords,
    build_pooler,
    is_segmentation_output,
    pool_hwc_buffer,
)

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _pack_vector_rays(
    source: wp.array2d(dtype=wp.vec3f),
    target: wp.array2d(dtype=wp.vec3f),
) -> None:
    i = wp.tid()
    source_i = wp.min(i, source.shape[0] - 1)
    target[i, 0] = source[source_i, 0]
    target[i, 1] = source[source_i, 1]


class FoveatedNewtonCamera(FoveatedCamera):
    """Newton camera that renders with foveated (CMF-distributed) rays.

    Instead of a uniform pixel grid, this sensor shoots *N* rays whose
    directions are derived from :class:`fovi.sensing.coords.SamplingCoords`
    and injected into Newton's Warp renderer. Vector sensors publish
    ``(B, N, 1, C)``; native grid sensors publish upright ``(B, H, W, C)``.

    In planar geometry the samples cover the horizontal field of view of the
    ``spawn`` optics. In spherical geometry they cover the calibrated retinal
    window, and gaze is applied as a per-environment rotation of the rays.
    """

    cfg: FoveatedCameraCfg

    def __init__(self, cfg: FoveatedCameraCfg):
        # Spawn optics are only needed for planar geometry; an existing prim has none.
        self._spawn_focal_length: float | None = None if cfg.spawn is None else cfg.spawn.focal_length
        self._spawn_horizontal_aperture: float | None = None if cfg.spawn is None else cfg.spawn.horizontal_aperture
        self._calibrated_rotation = None
        super().__init__(cfg)

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _initialize_impl(self):
        # 1. Derive camera physical FOV for ray generation.
        self._camera_hfov_rad: float | None = None
        if self._spawn_focal_length is not None:
            self._camera_hfov_rad = 2.0 * math.atan(self._spawn_horizontal_aperture / (2.0 * self._spawn_focal_length))

        # 2. Build coords on CPU so we know N_lo (and N_hi when pooling) before
        #    the parent allocates buffers.  The pooler is constructed later,
        #    after the sensor device is available, because its knn indices /
        #    distances / gaussian weights are plain tensor attributes that
        #    nn.Module.to() does not move.
        self._sampling_coords, matched_res = build_lowres_coords(self.cfg, device="cpu")
        self._native_grid = self.cfg.foveated_style == GRID_STYLE
        self._sampling_coords_highres = build_highres_coords(
            self.cfg,
            matched_res,
            device="cpu",
        )
        self._num_valid_samples = len(self._sampling_coords)  # public / RGC count
        self._num_ray_samples = len(self._sampling_coords_highres or self._sampling_coords)
        logger.info(
            "FoveatedNewtonCamera: %d samples (%d rendered), style=%s, cmf_a=%.4g deg, field of view=%.1f deg",
            self._num_valid_samples,
            self._num_ray_samples,
            self.cfg.foveated_style,
            self._sampling_coords.cmf_a,
            self._sampling_coords.fov,
        )

        # 3. Size the parent's render buffers to the *rendered* ray count.
        #    When pooling is enabled this is N_hi; otherwise it equals N_lo.
        ray_coords = self._sampling_coords_highres or self._sampling_coords
        self.cfg.height, self.cfg.width = grid_output_shape(ray_coords)
        renderer = self.cfg.renderer_cfg
        if renderer.render_order == "tiled":
            tile_width = renderer.tile_rendering_width
            tile_height = renderer.tile_rendering_height
            if tile_width <= 0 or tile_height <= 0:
                raise ValueError("Newton render tile dimensions must be positive.")
            if not self._native_grid:
                # Pack consecutive vector rays into complete tiles. Padding is
                # private renderer storage and never enters masking or pooling.
                self.cfg.width = tile_width
                self.cfg.height = (
                    (self._num_ray_samples + tile_width * tile_height - 1) // (tile_width * tile_height) * tile_height
                )

        # 4. Run parent initialisation (creates renderer, buffers, etc.).
        super()._initialize_impl()
        if self._native_grid:
            self._validate_native_render_layout()

        # 5. Move coords to the correct device.
        self._sampling_coords = self._sampling_coords.to(device=self._device)
        if self._sampling_coords_highres is not None:
            self._sampling_coords_highres = self._sampling_coords_highres.to(device=self._device)
            ray_coords_device = self._sampling_coords_highres
        else:
            ray_coords_device = self._sampling_coords
        self._ray_coords = ray_coords_device
        self._initialize_calibrated_gaze()

        # 6. Build the pooler(s) on the sensor device.  A second max-mode
        #    pooler is materialised lazily below if any integer-valued outputs
        #    are present (segmentation class IDs) and pool_int_mode differs
        #    from pool_mode.
        self._pooler, self._pool_k = build_pooler(
            self.cfg,
            self._sampling_coords,
            self._sampling_coords_highres,
            device=str(self._device),
        )
        self._pooler_int = None

        # 7. Pre-compute foveated rays from the highres (or lowres) cartesian.
        # Apply optional image-space rotation by rotating the 2D cartesian
        # coordinates around the optical axis BEFORE deriving rays. This
        # keeps the per-sample index ordering identical to the canonical
        # ``SamplingCoords`` (so the agent / foveated scatter visualization
        # see upright samples) while undoing a physical mount rotation of
        # the camera prim. Foveated analogue of a post-render
        # ``torch.flip`` — but flip would shuffle the sample-to-coord
        # mapping; this rotation does not.
        ray_cart = ray_coords_device.cartesian
        if self._native_grid:
            # Spherical grids take their extent from the sampling coordinates and ignore the tangent scale.
            tangent_scale = 1.0 if self._camera_hfov_rad is None else math.tan(self._camera_hfov_rad / 2)
            self._ray_directions = native_grid_directions(
                ray_coords_device,
                tangent_scale=tangent_scale,
                image_rotation_deg=self.cfg.image_rotation_deg,
                camera_fov_rad=self._camera_hfov_rad,
            )
            self._foveated_rays = compute_explicit_ray_array(self._ray_directions, device=str(self._device))
        elif self.cfg.field_geometry == "spherical":
            self._ray_directions = retinal_directions(ray_cart, ray_coords_device.fov, self.cfg.image_rotation_deg)
            self._foveated_rays = compute_explicit_ray_array(self._ray_directions, device=str(self._device))
        elif self.cfg.image_rotation_deg:
            theta = math.radians(float(self.cfg.image_rotation_deg))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            rot = torch.tensor(
                [[cos_t, -sin_t], [sin_t, cos_t]],
                device=ray_cart.device,
                dtype=ray_cart.dtype,
            )
            ray_cart = ray_cart @ rot.T
            self._foveated_rays = compute_foveated_ray_array(
                sampling_coords_cartesian=ray_cart,
                camera_fov_rad=self._camera_hfov_rad,
                device=str(self._device),
            )
        else:
            self._foveated_rays = compute_foveated_ray_array(
                sampling_coords_cartesian=ray_cart,
                camera_fov_rad=self._camera_hfov_rad,
                device=str(self._device),
            )

        render_samples = self.cfg.height * self.cfg.width
        if render_samples > self._num_ray_samples:
            packed = wp.empty((render_samples, 2), dtype=wp.vec3f, device=self._device)
            wp.launch(
                _pack_vector_rays,
                dim=render_samples,
                inputs=[self._foveated_rays.reshape((self._num_ray_samples, 2)), packed],
                device=self._device,
            )
            self._foveated_rays = packed
        self._foveated_rays = self._foveated_rays.reshape((1, self.cfg.height, self.cfg.width, 2))
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
        if self._native_grid:
            self._grid_output = allocate_grid_outputs(source, self._sampling_coords)
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            source = self._grid_output
        # The buffers the camera publishes: stable for the camera's lifetime and refilled each frame.
        self._public_output = proxy_outputs(source)
        self._data.output.clear()
        self._data.output.update(self._public_output)

        # 6. Inject foveated rays into render_data immediately.
        self._render_data.camera_rays = self._foveated_rays
        logger.info(
            "FoveatedNewtonCamera: injected foveated rays shape=%s onto render_data",
            self._foveated_rays.shape,
        )

    def _validate_native_render_layout(self) -> None:
        """Reject image dimensions that cannot be covered by Newton's tiles."""
        renderer = self._renderer.cfg
        if renderer.render_order != "tiled":
            return
        tile_width = renderer.tile_rendering_width
        tile_height = renderer.tile_rendering_height
        if self.cfg.width % tile_width or self.cfg.height % tile_height:
            raise ValueError(
                f"Native-grid Newton render size {self.cfg.width}x{self.cfg.height} "
                f"must be divisible by renderer tile size {tile_width}x{tile_height}. "
                "Choose foveated_res and pool_res_mult so the rendered grid satisfies "
                "these dimensions, or select a non-tiled Newton render_order."
            )

    def _allocate_pooled_buffers(self) -> None:
        """Allocate output-grid tensors from renderer-owned high-resolution buffers."""
        if self._pooler is not None:
            # Stash the parent-allocated highres buffers; the renderer will keep
            # writing into these, and we pool into a separate lowres dict that
            # becomes the public _data.output after each update.
            self._highres_output: dict[str, ProxyArray] = dict(self._data.output)
            self._pooled_output: dict[str, torch.Tensor] = {}
            needs_int_pooler = False
            for name, tex in self._highres_output.items():
                tensor = tex.torch
                B, _, _, C = tensor.shape
                self._pooled_output[name] = torch.zeros(
                    (B, self._num_valid_samples, 1, C),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                if is_segmentation_output(name) and self.cfg.pool_int_mode != self.cfg.pool_mode:
                    needs_int_pooler = True
            if "rgba" in self._pooled_output and "rgb" in self.cfg.data_types:
                # Re-alias rgb onto the first 3 channels of pooled rgba.
                self._pooled_output["rgb"] = self._pooled_output["rgba"][..., :3]
            if needs_int_pooler:
                self._pooler_int, _ = build_pooler(
                    self.cfg,
                    self._sampling_coords,
                    self._sampling_coords_highres,
                    device=str(self._device),
                    mode=self.cfg.pool_int_mode,
                )
            dtypes_summary = {n: str(t.dtype).removeprefix("torch.") for n, t in self._highres_output.items()}
            logger.info(
                "FoveatedNewtonCamera: KNN pooling enabled "
                "(mode=%s k=%s sample_cortex=%s N_hi=%d -> N_lo=%d) "
                "outputs=%s int_pooler=%s int_mode=%s",
                self.cfg.pool_mode,
                self._pool_k,
                self.cfg.pool_sample_cortex,
                self._num_ray_samples,
                self._num_valid_samples,
                dtypes_summary,
                self._pooler_int is not None,
                self.cfg.pool_int_mode,
            )

    # ------------------------------------------------------------------
    #  Per-frame overrides
    # ------------------------------------------------------------------

    def _update_poses(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        env_mask: wp.array | None = None,
        frame_op: int = 0,
    ) -> None:
        """Read base poses, apply the calibrated gaze, update renderer, inject rays.

        The scene pose is read even without a gaze: the camera may inherit
        motion from a parent body.
        """
        # 1. Read base camera poses, then rotate them by the calibrated gaze if one is set.
        Camera._update_poses(self, env_ids, env_mask=env_mask, frame_op=frame_op)
        self._apply_calibrated_pose(env_ids, env_mask)

        # 2. Update camera_transforms
        # The renderer maintains persistent transform buffers and updates
        # them in place after the pose changes.
        self._renderer.update_camera(
            self._render_data,
            self._data.pos_w,
            self._data.quat_w_world,
            self._data.intrinsic_matrices,
        )

        # 3. Ensure foveated rays (not pinhole)
        if hasattr(self, "_foveated_rays"):
            self._render_data.camera_rays = self._foveated_rays

    def _update_buffers_impl(self, env_mask: wp.array):
        """Render, then (optionally) pool highres -> lowres foveated samples."""
        self._data.output.clear()
        self._data.output.update(self._render_output)
        if self._pooler is None:
            super()._update_buffers_impl(env_mask)
            output = self._flat_render_output
            mask_grid_samples(output, self._ray_coords)
            self._apply_sensor_fov_mask(output)
            if self._native_grid:
                copy_grid_outputs(output, self._grid_output, self._grid_indices)
            self._data.output.clear()
            self._data.output.update(self._public_output)
            return

        # Ensure the renderer writes into the highres buffers, not the pooled ones.
        self._data.output.clear()
        self._data.output.update(self._render_output)
        super()._update_buffers_impl(env_mask)
        mask_grid_samples(self._highres_output, self._ray_coords)
        self._apply_sensor_fov_mask(self._highres_output)

        # Pool each output channel (B, N_hi, 1, C) -> (B, N_lo, 1, C).
        # Cast to float for pooling (KNN ops need real-valued NaN semantics),
        # then restore the original dtype.  Integer outputs (segmentation
        # class IDs) use the separate max-mode pooler so pooled values remain
        # valid class IDs rather than averaged floats.
        for name, hi in self._highres_output.items():
            if name == "rgb":
                continue  # rgba alias, refreshed below
            lo = self._pooled_output[name]
            use_int_pool = self._pooler_int is not None and is_segmentation_output(name)
            pooler = self._pooler_int if use_int_pool else self._pooler
            pooled = pool_hwc_buffer(pooler, hi.torch.float())
            lo.copy_(pooled.to(lo.dtype))

        # Expose pooled tensors as the public sensor output.
        if self._native_grid:
            copy_grid_outputs(self._pooled_output, self._grid_output, self._grid_indices)
        self._data.output.clear()
        self._data.output.update(self._public_output)

    def _apply_sensor_fov_mask(self, outputs: dict[str, torch.Tensor]) -> None:
        """Zero the samples that fall outside the calibrated camera at the current gaze."""
        if not self.cfg.mask_to_sensor_fov:
            return
        first_output = next(iter(outputs.values()))
        first_tensor = first_output.torch if hasattr(first_output, "torch") else first_output
        valid = self._calibrated_validity(first_tensor.shape[0])
        invalid = ~valid[:, :, None, None]
        has_rgba_alias = "rgba" in outputs
        for name, output in outputs.items():
            if name == "rgb" and has_rgba_alias:
                continue
            tensor = output.torch if hasattr(output, "torch") else output
            tensor.masked_fill_(invalid, 0)
        self._sensor_fov_valid_fraction = valid.float().mean(dim=1)

    # ------------------------------------------------------------------
    #  Output helpers
    # ------------------------------------------------------------------

    @property
    def hfov_rad(self) -> float:
        """Horizontal field of view in radians."""
        return self._camera_hfov_rad

    @property
    def sampling_coords(self):
        """The :class:`SamplingCoords` used for ray generation."""
        return self._sampling_coords

    @property
    def image_rotation_deg(self) -> float:
        """Pre-render rotation (degrees, CCW around the optical axis) applied
        to the per-ray sampling coordinates. ``0`` for the canonical mount."""
        return float(self.cfg.image_rotation_deg)

    @property
    def num_valid_samples(self) -> int:
        """Number of foveated sample points *N*."""
        return self._num_valid_samples

    @property
    def sensor_fov_valid_fraction(self) -> torch.Tensor | None:
        """Per-environment fraction of samples inside the calibrated camera, or None before the first masked frame."""
        return getattr(self, "_sensor_fov_valid_fraction", None)
