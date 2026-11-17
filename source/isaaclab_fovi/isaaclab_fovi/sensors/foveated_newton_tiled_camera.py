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
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab.utils.warp import ProxyArray

from .calibrated_geometry import convert_camera_frame_quaternion, retinal_directions
from .fixed_software_fov import (
    FIXED_FOV_ANGLE_MODELS,
    fixed_frustum_validity,
    retinal_ray_directions,
    validate_fixed_software_fov,
)
from .foveated_rays import compute_explicit_ray_array, compute_foveated_ray_array
from .native_grid import (
    GRID_STYLE,
    allocate_grid_outputs,
    copy_grid_outputs,
    grid_output_shape,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
)
from .pooling import (
    build_highres_coords,
    build_lowres_coords,
    build_pooler,
    is_segmentation_output,
    pool_hwc_buffer,
)
from .virtual_gaze import CalibratedVirtualGaze

if TYPE_CHECKING:
    from .foveated_tiled_camera_cfg import FoveatedTiledCameraCfg

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


class FoveatedNewtonTiledCamera(CalibratedVirtualGaze, Camera):
    """Newton tiled camera that renders with foveated (CMF-distributed) rays.

    Instead of a uniform pixel grid, this sensor shoots *N* rays whose
    directions are derived from :class:`fovi.sensing.coords.SamplingCoords`
    and injected into Newton's Warp renderer. Vector sensors publish
    ``(B, N, 1, C)``; native grid sensors publish upright ``(B, H, W, C)``.

    The camera's physical FOV (from the ``spawn`` optics) is used as the
    ``SamplingCoords.fov`` parameter so that the foveated sampling covers the
    full scene.  Fixation is applied as a per-environment camera rotation.
    """

    cfg: FoveatedTiledCameraCfg

    def __init__(self, cfg: FoveatedTiledCameraCfg):
        if cfg.field_geometry == "spherical":
            cfg.update_latest_camera_pose = True
        if cfg.camera_fov_deg is not None:
            # Back-compute spawn optics from desired FOV for USD prim compat.
            hfov_rad = math.radians(cfg.camera_fov_deg)
            aperture = cfg.spawn.horizontal_aperture  # keep default aperture
            cfg.spawn.focal_length = aperture / (2.0 * math.tan(hfov_rad / 2.0))
        self._spawn_focal_length: float = cfg.spawn.focal_length
        self._spawn_horizontal_aperture: float = cfg.spawn.horizontal_aperture
        self._fixation: torch.Tensor | None = None
        self._fixed_fov_fixation: torch.Tensor | None = None
        self._fixed_fov_angle_model = "camera_xyz"
        self._calibrated_rotation = None
        super().__init__(cfg)

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _initialize_impl(self):
        # 1. Derive camera physical FOV for ray generation.
        self._camera_hfov_rad = 2.0 * math.atan(self._spawn_horizontal_aperture / (2.0 * self._spawn_focal_length))

        # 2. Build coords on CPU so we know N_lo (and N_hi when pooling) before
        #    the parent allocates buffers.  The pooler is constructed later,
        #    after the sensor device is available, because its knn indices /
        #    distances / gaussian weights are plain tensor attributes that
        #    nn.Module.to() does not move.
        # cfg.cmf_a is already cmf_a_frac after resolve_cmf_a_frac runs
        self._sampling_coords, matched_res = build_lowres_coords(self.cfg, device="cpu")
        self._native_grid = self.cfg.foveated_style == GRID_STYLE
        self._sampling_coords_highres = build_highres_coords(
            self.cfg,
            matched_res,
            device="cpu",
        )
        self._num_valid_samples = len(self._sampling_coords)  # public / RGC count
        self._num_ray_samples = len(self._sampling_coords_highres or self._sampling_coords)
        self._camera_hfov_deg = math.degrees(self._camera_hfov_rad)
        self._fixed_fov_focal_length_px: float | None = None
        self._fixed_fov_image_height: int | None = None
        self._fixed_fov_image_width: int | None = None
        if self.cfg.fixed_software_fov:
            if self.cfg.projection != "pinhole" and self.cfg.field_geometry != "spherical":
                raise ValueError(
                    f"fixed_software_fov currently requires projection='pinhole', got {self.cfg.projection!r}."
                )
            image_height = self.cfg.software_fov_image_height
            image_width = self.cfg.software_fov_image_width
            if image_height is None or image_width is None:
                raise ValueError(
                    "fixed_software_fov=True requires software_fov_image_height and software_fov_image_width."
                )
            focal_length_px = image_width * self._spawn_focal_length / self._spawn_horizontal_aperture
            validate_fixed_software_fov(
                image_height=image_height,
                image_width=image_width,
                focal_length_px=focal_length_px,
                principal_point=(image_width * 0.5, image_height * 0.5),
                retinal_window_short_side_fraction=(self.cfg.retinal_window_short_side_fraction),
                retinal_window_reference_side=self.cfg.retinal_window_reference_side,
            )
            self._fixed_fov_focal_length_px = focal_length_px
            self._fixed_fov_image_height = image_height
            self._fixed_fov_image_width = image_width
        pool_line = (
            f"    Pool:       mode={self.cfg.pool_mode} "
            f"N_hi={self._num_ray_samples} -> N_lo={self._num_valid_samples}\n"
            if self.cfg.pool_enabled
            else ""
        )
        print(
            f"\n{'=' * 60}\n"
            f"  FoveatedNewtonTiledCamera\n"
            f"    HFOV:       {self._camera_hfov_deg:.1f}°\n"
            f"    Projection: {self.cfg.projection}\n"
            f"    Samples:    N={self._num_valid_samples}\n"
            f"    cmf_a_frac: {self.cfg.cmf_a}\n"
            f"    Style:      {self.cfg.foveated_style}\n"
            f"{pool_line}"
            f"{'=' * 60}\n"
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
            tangent_scale = math.tan(self._camera_hfov_rad / 2)
            if self.cfg.fixed_software_fov:
                reference = (
                    max(self._fixed_fov_image_height, self._fixed_fov_image_width)
                    if self.cfg.retinal_window_reference_side == "long"
                    else min(self._fixed_fov_image_height, self._fixed_fov_image_width)
                )
                tangent_scale = (
                    reference * self.cfg.retinal_window_short_side_fraction / (2 * self._fixed_fov_focal_length_px)
                )
            self._fixed_fov_ray_directions = native_grid_directions(
                ray_coords_device,
                tangent_scale=tangent_scale,
                image_rotation_deg=self.cfg.image_rotation_deg,
                projection=self.cfg.projection,
                camera_fov_rad=self._camera_hfov_rad,
            )
            self._foveated_rays = compute_explicit_ray_array(self._fixed_fov_ray_directions, device=str(self._device))
        elif self.cfg.field_geometry == "spherical":
            self._fixed_fov_ray_directions = retinal_directions(
                ray_cart, self._sampling_coords.fov, self.cfg.image_rotation_deg
            )
            self._foveated_rays = compute_explicit_ray_array(self._fixed_fov_ray_directions, device=str(self._device))
        elif self.cfg.fixed_software_fov:
            self._fixed_fov_ray_directions = retinal_ray_directions(
                ray_cart,
                image_height=self._fixed_fov_image_height,
                image_width=self._fixed_fov_image_width,
                focal_length_px=self._fixed_fov_focal_length_px,
                retinal_window_short_side_fraction=(self.cfg.retinal_window_short_side_fraction),
                retinal_window_reference_side=self.cfg.retinal_window_reference_side,
                image_rotation_deg=self.cfg.image_rotation_deg,
            )
            self._foveated_rays = compute_explicit_ray_array(
                self._fixed_fov_ray_directions,
                device=str(self._device),
            )
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
                projection=self.cfg.projection,
            )
        else:
            self._foveated_rays = compute_foveated_ray_array(
                sampling_coords_cartesian=ray_cart,
                camera_fov_rad=self._camera_hfov_rad,
                device=str(self._device),
                projection=self.cfg.projection,
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
        if self._native_grid:
            source = self._pooled_output if self._pooler is not None else self._flat_render_output
            self._grid_output = allocate_grid_outputs(source, self._sampling_coords)
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            self._data.output.clear()
            self._data.output.update(self._grid_output)

        # 6. Inject foveated rays into render_data immediately.
        self._render_data.camera_rays = self._foveated_rays
        logger.info(
            "FoveatedNewtonTiledCamera: injected foveated rays shape=%s onto render_data",
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
            print(
                f"[FoveatedNewtonTiledCamera] pool outputs={dtypes_summary} "
                f"pool_mode={self.cfg.pool_mode!r} int_mode={self.cfg.pool_int_mode!r} "
                f"int_pooler_built={self._pooler_int is not None}"
            )
            logger.info(
                "FoveatedNewtonTiledCamera: KNN pooling enabled "
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
    #  Fixation
    # ------------------------------------------------------------------

    def set_fixation(self, fixation_angles: torch.Tensor):
        """Set per-environment fixation as camera rotation angles.

        Args:
            fixation_angles: ``(B, 2)`` tensor in radians.
                ``[:, 0]`` = θ_x (X-axis rotation, positive = look up),
                ``[:, 1]`` = θ_y (Y-axis rotation, positive = look left).
                ``(0, 0)`` = center (no rotation).
        """
        self._fixation = fixation_angles

    def set_fixed_fov_fixation(
        self,
        fixation_angles: torch.Tensor,
        *,
        angle_model: str,
    ) -> None:
        """Update fixed-frustum masking without rotating the camera twice."""
        if not self.cfg.fixed_software_fov:
            raise RuntimeError("set_fixed_fov_fixation requires fixed_software_fov=True.")
        if angle_model not in FIXED_FOV_ANGLE_MODELS:
            raise ValueError(f"angle_model must be one of {FIXED_FOV_ANGLE_MODELS}, got {angle_model!r}.")
        self._fixed_fov_fixation = fixation_angles
        self._fixed_fov_angle_model = angle_model

    def _apply_fixation_rotation(self):
        """Rotate camera orientations based on the current fixation angles.

        Composes fixation rotation (in angle-space) with the base camera
        orientations in the OpenGL frame.
        """
        fix = self._fixation.float()  # (B, 2): [θ_x, θ_y] in radians
        look_roll = fix[:, 0]  # X-axis rotation (directly in radians)
        look_pitch = fix[:, 1]  # Y-axis rotation (directly in radians)
        zeros = torch.zeros_like(look_roll)

        fix_quat = quat_from_euler_xyz(look_roll, look_pitch, zeros)  # (B, 4)

        # Apply rotation in OpenGL camera frame:
        # 1. Convert stored world-convention quaternions to OpenGL
        quat_world = self._quat_w_world_torch()
        quat_gl = convert_camera_frame_quaternion(quat_world, origin="world", target="opengl")
        # 2. Compose: base rotation * fixation rotation (local frame)
        quat_gl_rotated = quat_mul(quat_gl, fix_quat)
        # 3. Convert back to world convention for storage
        quat_world[:] = convert_camera_frame_quaternion(quat_gl_rotated, origin="opengl", target="world")

    def _quat_w_world_torch(self) -> torch.Tensor:
        """Return the camera world quaternion as a Torch tensor view."""
        quat_world = self._data.quat_w_world
        return quat_world.torch if hasattr(quat_world, "torch") else quat_world

    # ------------------------------------------------------------------
    #  Per-frame overrides
    # ------------------------------------------------------------------

    def _update_poses(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        env_mask: wp.array | None = None,
        frame_op: int = 0,
    ) -> None:
        """Read base poses, apply fixation rotation, update renderer, inject rays.

        Read the scene pose even without a sensor-level fixation: the camera
        may inherit motion from a parent articulation.
        """
        # 1. Read base camera poses + apply fixation (only when fixation active)
        Camera._update_poses(self, env_ids, env_mask=env_mask, frame_op=frame_op)
        if self._fixation is not None:
            self._apply_fixation_rotation()

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
            self._apply_fixed_fov_mask(output)
            if self._native_grid:
                copy_grid_outputs(output, self._grid_output, self._grid_indices)
            self._data.output.clear()
            self._data.output.update(self._grid_output if self._native_grid else output)
            return

        # Ensure the renderer writes into the highres buffers, not the pooled ones.
        self._data.output.clear()
        self._data.output.update(self._render_output)
        super()._update_buffers_impl(env_mask)
        mask_grid_samples(self._highres_output, self._ray_coords)
        self._apply_fixed_fov_mask(self._highres_output)

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
        if "rgba" in self._pooled_output and "rgb" in self.cfg.data_types:
            self._pooled_output["rgb"] = self._pooled_output["rgba"][..., :3]

        # Expose pooled tensors as the public sensor output.
        self._data.output.clear()
        if self._native_grid:
            copy_grid_outputs(self._pooled_output, self._grid_output, self._grid_indices)
            self._data.output.update(self._grid_output)
        else:
            self._data.output.update(self._pooled_output)

    def _apply_fixed_fov_mask(self, outputs: dict[str, torch.Tensor]) -> None:
        if not self.cfg.fixed_software_fov:
            return
        first_output = next(iter(outputs.values()))
        first_tensor = first_output.torch if hasattr(first_output, "torch") else first_output
        batch_size = first_tensor.shape[0]
        fixation = self._fixed_fov_fixation
        if fixation is None:
            fixation = self._fixation
        if fixation is None:
            fixation = torch.zeros(
                batch_size,
                2,
                device=self._fixed_fov_ray_directions.device,
                dtype=self._fixed_fov_ray_directions.dtype,
            )
        if self.cfg.field_geometry == "spherical":
            valid = self._calibrated_validity(batch_size)
        else:
            valid = fixed_frustum_validity(
                self._fixed_fov_ray_directions,
                fixation,
                image_height=self._fixed_fov_image_height,
                image_width=self._fixed_fov_image_width,
                focal_length_px=self._fixed_fov_focal_length_px,
                principal_point=(
                    self._fixed_fov_image_width * 0.5,
                    self._fixed_fov_image_height * 0.5,
                ),
                angle_model=self._fixed_fov_angle_model,
                image_rotation_deg=self.cfg.image_rotation_deg,
            )
        invalid = ~valid[:, :, None, None]
        has_rgba_alias = "rgba" in outputs
        for name, output in outputs.items():
            if name == "rgb" and has_rgba_alias:
                continue
            tensor = output.torch if hasattr(output, "torch") else output
            tensor.masked_fill_(invalid, 0)
        self._fixed_fov_valid_fraction = valid.float().mean(dim=1)

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
    def fixed_fov_valid_fraction(self) -> torch.Tensor | None:
        """Per-environment fraction of rays inside the fixed dense frustum."""
        return getattr(self, "_fixed_fov_valid_fraction", None)
