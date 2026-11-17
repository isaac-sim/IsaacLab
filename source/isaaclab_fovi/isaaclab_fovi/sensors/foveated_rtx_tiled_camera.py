# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX foveated tiled camera sensor.

Subclasses :class:`isaaclab.sensors.Camera` to render foveated images
via the RTX path tracer using ``OmniLensDistortionLutAPI`` textures.
Per-sample values are extracted from the rendered 2D texture to produce
vector ``(B, N, 1, C)`` or upright native-grid ``(B, H, W, C)`` output buffers,
identical to :class:`FoveatedNewtonTiledCamera`.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab import sim as sim_utils
from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from . import generalized_projection as gp
from .calibrated_geometry import compose_rotations, retinal_directions
from .fixed_software_fov import (
    FIXED_FOV_ANGLE_MODELS,
    fixed_frustum_validity,
    retinal_ray_directions,
    validate_fixed_software_fov,
)
from .native_grid import (
    GRID_STYLE,
    allocate_grid_outputs,
    copy_grid_outputs,
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
    from isaaclab.renderers.base_renderer import BaseRenderer

    from .foveated_tiled_camera_cfg import FoveatedTiledCameraCfg

logger = logging.getLogger(__name__)


class FoveatedRTXTiledCamera(CalibratedVirtualGaze, Camera):
    """RTX tiled camera that renders with foveated (CMF-distributed) rays.

    Instead of Newton's Warp ray arrays, this sensor creates
    ``OmniLensDistortionLutAPI`` Camera prims with LUT textures that define
    per-texel ray directions.  The RTX path tracer renders a 2D texture
    which is then sampled at pre-computed UV positions to extract per-sample
    values, producing ``(B, N, 1, C)`` output buffers.

    The cortical disc layout (default) ensures spatial coherence for DLSS
    denoising. Camera motion comes from its parent articulation, using
    a physical or idealized actuator.
    """

    cfg: FoveatedTiledCameraCfg

    @property
    def supports_batch_update(self) -> bool:
        return True

    @staticmethod
    def _update_buffers_batch_impl(sensors: Sequence[FoveatedRTXTiledCamera]) -> None:
        Camera._update_buffers_batch_impl(sensors)
        if sim_utils.SimulationContext.instance() is None:
            return  # The base implementation already invoked scalar capture.
        for sensor in sensors:
            sensor.process_rendered_frame()

    @property
    def renderer(self) -> BaseRenderer:
        """Renderer shared with task-owned visual animation bindings."""
        return self._renderer

    def __init__(self, cfg: FoveatedTiledCameraCfg):
        if cfg.field_geometry == "spherical":
            cfg.update_latest_camera_pose = True
        if cfg.camera_fov_deg is not None:
            hfov_rad = math.radians(cfg.camera_fov_deg)
            aperture = cfg.spawn.horizontal_aperture
            cfg.spawn.focal_length = aperture / (2.0 * math.tan(hfov_rad / 2.0))
        self._spawn_focal_length: float = cfg.spawn.focal_length
        self._spawn_horizontal_aperture: float = cfg.spawn.horizontal_aperture
        self._fixed_fov_fixation: torch.Tensor | None = None
        self._fixed_fov_angle_model = "pan_tilt"
        self._calibrated_rotation = None
        # Texture buffers are owned by the parent ``Camera`` (allocated in
        # ``_create_buffers`` and bound to the renderer via ``set_outputs``).
        # Foveated / pooled buffers are allocated in ``_initialize_impl``
        # once the texture buffers are known.
        self._texture_output: dict[str, torch.Tensor] = {}
        self._foveated_output: dict[str, torch.Tensor] = {}
        self._pooled_output: dict[str, torch.Tensor] | None = None
        if cfg.renderer_cfg.renderer_type == "ovrtx":
            self.cfg = cfg
            self._prepare_projection()
        super().__init__(cfg)
        if cfg.renderer_cfg.renderer_type == "ovrtx":
            for prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage):
                gp.apply_generalized_projection_to_prim(prim, self._dir_tex_path, self._exit_tex_path)

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _prepare_projection(self) -> None:
        # 1. Derive camera physical FOV.
        self._camera_hfov_rad = 2.0 * math.atan(self._spawn_horizontal_aperture / (2.0 * self._spawn_focal_length))
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

        # 2. Build coords on CPU.  The LUT textures + UV extraction use the
        #    *rendered* grid (highres when pooling, else lowres).  The
        #    public `sampling_coords` property stays on the lowres / RGC grid.
        self._sampling_coords, matched_res = build_lowres_coords(self.cfg, device="cpu")
        self._sampling_coords_highres = build_highres_coords(
            self.cfg,
            matched_res,
            device="cpu",
        )
        ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._ray_coords = ray_coords
        self._native_grid = self.cfg.foveated_style == GRID_STYLE

        self._num_valid_samples = len(self._sampling_coords)  # public / RGC count
        self._num_ray_samples = len(ray_coords)  # rendered ray count
        N = self._num_ray_samples

        # 3. Compute ray directions and UV mapping from the rendered grid.
        # Apply optional image-space rotation (``cfg.image_rotation_deg``) by
        # rotating the per-ray polar angle ``theta`` around the optical axis
        # *before* computing the ray directions. This undoes a physical
        # mount rotation of the camera prim while keeping the per-sample
        # index ordering identical to the canonical ``SamplingCoords`` — so
        # the agent's foveator (and the foveated scatter plot) read out the
        # same upright scene at every sample position. A post-render
        # ``torch.flip`` would reshuffle the sample order and corrupt the
        # downstream visualization / KNN mapping; this rotation is the
        # foveated-camera-native analogue of that flip.
        theta_rad = ray_coords.polar[:, 1].to(torch.float64)
        if self.cfg.image_rotation_deg:
            theta_rad = theta_rad + math.radians(float(self.cfg.image_rotation_deg))
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
            self._native_tangent_scale = tangent_scale
            dirs = native_grid_directions(
                ray_coords,
                tangent_scale=tangent_scale,
                image_rotation_deg=self.cfg.image_rotation_deg,
                projection=self.cfg.projection,
                camera_fov_rad=self._camera_hfov_rad,
            )
            self._fixed_fov_ray_directions_cpu = dirs
        elif self.cfg.field_geometry == "spherical":
            dirs = retinal_directions(ray_coords.cartesian, ray_coords.fov, self.cfg.image_rotation_deg)
            self._fixed_fov_ray_directions_cpu = dirs
        elif self.cfg.fixed_software_fov:
            dirs = retinal_ray_directions(
                ray_coords.cartesian,
                image_height=self._fixed_fov_image_height,
                image_width=self._fixed_fov_image_width,
                focal_length_px=self._fixed_fov_focal_length_px,
                retinal_window_short_side_fraction=(self.cfg.retinal_window_short_side_fraction),
                retinal_window_reference_side=self.cfg.retinal_window_reference_side,
                image_rotation_deg=self.cfg.image_rotation_deg,
            )
            self._fixed_fov_ray_directions_cpu = dirs
        else:
            r_deg = ray_coords.polar[:, 0].to(torch.float64) * (self._camera_hfov_deg / 2.0)
            dirs = gp.ray_directions_rectilinear(r_deg, theta_rad, device="cpu")

        # 4. Texture dimensions and UV layout.
        #    UV mapping uses fov=1 normalized space (matching the profiling
        #    script and SamplingCoords convention).  Ray directions use
        #    physical degrees for correct pinhole projection.
        if self.cfg.ndc_layout == "native_grid":
            tw = th = ray_coords.resolution
            axis = (torch.arange(tw, dtype=torch.float32) + 0.5) / tw
            row, col = torch.meshgrid(axis, axis, indexing="ij")
            uv = torch.stack((col, row), -1).reshape(-1, 2)
        elif self.cfg.ndc_layout == "cortical_disc":
            tw = th = max(64, math.ceil(math.sqrt(4 * N / math.pi)))
            r_norm = ray_coords.polar[:, 0].to(torch.float64) * 0.5
            uv = gp.samples_uv_cortical_disc(
                r_norm,
                theta_rad,
                1.0,
                self.cfg.cmf_a,
                device="cpu",
            )
        else:
            raise ValueError(f"Unknown ndc_layout: {self.cfg.ndc_layout!r}")

        self._texture_width = tw
        self._texture_height = th
        self._sample_uv = uv  # (N, 2) float32

        # 5. Build and save LUT textures.
        #    The exit texture is an octahedral inverse map (ray direction → sensor NDC).
        #    It covers the full hemisphere, but only the fraction within the camera
        #    FOV is populated.  The resolution must be high enough that each foveated
        #    sample maps to a distinct exit texel within that fraction.  Scaling as
        #    4× the direction texture dimension gives ~16× the texel count, which
        #    provides sufficient angular oversampling for the NN lookup.
        if self.cfg.exit_texture_size is not None:
            EW = EH = self.cfg.exit_texture_size
        else:
            EW = EH = max(128, 4 * max(tw, th))
        if self._native_grid:
            dir_tex, exit_tex = gp.build_native_grid_textures(
                ray_coords,
                tangent_scale=self._native_tangent_scale,
                image_rotation_deg=self.cfg.image_rotation_deg,
                exit_size=EW,
                projection=self.cfg.projection,
                camera_fov_rad=self._camera_hfov_rad,
            )
        else:
            dir_tex, exit_tex = gp.build_generalized_projection_textures(
                uv,
                dirs,
                tw,
                th,
                EW,
                EH,
                device="cpu",
            )
        self._lut_tmpdir = tempfile.mkdtemp(prefix="rtx_foveated_luts_")
        self._dir_tex_path, self._exit_tex_path = gp.save_generalized_projection_textures(
            dir_tex,
            exit_tex,
            self._lut_tmpdir,
            "foveated",
        )

        pool_line = (
            f"    Pool:       mode={self.cfg.pool_mode} "
            f"N_hi={self._num_ray_samples} -> N_lo={self._num_valid_samples}\n"
            if self.cfg.pool_enabled
            else ""
        )
        print(
            f"\n{'=' * 60}\n"
            f"  FoveatedRTXTiledCamera\n"
            f"    HFOV:       {self._camera_hfov_deg:.1f}°\n"
            f"    Samples:    N={self._num_valid_samples}\n"
            f"    Texture:    {tw}x{th}\n"
            f"    Exit LUT:   {EW}x{EH}\n"
            f"    Layout:     {self.cfg.ndc_layout}\n"
            f"    cmf_a_frac: {self.cfg.cmf_a}\n"
            f"    Style:      {self.cfg.foveated_style}\n"
            f"{pool_line}"
            f"    LUT dir:    {self._lut_tmpdir}\n"
            f"{'=' * 60}\n"
        )

        # 6. Set texture dimensions so parent allocates (B, th, tw, C) buffers.
        self.cfg.height = th
        self.cfg.width = tw

    def _initialize_impl(self) -> None:
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            self._prepare_projection()
        tw, th = self._texture_width, self._texture_height
        # Author the lens before Camera prepares the renderer's stage. OVRTX
        # snapshots a detached stage, so later USD edits would miss its cameras.
        for prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage):
            gp.apply_generalized_projection_to_prim(
                prim,
                direction_texture_path=self._dir_tex_path,
                exit_texture_path=self._exit_tex_path,
            )

        # 8. Run parent initialisation (creates renderer, buffers, etc.).
        super()._initialize_impl()

        # OVRTX owns its runtime; Kit extensions belong only to Isaac RTX.
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            try:
                import omni.kit.app

                ext_manager = omni.kit.app.get_app().get_extension_manager()
                for ext_name in ("omni.sensors.nv.common", "omni.sensors.nv.camera"):
                    try:
                        ext_manager.set_extension_enabled_immediate(ext_name, True)
                    except Exception:
                        pass
            except ImportError:
                pass

        # Moving-gaze DLSS quality still needs validation with native LUTs;
        # geometry smoke tests disable temporal accumulation.

        # 10. Pre-compute extraction indices and build the pooler.  The
        #     parent ``Camera._create_buffers`` (called from super
        #     ``_initialize_impl``) has already allocated the per-data-type
        #     ``(B, H, W, C)`` texture tensors as ``self._data.output`` and
        #     bound them to the renderer via ``set_outputs``.  We snapshot
        #     them as ``_texture_output`` (same dict reference, so the
        #     renderer keeps writing to it across frames) and allocate the
        #     foveated / pooled output buffers from those shapes.  Then we
        #     point ``self._data.output`` at the foveated/pooled dict so
        #     downstream consumers see the per-sample buffers.
        self._sampling_coords = self._sampling_coords.to(device=self._device)
        if self.cfg.fixed_software_fov:
            self._fixed_fov_ray_directions = self._fixed_fov_ray_directions_cpu.to(self._device)
        if self._sampling_coords_highres is not None:
            self._sampling_coords_highres = self._sampling_coords_highres.to(device=self._device)
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._initialize_calibrated_gaze()
        if self.cfg.field_geometry == "spherical":
            _, local_orientation = self._view.get_local_poses()
            self._calibrated_base_local_orientation = matrix_from_quat(local_orientation.torch).clone()

        # Build the pooler on the sensor device (so knn tensors live there).
        # A second max-mode pooler is built inside ``_allocate_foveated_buffers``
        # if any integer-valued output buffers are present (segmentation IDs).
        self._pooler, self._pool_k = build_pooler(
            self.cfg,
            self._sampling_coords,
            self._sampling_coords_highres,
            device=str(self._device),
        )
        self._pooler_int = None

        uv_gpu = self._sample_uv.to(self._device)
        self._extract_px = (uv_gpu[:, 0] * tw).long().clamp(0, tw - 1)
        self._extract_py = (uv_gpu[:, 1] * th).long().clamp(0, th - 1)

        # Capture the renderer-bound texture dict and allocate foveated /
        # pooled output buffers.
        self._texture_output = self._data.output
        self._allocate_foveated_buffers()
        # Expose foveated (or pooled) buffers as the public output.
        # IsaacLab 3.0 made ``CameraData.output`` a read-only property; set
        # the private ``_output`` backing field directly. Downstream
        # readers (``CameraNode._read_rgb_raw`` etc.) just dict-index
        # ``camera.data.output[name]`` and pass the value through
        # ``_camera_output_to_torch``, which handles plain torch tensors
        # natively — the renderer still writes to the original
        # ProxyArray-backed buffers via the ``set_outputs`` binding done
        # by the parent ``_create_buffers``.
        new_output = self._pooled_output if self._pooled_output is not None else self._foveated_output
        if self._native_grid:
            self._grid_output = allocate_grid_outputs(new_output, self._sampling_coords)
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            new_output = self._grid_output
        try:
            self._data._output = new_output
        except AttributeError:
            # Older IsaacLab where ``output`` was a writable attribute.
            self._data.output = new_output

        logger.info(
            "FoveatedRTXTiledCamera: %d samples, %dx%d texture, layout=%s",
            self._num_ray_samples,
            tw,
            th,
            self.cfg.ndc_layout,
        )

    def _allocate_foveated_buffers(self):
        """Allocate ``(B, N_hi, 1, C)`` (and optionally ``(B, N_lo, 1, C)``)
        per-data-type buffers from the captured ``_texture_output`` shapes.

        Called once from :meth:`_initialize_impl` after the parent ``Camera``
        has allocated the texture buffers.
        """
        N = self._num_ray_samples
        has_rgba_alias = "rgba" in self._texture_output
        self._foveated_output = {}
        for name, buf in self._texture_output.items():
            if name == "rgb" and has_rgba_alias:
                continue  # handled via rgba alias below
            # IsaacLab 3.0: ``_data.output[name]`` is a ``ProxyArray``
            # wrapping a Warp array, not a torch tensor. ``buf.dtype`` is
            # a Warp dtype (a Python class), incompatible with
            # ``torch.zeros(dtype=...)``. Go through ``.torch`` to get
            # a torch view whose dtype / device are torch-native.
            buf_t = buf.torch if hasattr(buf, "torch") else buf
            C = buf_t.shape[-1]
            self._foveated_output[name] = torch.zeros(
                (self._view.count, N, 1, C),
                device=buf_t.device,
                dtype=buf_t.dtype,
            ).contiguous()
        if "rgba" in self._foveated_output and "rgb" in self.cfg.data_types:
            self._foveated_output["rgb"] = self._foveated_output["rgba"][:, :, :, :3]

        if self._pooler is not None:
            self._pooled_output = {}
            needs_int_pooler = False
            for name, buf in self._foveated_output.items():
                if name == "rgb" and has_rgba_alias:
                    continue
                self._pooled_output[name] = torch.zeros(
                    (self._view.count, self._num_valid_samples, 1, buf.shape[-1]),
                    device=self._device,
                    dtype=buf.dtype,
                ).contiguous()
                if is_segmentation_output(name) and self.cfg.pool_int_mode != self.cfg.pool_mode:
                    needs_int_pooler = True
            if "rgba" in self._pooled_output and "rgb" in self.cfg.data_types:
                self._pooled_output["rgb"] = self._pooled_output["rgba"][:, :, :, :3]
            if needs_int_pooler:
                self._pooler_int, _ = build_pooler(
                    self.cfg,
                    self._sampling_coords,
                    self._sampling_coords_highres,
                    device=str(self._device),
                    mode=self.cfg.pool_int_mode,
                )
            logger.info(
                "FoveatedRTXTiledCamera: KNN pooling enabled (mode=%s k=%s sample_cortex=%s N_hi=%d -> N_lo=%d)",
                self.cfg.pool_mode,
                self._pool_k,
                self.cfg.pool_sample_cortex,
                self._num_ray_samples,
                self._num_valid_samples,
            )
        else:
            self._pooled_output = None

    # ------------------------------------------------------------------
    #  Per-frame overrides
    # ------------------------------------------------------------------

    def set_calibrated_fixation(self, fixation: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        super().set_calibrated_fixation(fixation, env_ids=env_ids)
        self._write_calibrated_local_pose(env_ids)

    def set_calibrated_angles(self, angles: torch.Tensor) -> None:
        super().set_calibrated_angles(angles)
        self._write_calibrated_local_pose()

    def _write_calibrated_local_pose(self, env_ids: torch.Tensor | None = None) -> None:
        # Replicator reads camera prims, not update_camera's pose arguments.
        # Write when the command arrives, before a visualizer can render the step.
        # The saved local base prevents accumulation and preserves parent motion.
        basis = self._calibrated_basis
        with torch.autocast(basis.device.type, enabled=False):
            relative = compose_rotations(compose_rotations(basis, self._calibrated_rotation), basis.T)
            local_orientation = quat_from_matrix(compose_rotations(self._calibrated_base_local_orientation, relative))
            if env_ids is None:
                self._view.set_local_poses(orientations=wp.from_torch(local_orientation.contiguous(), dtype=wp.vec4f))
            else:
                self._view.set_local_poses(
                    orientations=wp.from_torch(local_orientation[env_ids].contiguous(), dtype=wp.vec4f),
                    indices=wp.from_torch(env_ids.to(dtype=torch.int32).contiguous()),
                )

    def set_fixed_fov_fixation(
        self,
        fixation_angles: torch.Tensor,
        *,
        angle_model: str,
    ) -> None:
        """Update fixed-frustum masking without changing the camera pose."""
        if not self.cfg.fixed_software_fov:
            raise RuntimeError("set_fixed_fov_fixation requires fixed_software_fov=True.")
        if angle_model not in FIXED_FOV_ANGLE_MODELS:
            raise ValueError(f"angle_model must be one of {FIXED_FOV_ANGLE_MODELS}, got {angle_model!r}.")
        self._fixed_fov_fixation = fixation_angles
        self._fixed_fov_angle_model = angle_model

    def _update_buffers_impl(self, env_mask: wp.array):
        if not self._env_mask_has_any(env_mask):
            return

        # The renderer was bound to ``self._texture_output`` at create time
        # via ``set_outputs`` and continues to write per-env pixel data into
        # that dict regardless of what ``self._data.output`` currently
        # references.  ``self._data.output`` already points at the foveated
        # (or pooled) buffer dict; the parent's ``render_into_camera`` only
        # uses ``camera_data`` to populate ``camera_data.info`` (now a flat
        # ``dict[str, Any]``), so we don't need to swap anything.
        super()._update_buffers_impl(env_mask)
        self.process_rendered_frame()

    def process_rendered_frame(self) -> None:
        """Extract, mask and pool the frame written by scalar or batched rendering."""
        if not self._env_mask_has_any(self._is_outdated):
            return
        # Extract per-sample values from the per-env texture buffer.
        self._extract_foveated_samples()
        if self.cfg.renderer_cfg.renderer_type == "ovrtx" and "distance_to_camera" in self._foveated_output:
            # OVRTX's LUT range AOV measures from the near-clipped ray origin.
            self._foveated_output["distance_to_camera"].add_(self.cfg.spawn.clipping_range[0])
        mask_grid_samples(self._foveated_output, self._ray_coords)
        self._apply_fixed_fov_mask()

        # Optionally KNN-pool highres -> lowres samples.  Integer-dtype
        # outputs (segmentation class IDs) use the dedicated max-mode
        # pooler so pooled values stay valid IDs.
        if self._pooler is not None and self._pooled_output is not None:
            has_rgba_alias = "rgba" in self._foveated_output
            for name, hi in self._foveated_output.items():
                if name == "rgb" and has_rgba_alias:
                    continue  # alias — refreshed below
                lo = self._pooled_output[name]
                use_int_pool = self._pooler_int is not None and is_segmentation_output(name)
                pooler = self._pooler_int if use_int_pool else self._pooler
                pooled = pool_hwc_buffer(pooler, hi.float())
                lo.copy_(pooled.to(lo.dtype))
            if "rgba" in self._pooled_output and "rgb" in self.cfg.data_types:
                self._pooled_output["rgb"] = self._pooled_output["rgba"][:, :, :, :3]
        if self._native_grid:
            source = self._pooled_output if self._pooled_output is not None else self._foveated_output
            copy_grid_outputs(source, self._grid_output, self._grid_indices)

    def _extract_foveated_samples(self):
        """Extract per-sample values from rendered 2D textures into (B, N, 1, C) buffers."""
        px = self._extract_px
        py = self._extract_py
        has_rgba_alias = "rgba" in self._texture_output
        for name in self._texture_output:
            if name == "rgb" and has_rgba_alias:
                continue  # handled via rgba alias below
            tex = self._texture_output[name]  # (B, th, tw, C)
            fov = self._foveated_output[name]  # (B, N, 1, C)
            if self._native_grid:
                tensor = tex.torch if hasattr(tex, "torch") else tex
                fov.copy_(tensor.reshape_as(fov))
            else:
                fov[:, :, 0, :] = tex[:, py, px, :]
        if "rgba" in self._foveated_output and "rgb" in self.cfg.data_types:
            self._foveated_output["rgb"] = self._foveated_output["rgba"][:, :, :, :3]

    def _apply_fixed_fov_mask(self) -> None:
        if not self.cfg.fixed_software_fov:
            return
        batch_size = next(iter(self._foveated_output.values())).shape[0]
        fixation = self._fixed_fov_fixation
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
        has_rgba_alias = "rgba" in self._foveated_output
        for name, output in self._foveated_output.items():
            if name == "rgb" and has_rgba_alias:
                continue
            output.masked_fill_(invalid, 0)
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
