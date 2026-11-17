# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX foveated tiled camera sensor.

Subclasses :class:`isaaclab.sensors.Camera` to render foveated images
via the RTX path tracer using ``OmniLensDistortionLutAPI`` textures.
Per-sample values are extracted from the rendered 2D texture to produce
vector ``(B, N, 1, C)`` or upright native-grid ``(B, H, W, C)`` output buffers,
identical to :class:`FoveatedNewtonCamera`.
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
from .foveated_camera import FoveatedCamera
from .native_grid import (
    GRID_STYLE,
    allocate_grid_outputs,
    copy_grid_outputs,
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
    from isaaclab.renderers.base_renderer import BaseRenderer

    from .foveated_camera_cfg import FoveatedCameraCfg

logger = logging.getLogger(__name__)


class FoveatedRTXCamera(FoveatedCamera):
    """RTX camera that renders with foveated (CMF-distributed) rays.

    Instead of Newton's Warp ray arrays, this sensor creates
    ``OmniLensDistortionLutAPI`` Camera prims with LUT textures that define
    per-texel ray directions.  The RTX path tracer renders a 2D texture
    which is then sampled at pre-computed UV positions to extract per-sample
    values, producing ``(B, N, 1, C)`` output buffers.

    Grid styles render straight into their native grid. Sample-set styles are
    packed onto a cortical disc, which keeps neighbouring samples adjacent for
    DLSS denoising. Camera motion comes from the prim's parent and from the gaze.
    """

    cfg: FoveatedCameraCfg

    @property
    def supports_batch_update(self) -> bool:
        return True

    @staticmethod
    def _update_buffers_batch_impl(sensors: Sequence[FoveatedRTXCamera]) -> None:
        Camera._update_buffers_batch_impl(sensors)
        if sim_utils.SimulationContext.instance() is None:
            return  # The base implementation already invoked scalar capture.
        for sensor in sensors:
            sensor.process_rendered_frame()

    @property
    def renderer(self) -> BaseRenderer:
        """Renderer shared with task-owned visual animation bindings."""
        return self._renderer

    def __init__(self, cfg: FoveatedCameraCfg):
        self._gaze_rotation = None
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
        # 1. Build coords on CPU.  The LUT textures + UV extraction use the
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

        # 2. Compute ray directions and UV mapping from the rendered grid.
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
            dirs = native_grid_directions(ray_coords, image_rotation_deg=self.cfg.image_rotation_deg)
        else:
            dirs = retinal_directions(ray_coords.cartesian, ray_coords.fov, self.cfg.image_rotation_deg)

        # 3. Texture dimensions and UV layout. Grid styles render straight into their
        #    native grid; sample-set styles are packed onto a cortical disc, which keeps
        #    neighbouring samples adjacent for the renderer's spatial filtering.
        if self._native_grid:
            tw = th = ray_coords.resolution
            axis = (torch.arange(tw, dtype=torch.float32) + 0.5) / tw
            row, col = torch.meshgrid(axis, axis, indexing="ij")
            uv = torch.stack((col, row), -1).reshape(-1, 2)
        else:
            tw = th = max(64, math.ceil(math.sqrt(4 * N / math.pi)))
            # Eccentricity [deg] of each sample; the disc is laid out with the same
            # field of view and CMF constant the samples were generated with.
            eccentricity = ray_coords.polar[:, 0].to(torch.float64) * (ray_coords.fov / 2.0)
            uv = gp.samples_uv_cortical_disc(
                eccentricity,
                theta_rad,
                ray_coords.fov,
                ray_coords.cmf_a,
                device="cpu",
            )

        self._texture_width = tw
        self._texture_height = th
        self._sample_uv = uv  # (N, 2) float32

        # 4. Build and save LUT textures.
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
                ray_coords, image_rotation_deg=self.cfg.image_rotation_deg, exit_size=EW
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

        logger.info(
            "FoveatedRTXCamera: %d samples (%d rendered), style=%s, cmf_a=%.4g deg, field of view=%.1f deg, "
            "texture %dx%d, exit lookup %dx%d, lookup textures in %s",
            self._num_valid_samples,
            self._num_ray_samples,
            self.cfg.foveated_style,
            ray_coords.cmf_a,
            ray_coords.fov,
            tw,
            th,
            EW,
            EH,
            self._lut_tmpdir,
        )

        # 5. Set texture dimensions so parent allocates (B, th, tw, C) buffers.
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
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            for ext_name in ("omni.sensors.nv.common", "omni.sensors.nv.camera"):
                if not ext_manager.set_extension_enabled_immediate(ext_name, True):
                    logger.warning("Could not enable Kit extension '%s'; the lens lookup may not render.", ext_name)

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
        if self._sampling_coords_highres is not None:
            self._sampling_coords_highres = self._sampling_coords_highres.to(device=self._device)
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._initialize_gaze()
        _, local_orientation = self._view.get_local_poses()
        self._base_local_orientation = matrix_from_quat(local_orientation.torch).clone()

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
        # Publish the foveated (or pooled) buffers. ``CameraData.output`` is read-only, so the
        # backing field is replaced; the renderer keeps writing to the texture buffers it was
        # bound to by the parent ``_create_buffers``.
        new_output = self._pooled_output if self._pooled_output is not None else self._foveated_output
        if self._native_grid:
            self._grid_output = allocate_grid_outputs(new_output, self._sampling_coords)
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            new_output = self._grid_output
        self._data._output = proxy_outputs(new_output)

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
                "FoveatedRTXCamera: KNN pooling enabled (mode=%s k=%s sample_cortex=%s N_hi=%d -> N_lo=%d)",
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

    def set_gaze_angles(self, angles_deg: torch.Tensor) -> None:
        super().set_gaze_angles(angles_deg)
        self._write_gaze_local_pose()

    def _write_gaze_local_pose(self) -> None:
        # Replicator reads camera prims, not update_camera's pose arguments.
        # Write when the command arrives, before a visualizer can render the step.
        # The saved local base prevents accumulation and preserves parent motion.
        basis = self._gaze_basis
        with torch.autocast(basis.device.type, enabled=False):
            relative = compose_rotations(compose_rotations(basis, self._gaze_rotation), basis.T)
            local_orientation = quat_from_matrix(compose_rotations(self._base_local_orientation, relative))
            self._view.set_local_poses(orientations=wp.from_torch(local_orientation.contiguous(), dtype=wp.vec4f))

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
        self._apply_sensor_fov_mask()

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
            tensor = tex.torch if hasattr(tex, "torch") else tex
            if self._native_grid:
                fov.copy_(tensor.reshape_as(fov))
            else:
                fov[:, :, 0, :] = tensor[:, py, px, :]
        if "rgba" in self._foveated_output and "rgb" in self.cfg.data_types:
            self._foveated_output["rgb"] = self._foveated_output["rgba"][:, :, :, :3]

    def _apply_sensor_fov_mask(self) -> None:
        """Zero the samples that fall outside the physical camera at the current gaze."""
        if not self.cfg.mask_to_sensor_fov:
            return
        batch_size = next(iter(self._foveated_output.values())).shape[0]
        valid = self._gaze_validity(batch_size)
        invalid = ~valid[:, :, None, None]
        has_rgba_alias = "rgba" in self._foveated_output
        for name, output in self._foveated_output.items():
            if name == "rgb" and has_rgba_alias:
                continue
            output.masked_fill_(invalid, 0)
        self._sensor_fov_valid_fraction = valid.float().mean(dim=1)

    # ------------------------------------------------------------------
    #  Output helpers
    # ------------------------------------------------------------------

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
        """Per-environment fraction of samples inside the physical camera, or None before the first masked frame."""
        return getattr(self, "_sensor_fov_valid_fraction", None)
