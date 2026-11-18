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

from . import lens_lut
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
from .pooling import build_highres_coords, build_lowres_coords, build_pooler, is_segmentation_output

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
        ovrtx = cfg.renderer_cfg.renderer_type == "ovrtx"
        if ovrtx and getattr(cfg.renderer_cfg, "async_rendering", False) and cfg.mask_to_sensor_fov:
            raise ValueError(
                "OVRTX async_rendering returns the previous capture, which the sensor mask would cut with the current"
                " gaze. Disable async_rendering or mask_to_sensor_fov."
            )
        # OVRTX renders the poses the camera hands it, so its pose update turns the gaze. Replicator reads camera
        # prims, so Isaac RTX writes the gaze into them instead.
        self._gaze_pose_reads_angles = ovrtx
        # The parent ``Camera`` allocates the renderer-bound texture buffers; the published ones follow from them.
        self._texture_output: dict[str, torch.Tensor] = {}
        self._foveated_output: dict[str, torch.Tensor] = {}
        self._pooled_output: dict[str, torch.Tensor] | None = None
        if ovrtx:
            self.cfg = cfg
            self._prepare_projection()
        super().__init__(cfg)
        if ovrtx:
            for prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage):
                lens_lut.apply_lens_lut_to_prim(prim, self._dir_tex_path, self._exit_tex_path)

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _prepare_projection(self) -> None:
        # The lens textures and sample extraction use the rendered samples: the high-resolution ones when pooling.
        self._sampling_coords, matched_res = build_lowres_coords(self.cfg, device="cpu")
        self._sampling_coords_highres = build_highres_coords(self.cfg, matched_res, device="cpu")
        ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._ray_coords = ray_coords
        self._native_grid = self.cfg.foveated_style == GRID_STYLE
        self._num_valid_samples = len(self._sampling_coords)
        self._num_ray_samples = len(ray_coords)

        # image_rotation_deg rolls each ray's polar angle before its direction is computed; see its configuration.
        theta_rad = ray_coords.polar[:, 1].to(torch.float64)
        if self.cfg.image_rotation_deg:
            theta_rad = theta_rad + math.radians(float(self.cfg.image_rotation_deg))
        if self._native_grid:
            dirs = native_grid_directions(ray_coords, image_rotation_deg=self.cfg.image_rotation_deg)
        else:
            dirs = retinal_directions(ray_coords.cartesian, ray_coords.fov, self.cfg.image_rotation_deg)

        # Grid styles render straight into their native grid; sample-set styles are packed onto a cortical disc,
        # which keeps neighbouring samples adjacent for the renderer's spatial filtering.
        if self._native_grid:
            tw = th = ray_coords.resolution
            axis = (torch.arange(tw, dtype=torch.float32) + 0.5) / tw
            row, col = torch.meshgrid(axis, axis, indexing="ij")
            uv = torch.stack((col, row), -1).reshape(-1, 2)
        else:
            tw = th = max(64, math.ceil(math.sqrt(4 * self._num_ray_samples / math.pi)))
            # Eccentricity [deg] of each sample; the disc is laid out with the same
            # field of view and CMF constant the samples were generated with.
            eccentricity = ray_coords.polar[:, 0].to(torch.float64) * (ray_coords.fov / 2.0)
            uv = lens_lut.samples_uv_cortical_disc(
                eccentricity, theta_rad, ray_coords.fov, ray_coords.cmf_a, device="cpu"
            )
        self._texture_width = tw
        self._texture_height = th
        self._sample_uv = uv

        # The exit texture is an octahedral inverse map (ray direction -> sensor NDC) over the hemisphere, of which
        # only the camera's field of view is populated. Four times the direction texture's size gives each sample a
        # distinct exit texel for the nearest-neighbour lookup.
        if self.cfg.exit_texture_size is not None:
            EW = EH = self.cfg.exit_texture_size
        else:
            EW = EH = max(128, 4 * max(tw, th))
        if self._native_grid:
            dir_tex, exit_tex = lens_lut.build_native_grid_textures(
                ray_coords, image_rotation_deg=self.cfg.image_rotation_deg, exit_size=EW
            )
        else:
            dir_tex, exit_tex = lens_lut.build_lut_textures(uv, dirs, tw, th, EW, EH, device="cpu")
        # The camera owns its textures' directory, which is removed when the camera is released.
        self._lut_dir = tempfile.TemporaryDirectory(prefix="rtx_foveated_luts_")
        self._dir_tex_path, self._exit_tex_path = lens_lut.save_lut_textures(
            dir_tex, exit_tex, self._lut_dir.name, "foveated"
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
            self._lut_dir.name,
        )
        # The parent allocates (B, th, tw, C) texture buffers.
        self.cfg.height = th
        self.cfg.width = tw

    def _initialize_impl(self) -> None:
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            self._prepare_projection()
        tw, th = self._texture_width, self._texture_height
        # Author the lens before Camera prepares the renderer's stage. OVRTX
        # snapshots a detached stage, so later USD edits would miss its cameras.
        prims = sim_utils.find_matching_prims(self.cfg.prim_path, self.stage)
        for prim in prims:
            lens_lut.apply_lens_lut_to_prim(
                prim, direction_texture_path=self._dir_tex_path, exit_texture_path=self._exit_tex_path
            )
        # OVRTX's lookup-table range is measured from the near-clipped ray origin.
        self._near_clip = self._near_clipping_distance(prims[0]) if prims else 0.0

        super()._initialize_impl()

        # OVRTX owns its runtime; Kit extensions belong only to Isaac RTX.
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            for ext_name in ("omni.sensors.nv.common", "omni.sensors.nv.camera"):
                if not ext_manager.set_extension_enabled_immediate(ext_name, True):
                    logger.warning("Could not enable Kit extension '%s'; the lens lookup may not render.", ext_name)

        self._sampling_coords = self._sampling_coords.to(device=self._device)
        if self._sampling_coords_highres is not None:
            self._sampling_coords_highres = self._sampling_coords_highres.to(device=self._device)
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._initialize_gaze()
        _, local_orientation = self._view.get_local_poses()
        self._base_local_orientation = matrix_from_quat(local_orientation.torch).clone()
        self._pooler, self._pool_k = build_pooler(
            self.cfg, self._sampling_coords, self._sampling_coords_highres, device=str(self._device)
        )
        self._pooler_int = None

        # The renderer keeps writing the texture buffers the parent bound to it; the camera publishes others.
        self._texture_output = self._data.output
        textures = {name: tex.torch if hasattr(tex, "torch") else tex for name, tex in self._texture_output.items()}
        # A native grid texture already holds the samples in order, so one kernel per output masks it and lays it
        # out. OVRTX's camera distance needs its near-plane offset first, and keeps the step-by-step path.
        offset_distance = self.cfg.renderer_cfg.renderer_type == "ovrtx" and "distance_to_camera" in textures
        fused = self._native_grid and self._pooler is None and not offset_distance
        if fused:
            count = self._num_ray_samples
            samples = {name: tex.reshape(tex.shape[0], count, tex.shape[-1]) for name, tex in textures.items()}
            self._grid_output = allocate_grid_outputs(samples, self._sampling_coords)
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            self._prepare_grid_publish(samples)
            published = self._grid_output
        else:
            if not self._native_grid:
                uv_gpu = self._sample_uv.to(self._device)
                self._extract_px = (uv_gpu[:, 0] * tw).long().clamp(0, tw - 1)
                self._extract_py = (uv_gpu[:, 1] * th).long().clamp(0, th - 1)
            self._allocate_foveated_buffers()
            published = self._pooled_output if self._pooled_output is not None else self._foveated_output
            if self._native_grid:
                self._grid_output = allocate_grid_outputs(published, self._sampling_coords)
                self._grid_indices = grid_sample_indices(self._sampling_coords)
                published = self._grid_output
        # ``CameraData.output`` is read-only, so its backing field is replaced.
        self._data._output = proxy_outputs(published)
        self._set_published_shape(self._data._output)

    def _near_clipping_distance(self, prim) -> float:
        """Near clipping distance [m] of the camera, from its spawn configuration or else its prim."""
        if self.cfg.spawn is not None:
            return float(self.cfg.spawn.clipping_range[0])
        from pxr import UsdGeom

        clipping = UsdGeom.Camera(prim).GetClippingRangeAttr().Get()
        return float(clipping[0]) if clipping is not None else 0.0

    def _allocate_foveated_buffers(self):
        """Allocate (B, N_hi, 1, C) and, when pooling, (B, N_lo, 1, C) buffers from the texture buffers' shapes."""
        has_rgba_alias = "rgba" in self._texture_output
        self._foveated_output = {}
        for name, buf in self._texture_output.items():
            if name == "rgb" and has_rgba_alias:
                continue  # an alias of rgba, below
            # Texture buffers are ProxyArrays; their Torch view carries the dtype and device.
            buf_t = buf.torch if hasattr(buf, "torch") else buf
            self._foveated_output[name] = torch.zeros(
                (self._view.count, self._num_ray_samples, 1, buf_t.shape[-1]), device=buf_t.device, dtype=buf_t.dtype
            )
        if "rgba" in self._foveated_output and "rgb" in self.cfg.data_types:
            self._foveated_output["rgb"] = self._foveated_output["rgba"][:, :, :, :3]

        if self._pooler is None:
            self._pooled_output = None
            return
        self._pooled_output = {}
        needs_int_pooler = False
        for name, buf in self._foveated_output.items():
            if name == "rgb" and has_rgba_alias:
                continue
            self._pooled_output[name] = torch.zeros(
                (self._view.count, self._num_valid_samples, 1, buf.shape[-1]), device=self._device, dtype=buf.dtype
            )
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

    # ------------------------------------------------------------------
    #  Gaze and mount
    # ------------------------------------------------------------------

    def set_gaze_angles(self, angles_deg: torch.Tensor) -> None:
        super().set_gaze_angles(angles_deg)
        if not self._gaze_pose_reads_angles:
            self._write_gaze_local_pose()

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
        """Move the camera's mount; the gaze stays relative to the new mount.

        Args are those of :meth:`isaaclab.sensors.Camera.set_world_poses`.
        """
        super().set_world_poses(positions, orientations, env_ids, convention)
        # A position-only move keeps the prims' gaze-turned orientation, and with it the saved mount.
        if orientations is not None:
            self._refresh_mount(env_ids)

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        """Point the camera's mount from ``eyes`` at ``targets``; the gaze stays relative to the new mount.

        Args are those of :meth:`isaaclab.sensors.Camera.set_world_poses_from_view`.
        """
        super().set_world_poses_from_view(eyes, targets, env_ids)
        ids = self._env_indices(env_ids)
        # Rows whose eye equals their target are skipped by the base method and keep their mount.
        moved = (self._as_device_tensor(eyes, 3) - self._as_device_tensor(targets, 3)).norm(dim=-1) > 0
        self._refresh_mount(ids[moved])

    def _env_indices(self, env_ids) -> torch.Tensor:
        """The camera indices that ``env_ids`` selects, as a tensor."""
        every = torch.arange(self._view.count, device=self._device)
        if env_ids is None or isinstance(env_ids, slice):
            return every[slice(None) if env_ids is None else env_ids]
        if isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long).reshape(-1)

    def _refresh_mount(self, env_ids) -> None:
        """Save the moved prims' orientations as their bare mounts and write the gaze on top of them again."""
        if self._gaze_pose_reads_angles:
            return  # The pose update turns the new mount by the gaze.
        ids = self._env_indices(env_ids)
        _, local_orientation = self._view.get_local_poses()
        self._base_local_orientation[ids] = matrix_from_quat(local_orientation.torch[ids])
        if self._gaze_rotation is not None:
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

    # ------------------------------------------------------------------
    #  Per-frame overrides
    # ------------------------------------------------------------------

    def _update_buffers_impl(self, env_mask: wp.array):
        if not self._env_mask_has_any(env_mask):
            return
        # The renderer writes the texture buffers bound at creation, whatever ``self._data.output`` publishes.
        super()._update_buffers_impl(env_mask)
        self.process_rendered_frame()

    def process_rendered_frame(self) -> None:
        """Mask and publish the frame of the environments that scalar or batched rendering updated."""
        if not self._env_mask_has_any(self._is_outdated):
            return
        if self._grid_publish:
            self._publish_grid(self._is_outdated)
            return
        self._extract_foveated_samples()
        if self.cfg.renderer_cfg.renderer_type == "ovrtx" and "distance_to_camera" in self._foveated_output:
            self._foveated_output["distance_to_camera"].add_(self._near_clip)
        mask_grid_samples(self._foveated_output, self._ray_coords)
        self._apply_sensor_fov_mask(self._foveated_output)

        if self._pooled_output is not None:
            for name, hi in self._foveated_output.items():
                if name == "rgb" and "rgba" in self._foveated_output:
                    continue  # rgba alias
                self._pooled_output[name].copy_(self._pool_output(name, hi))
        if self._native_grid:
            source = self._pooled_output if self._pooled_output is not None else self._foveated_output
            copy_grid_outputs(source, self._grid_output, self._grid_indices)

    def _extract_foveated_samples(self):
        """Extract per-sample values from rendered 2D textures into (B, N, 1, C) buffers."""
        for name in self._texture_output:
            if name == "rgb" and "rgba" in self._texture_output:
                continue  # rgba alias
            tex = self._texture_output[name]  # (B, th, tw, C)
            fov = self._foveated_output[name]  # (B, N, 1, C)
            tensor = tex.torch if hasattr(tex, "torch") else tex
            if self._native_grid:
                fov.copy_(tensor.reshape_as(fov))
            else:
                fov[:, :, 0, :] = tensor[:, self._extract_py, self._extract_px, :]
