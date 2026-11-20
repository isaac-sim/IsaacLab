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
from isaaclab.utils import clone
from isaaclab.utils.math import convert_camera_frame_orientation_convention, matrix_from_quat, quat_from_matrix

from . import lens_lut
from .calibrated_geometry import compose_rotations, retinal_directions
from .foveated_camera import FoveatedCamera
from .native_grid import (
    allocate_outputs,
    copy_grid_outputs,
    copy_updated_rows,
    grid_output_shape,
    grid_sample_indices,
    mask_grid_samples,
    proxy_outputs,
)
from .pooling import build_pooler

if TYPE_CHECKING:
    from isaaclab.utils.warp import ProxyArray

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

    def __init__(self, cfg: FoveatedCameraCfg):
        ovrtx = cfg.renderer_cfg.renderer_type == "ovrtx"
        if ovrtx and cfg.renderer_cfg.async_rendering and cfg.mask_to_sensor_fov:
            raise ValueError(
                "OVRTX async_rendering returns the previous capture, which the sensor mask would cut with the current"
                " gaze. Disable async_rendering or mask_to_sensor_fov."
            )
        # The parent ``Camera`` allocates the renderer-bound texture buffers; the published ones follow from them.
        self._texture_output: dict[str, ProxyArray] = {}
        self._foveated_output: dict[str, torch.Tensor] = {}
        self._pooled_output: dict[str, torch.Tensor] | None = None
        # The projection sets the texture size on the configuration before the parent allocates for it; the caller's
        # configuration stays untouched.
        # The textures are square, so the parent's inferred vertical aperture agrees with the lens calibration.
        cfg = clone(cfg)
        self.cfg = cfg
        self._prepare_projection()
        # OVRTX uses the pose the camera hands it. Replicator reads the camera prims.
        super().__init__(cfg, turns_pose_by_gaze=ovrtx)
        # The lens belongs to the camera prims from construction: OVRTX snapshots the stage when the renderer starts.
        for prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage):
            # A locked camera may be treated as static, which skips the motion vectors that DLSS reprojection needs;
            # the gaze moves every foveated camera.
            lock = prim.GetAttribute("omni:kit:cameraLock")
            if lock.IsValid() and lock.Get():
                lock.Set(False)
            lens_lut.apply_lens_lut_to_prim(prim, self._dir_tex_path, self._exit_tex_path)

    def __del__(self, *args, **kwargs):
        """Release the renderer, then the lens textures it reads."""
        super().__del__(*args, **kwargs)
        lut_dir = getattr(self, "_lut_dir", None)
        if lut_dir is not None:
            lut_dir.cleanup()
            self._lut_dir = None

    @property
    def supports_batch_update(self) -> bool:
        return True

    def set_gaze_angles(self, angles_deg: torch.Tensor) -> None:
        super().set_gaze_angles(angles_deg)
        if not self._gaze_pose_reads_angles:
            self._write_gaze_local_pose()

    def get_world_poses(self, env_ids=None, convention="ros"):
        """Read the cameras' mount poses, without the gaze, as every foveated camera reports them.

        Isaac RTX holds the gaze in the camera prims, so it is taken back out; passing the result to
        :meth:`set_world_poses` keeps the mount and the gaze as they are.

        Args and returns are those of :meth:`isaaclab.sensors.Camera.get_world_poses`.
        """
        positions, orientations = super().get_world_poses(env_ids, convention="opengl")
        if not self._gaze_pose_reads_angles and self._gaze_rotation is not None:
            # The prims hold the mount turned by the gaze's rotation in their own (OpenGL) frame.
            basis = self._gaze_basis
            gaze = self._gaze_rotation.torch[self._env_indices(env_ids)]
            with torch.autocast(basis.device.type, enabled=False):
                relative = compose_rotations(compose_rotations(basis, gaze), basis.T)
                orientations = quat_from_matrix(
                    compose_rotations(matrix_from_quat(orientations), relative.transpose(-1, -2))
                )
        return positions, convert_camera_frame_orientation_convention(orientations, origin="opengl", target=convention)

    def process_rendered_frame(self) -> None:
        """Mask and publish the frame of the environments that scalar or batched rendering updated."""
        if not self._env_mask_has_any(self._is_outdated):
            return
        if self._grid_publish:
            self._publish_grid(self._is_outdated)
            return
        # Only the updated environments' samples change, so pooling and the grid layout reproduce every other
        # environment's last frame.
        self._extract_foveated_samples(self._is_outdated)
        mask_grid_samples(self._foveated_output, self._ray_coords)
        self._apply_sensor_fov_mask(self._foveated_output, self._is_outdated)

        if self._pooled_output is not None:
            for name, hi in self._foveated_output.items():
                if name == "rgb" and "rgba" in self._foveated_output:
                    continue  # rgba alias
                self._pooled_output[name].copy_(self._pool_output(name, hi))
        if self._native_grid:
            source = self._pooled_output if self._pooled_output is not None else self._foveated_output
            copy_grid_outputs(source, self._grid_output, self._grid_indices)

    def _prepare_projection(self) -> None:
        # The lens textures and sample extraction use the rendered samples: the high-resolution ones when pooling.
        self._prepare_sampling()
        ray_coords = self._ray_coords

        size = ray_coords.resolution
        if not self._native_grid:
            size = max(64, math.ceil(math.sqrt(4 * self._num_ray_samples / math.pi)))
        exit_size = self.cfg.exit_texture_size
        if exit_size is None:
            exit_size = max(128, 4 * size)
        if self._native_grid:
            dir_tex, exit_tex = lens_lut.build_native_grid_textures(
                ray_coords, image_rotation_deg=self.cfg.image_rotation_deg, exit_size=exit_size
            )
        else:
            # Pack sample sets on the cortical disc so nearby samples stay adjacent for RTX filtering.
            theta = ray_coords.polar[:, 1].to(torch.float64)
            if self.cfg.image_rotation_deg:
                theta = theta + math.radians(float(self.cfg.image_rotation_deg))
            eccentricity = ray_coords.polar[:, 0].to(torch.float64) * (ray_coords.fov / 2.0)
            self._sample_uv = lens_lut.samples_uv_cortical_disc(
                eccentricity, theta, ray_coords.fov, ray_coords.cmf_a, device="cpu"
            )
            dirs = retinal_directions(ray_coords.cartesian, ray_coords.fov, self.cfg.image_rotation_deg)
            dir_tex, exit_tex = lens_lut.build_lut_textures(
                self._sample_uv, dirs, size, size, exit_size, exit_size, device="cpu"
            )
        self._texture_size = size

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
            size,
            size,
            exit_size,
            exit_size,
            self._lut_dir.name,
        )
        self.cfg.height = self.cfg.width = size

    def _initialize_impl(self) -> None:
        # OVRTX measures distance_to_camera from each camera's near-clipped ray origin, so it needs the near clips.
        self._offset_distance = (
            self.cfg.renderer_cfg.renderer_type == "ovrtx" and "distance_to_camera" in self.cfg.data_types
        )
        near_clips = None
        if self._offset_distance:
            prims = sim_utils.find_matching_prims(self.cfg.prim_path, self.stage)
            near_clips = [self._near_clipping_distance(prim) for prim in prims]

        super()._initialize_impl()

        self._near_clip = None
        if near_clips is not None:
            self._near_clip = torch.tensor(near_clips, dtype=torch.float32, device=self._device).view(-1, 1, 1, 1)

        # OVRTX owns its runtime; Kit extensions belong only to Isaac RTX.
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            for ext_name in ("omni.sensors.nv.common", "omni.sensors.nv.camera"):
                if not ext_manager.set_extension_enabled_immediate(ext_name, True):
                    logger.warning("Could not enable Kit extension '%s'; the lens lookup may not render.", ext_name)

        self._initialize_sampling()
        _, local_orientation = self._view.get_local_poses()
        self._base_local_orientation = matrix_from_quat(local_orientation.torch).clone()
        self._pooler, self._pool_k = build_pooler(
            self.cfg, self._sampling_coords, self._sampling_coords_highres, device=str(self._device)
        )

        # The renderer keeps writing the texture buffers the parent bound to it; the camera publishes others.
        self._texture_output = self._data.output
        textures = {name: tex.torch for name, tex in self._texture_output.items()}
        # A native grid texture already holds the samples in order, so one kernel per output masks it and lays it
        # out. OVRTX's camera distance needs its near-plane offset first, and keeps the step-by-step path.
        fused = self._native_grid and self._pooler is None and not self._offset_distance
        if fused:
            count = self._num_ray_samples
            samples = {name: tex.reshape(tex.shape[0], count, tex.shape[-1]) for name, tex in textures.items()}
            self._grid_output = allocate_outputs(samples, *grid_output_shape(self._sampling_coords))
            self._grid_indices = grid_sample_indices(self._sampling_coords)
            self._prepare_grid_publish(samples)
            published = self._grid_output
        else:
            if not self._native_grid:
                uv_gpu = self._sample_uv.to(self._device)
                size = self._texture_size
                self._extract_px = (uv_gpu[:, 0] * size).long().clamp(0, size - 1)
                self._extract_py = (uv_gpu[:, 1] * size).long().clamp(0, size - 1)
            self._foveated_output = allocate_outputs(textures, self._num_ray_samples, 1)
            published = self._foveated_output
            self._pooled_output = None
            if self._pooler is not None:
                published = self._pooled_output = allocate_outputs(published, self._num_valid_samples, 1)
                self._prepare_segmentation_pooling(published)
                logger.info(
                    "FoveatedRTXCamera: %s pooling (mode=%s k=%s sample_cortex=%s N_hi=%d -> N_lo=%d)",
                    type(self._pooler).__name__,
                    self.cfg.pool_mode,
                    self._pool_k,
                    self.cfg.pool_sample_cortex,
                    self._num_ray_samples,
                    self._num_valid_samples,
                )
            if self._native_grid:
                self._grid_output = allocate_outputs(published, *grid_output_shape(self._sampling_coords))
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

    def _mount_written(self, env_ids) -> None:
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
            relative = compose_rotations(compose_rotations(basis, self._gaze_rotation.torch), basis.T)
            local_orientation = quat_from_matrix(compose_rotations(self._base_local_orientation, relative))
            self._view.set_local_poses(orientations=wp.from_torch(local_orientation.contiguous(), dtype=wp.vec4f))

    def _update_buffers_impl(self, env_mask: wp.array):
        if not self._env_mask_has_any(env_mask):
            return
        # The renderer writes the texture buffers bound at creation, whatever ``self._data.output`` publishes.
        super()._update_buffers_impl(env_mask)
        self.process_rendered_frame()

    def _extract_foveated_samples(self, env_mask: wp.array):
        """Extract the updated environments' per-sample values from the rendered 2D textures into the (B, N, 1, C)
        buffers; the other environments keep their last samples."""
        for name in self._texture_output:
            if name == "rgb" and "rgba" in self._texture_output:
                continue  # rgba alias
            tex = self._texture_output[name]  # (B, th, tw, C)
            fov = self._foveated_output[name]  # (B, N, 1, C)
            tensor = tex.torch
            if self._native_grid:
                samples = tensor.reshape_as(fov)
            else:
                samples = tensor[:, self._extract_py, self._extract_px, :].unsqueeze(2)
            if self._offset_distance and name == "distance_to_camera":
                # OVRTX measures the distance from the near-clipped ray origin.
                samples = samples + self._near_clip.to(samples.dtype)
            copy_updated_rows(samples, fov, env_mask)

    @staticmethod
    def _update_buffers_batch_impl(sensors: Sequence[FoveatedRTXCamera]) -> None:
        Camera._update_buffers_batch_impl(sensors)
        if sim_utils.SimulationContext.instance() is None:
            return  # The base implementation already invoked scalar capture.
        for sensor in sensors:
            sensor.process_rendered_frame()
