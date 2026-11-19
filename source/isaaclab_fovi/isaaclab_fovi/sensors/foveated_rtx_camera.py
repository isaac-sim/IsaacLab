# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native foveated lens textures for Isaac RTX and OVRTX."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab import sim as sim_utils
from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    normalize,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

from . import lens_lut
from .foveated_camera import FoveatedCamera

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg

logger = logging.getLogger(__name__)


class FoveatedRTXCamera(FoveatedCamera):
    """Render the shared retinal grid through an RTX lens lookup texture."""

    cfg: FoveatedCameraCfg

    def __init__(self, cfg: FoveatedCameraCfg):
        ovrtx = cfg.renderer_cfg.renderer_type == "ovrtx"
        if ovrtx and cfg.renderer_cfg.async_rendering and cfg.mask_to_sensor_fov:
            raise ValueError(
                "OVRTX async_rendering returns the previous capture, which the sensor mask would cut with the current"
                " gaze. Disable async_rendering or mask_to_sensor_fov."
            )
        super().__init__(cfg)
        self._prepare_projection()
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
            ids = self._resolve_env_ids_wp(env_ids)
            rotation = self._relative_gaze()
            if ids is not None:
                rotation = rotation[wp.to_torch(ids).long()]
            orientations = quat_mul(orientations, quat_conjugate(rotation))
        return positions, convert_camera_frame_orientation_convention(orientations, origin="opengl", target=convention)

    def process_rendered_frame(self) -> None:
        """Publish the grid after scalar or batched rendering."""
        env_mask = self._is_outdated
        if not self._env_mask_has_any(env_mask):
            return
        if self._near_clip is not None:
            # OVRTX measures distance from the near-clipped ray origin. The scratch is a publication
            # source, so this leaves the renderer's bound buffer and unselected public rows untouched.
            torch.add(self._distance_source, self._near_clip, out=self._distance_corrected)
        self._publish_grid(env_mask)

    def _prepare_projection(self) -> None:
        exit_size = self.cfg.exit_texture_size
        if exit_size is None:
            exit_size = max(128, 4 * self.cfg.foveated_res)
        direction, exit_texture = lens_lut.build_native_grid_textures(
            self._sampling_coords,
            self._retinal_rays,
            image_rotation_deg=self.cfg.image_rotation_deg,
            exit_size=exit_size,
        )
        self._lut_dir = tempfile.TemporaryDirectory(prefix="rtx_foveated_luts_")
        self._dir_tex_path, self._exit_tex_path = lens_lut.save_lut_textures(
            direction, exit_texture, self._lut_dir.name, "foveated"
        )

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._initialize_gaze()
        if not self._gaze_pose_reads_angles:
            _, local_orientation = self._view.get_local_poses()
            self._base_local_orientation = local_orientation.torch.clone()
            self._gaze_basis_quat = quat_from_matrix(self._gaze_basis)

        sources = {name: value.torch for name, value in self._data.output.items()}
        self._near_clip = None
        if self.cfg.renderer_cfg.renderer_type == "ovrtx" and "distance_to_camera" in sources:
            near_clips = [self._near_clipping_distance(camera.GetPrim()) for camera in self._sensor_prims]
            self._distance_source = sources["distance_to_camera"]
            self._near_clip = self._distance_source.new_tensor(near_clips).view(-1, 1, 1, 1)
            self._distance_corrected = torch.empty_like(self._distance_source)
            sources["distance_to_camera"] = self._distance_corrected
        self._prepare_grid_publish(sources)

        # OVRTX owns its runtime; Kit extensions belong only to Isaac RTX.
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            for ext_name in ("omni.sensors.nv.common", "omni.sensors.nv.camera"):
                if not ext_manager.set_extension_enabled_immediate(ext_name, True):
                    logger.warning("Could not enable Kit extension '%s'; the lens lookup may not render.", ext_name)

    def _near_clipping_distance(self, prim) -> float:
        """Near clipping distance [m] of the camera, from its spawn configuration or else its prim."""
        if self.cfg.spawn is not None:
            return float(self.cfg.spawn.clipping_range[0])
        from pxr import UsdGeom

        clipping = UsdGeom.Camera(prim).GetClippingRangeAttr().Get()
        return float(clipping[0]) if clipping is not None else 0.0

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
        """Move the mount while retaining its relative gaze."""
        super().set_world_poses(positions, orientations, env_ids, convention)
        if orientations is not None and not self._gaze_pose_reads_angles:
            ids = self._resolve_env_ids_wp(env_ids)
            self._mount_written(slice(None) if ids is None else wp.to_torch(ids).long())

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        """Point the mount at a target while retaining its relative gaze."""
        super().set_world_poses_from_view(eyes, targets, env_ids)
        if not self._gaze_pose_reads_angles:
            # The base setter leaves rows with a degenerate look-at direction untouched.
            moved = (self._as_device_tensor(eyes, 3) != self._as_device_tensor(targets, 3)).any(dim=-1)
            ids = self._resolve_env_ids_wp(env_ids)
            self._mount_written(moved if ids is None else wp.to_torch(ids).long()[moved])

    def _mount_written(self, ids) -> None:
        _, local_orientation = self._view.get_local_poses()
        self._base_local_orientation[ids] = local_orientation.torch[ids]
        if self._gaze_rotation is not None:
            self._write_gaze_local_pose()

    def _relative_gaze(self) -> torch.Tensor:
        basis = self._gaze_basis_quat.expand(self._view.count, -1)
        gaze = quat_from_matrix(self._gaze_rotation.torch)
        return quat_mul(quat_mul(basis, gaze), quat_conjugate(basis))

    def _write_gaze_local_pose(self) -> None:
        # Replicator reads camera prims. Write the gaze onto the bare local mount so parent motion is preserved.
        orientation = normalize(quat_mul(self._base_local_orientation, self._relative_gaze()))
        self._view.set_local_poses(orientations=wp.from_torch(orientation.contiguous(), dtype=wp.vec4f))

    def _update_buffers_impl(self, env_mask: wp.array):
        if not self._env_mask_has_any(env_mask):
            return
        # The renderer writes the texture buffers bound at creation, whatever ``self._data.output`` publishes.
        super()._update_buffers_impl(env_mask)
        self.process_rendered_frame()

    @staticmethod
    def _pose_update_turns_gaze(cfg: FoveatedCameraCfg) -> bool:
        # OVRTX renders the poses the camera hands it, so its pose update turns the gaze. Replicator reads camera
        # prims, so Isaac RTX writes the gaze into them instead.
        return cfg.renderer_cfg.renderer_type == "ovrtx"

    @staticmethod
    def _update_buffers_batch_impl(sensors: Sequence[FoveatedRTXCamera]) -> None:
        Camera._update_buffers_batch_impl(sensors)
        if sim_utils.SimulationContext.instance() is None:
            return  # The base implementation already invoked scalar capture.
        for sensor in sensors:
            sensor.process_rendered_frame()
