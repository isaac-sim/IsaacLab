# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual gaze with a static physical camera and a static rendering ray bundle."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp
from fovi.sensing.projection import CameraModel, angular_directions, gaze_rotation
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .calibrated_geometry import (
    calibrated_fixation_rotation,
    calibrated_mount_angles,
    calibrated_visibility,
    compose_rotations,
    convert_camera_frame_quaternion,
)
from .native_grid import GRID_STYLE, native_grid_directions


class CalibratedVirtualGaze:
    """Calibrated virtual-gaze operations shared by tiled-camera backends."""

    def get_world_positions(self) -> torch.Tensor:
        """Read live camera origins without capturing RGB or changing image metadata.

        Unlike ``data``, this only queries the camera's FrameView and never
        refreshes lazy image buffers. Hybrid virtual gaze leaves origins unchanged.
        """
        position, _ = self.get_world_poses()
        return position

    def get_world_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read live scene-frame poses without refreshing images or their metadata.

        Returns:
            World positions (B, 3), in meters, and OpenGL orientations (B, 4),
            as XYZW quaternions. The tensors borrow the sensor view's buffers.
            RTX gaze writes are included; Warp's renderer-only gaze is not.
        """
        position, orientation = self._view.get_world_poses()
        return position.torch, orientation.torch

    def calibrated_fov_sample_mask(self) -> torch.Tensor:
        """Return the physical-sensor visibility mask in canonical ray order (B,N)."""
        if self._calibrated_rotation is None:
            raise RuntimeError("Set calibrated gaze before requesting its physical-FoV mask")
        return self._calibrated_validity(len(self._calibrated_rotation))

    def set_calibrated_fixation(self, fixation: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set full-batch image fixation, optionally changing only selected environments."""
        with torch.autocast(fixation.device.type, enabled=False):
            rotation, _ = calibrated_fixation_rotation(self._calibrated_camera, fixation.float())
            if env_ids is None:
                self._calibrated_rotation = rotation
            else:
                if self._calibrated_rotation is None:
                    self._calibrated_rotation = self._calibrated_identity.expand_as(rotation).clone()
                self._calibrated_rotation[env_ids] = rotation[env_ids]

    def calibrated_mount_angles(self, fixation: torch.Tensor) -> torch.Tensor:
        """Convert upright image gaze to mount (tilt, pan) radians."""
        with torch.autocast(fixation.device.type, enabled=False):
            return calibrated_mount_angles(self._calibrated_camera, fixation.float())

    def set_calibrated_angles(self, angles: torch.Tensor) -> None:
        """Set virtual gaze from the mount's (tilt, pan) radians."""
        # Geometry must remain float32 even when the policy runs under AMP.
        with torch.autocast(angles.device.type, enabled=False):
            tilt, pan = angles.float().unbind(-1)
            direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
            self._calibrated_rotation = gaze_rotation(direction, "pan_tilt")

    def _apply_calibrated_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        if self._calibrated_rotation is None:
            return
        stored = self._data.quat_w_world
        quat = stored.torch if hasattr(stored, "torch") else stored
        with torch.autocast(quat.device.type, enabled=False):
            base = convert_camera_frame_quaternion(quat, origin="world", target="opengl")
            # Calibration uses the upright image frame; rays include mount roll.
            basis = self._calibrated_basis
            relative = compose_rotations(compose_rotations(basis, self._calibrated_rotation), basis.T)
            composed = quat_from_matrix(compose_rotations(matrix_from_quat(base), relative))
            updated = convert_camera_frame_quaternion(composed, origin="opengl", target="world")
            if env_mask is not None:
                quat.copy_(torch.where(wp.to_torch(env_mask)[:, None], updated, quat))
            elif env_ids is not None:
                indices = wp.to_torch(env_ids) if isinstance(env_ids, wp.array) else env_ids
                quat[indices] = updated[indices]
            else:
                quat.copy_(updated)

    def _calibrated_validity(self, batch_size: int) -> torch.Tensor:
        rays = self._calibrated_rays
        with torch.autocast(rays.device.type, enabled=False):
            rotation = self._calibrated_rotation
            if rotation is None:
                rotation = self._calibrated_identity.expand(batch_size, -1, -1)
            # A receptive field can overlap the sensor even when its center is outside it.
            return calibrated_visibility(self._calibrated_camera, rays, rotation)

    def _initialize_calibrated_gaze(self) -> None:
        if self.cfg.field_geometry != "spherical":
            return
        self._calibrated_camera = CameraModel(**self.cfg.camera_calibration)
        coords = self._sampling_coords_highres or self._sampling_coords
        if coords.style == GRID_STYLE:
            self._calibrated_rays = native_grid_directions(coords, tangent_scale=1.0) * coords.cartesian.new_tensor(
                (1, -1, -1)
            )
        else:
            self._calibrated_rays = angular_directions(coords.cartesian, coords.fov)
        self._calibrated_identity = torch.eye(3, device=coords.device)
        angle = math.radians(self.cfg.image_rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        self._calibrated_basis = self._calibrated_rays.new_tensor(((c, s, 0), (s, -c, 0), (0, 0, -1)))
