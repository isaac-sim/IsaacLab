# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual gaze: the retina turns inside a fixed physical camera."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp
from fovi.sensing.projection import angular_directions, gaze_rotation

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .calibrated_geometry import (
    compose_rotations,
    convert_camera_frame_quaternion,
    resolve_camera_model,
    sensor_visibility,
)
from .native_grid import GRID_STYLE, native_grid_directions


class VirtualGaze:
    """Gaze shared by the foveated camera backends.

    The camera prim stays where the scene puts it. Gaze turns the retina relative
    to the optical axis of the physical camera it looks through.
    """

    def set_gaze_angles(self, angles: torch.Tensor) -> None:
        """Turn the retina to (tilt, pan) [rad] from the physical camera's optical axis.

        Positive tilt looks up and positive pan looks left. Zero looks along the
        optical axis.

        Args:
            angles: Gaze angles for every environment, shape (num_envs, 2).
        """
        # Geometry must remain float32 even when the policy runs under AMP.
        with torch.autocast(angles.device.type, enabled=False):
            tilt, pan = angles.float().unbind(-1)
            direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
            self._gaze_rotation = gaze_rotation(direction, "pan_tilt")

    def _apply_gaze_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        if self._gaze_rotation is None:
            return
        stored = self._data.quat_w_world
        quat = stored.torch if hasattr(stored, "torch") else stored
        with torch.autocast(quat.device.type, enabled=False):
            base = convert_camera_frame_quaternion(quat, origin="world", target="opengl")
            # Gaze is defined in the upright image frame; rays include mount roll.
            basis = self._gaze_basis
            relative = compose_rotations(compose_rotations(basis, self._gaze_rotation), basis.T)
            composed = quat_from_matrix(compose_rotations(matrix_from_quat(base), relative))
            updated = convert_camera_frame_quaternion(composed, origin="opengl", target="world")
            if env_mask is not None:
                quat.copy_(torch.where(wp.to_torch(env_mask)[:, None], updated, quat))
            elif env_ids is not None:
                indices = wp.to_torch(env_ids) if isinstance(env_ids, wp.array) else env_ids
                quat[indices] = updated[indices]
            else:
                quat.copy_(updated)

    def _gaze_validity(self, batch_size: int) -> torch.Tensor:
        """Return which samples the physical camera sees at the current gaze, shape (batch_size, num_samples)."""
        rays = self._gaze_rays
        with torch.autocast(rays.device.type, enabled=False):
            rotation = self._gaze_rotation
            if rotation is None:
                rotation = self._gaze_identity.expand(batch_size, -1, -1)
            # A receptive field can overlap the sensor even when its center is outside it.
            return sensor_visibility(self._camera_model, rays, rotation)

    def _initialize_gaze(self) -> None:
        self._camera_model = resolve_camera_model(self.cfg)
        coords = self._sampling_coords_highres or self._sampling_coords
        if coords.style == GRID_STYLE:
            self._gaze_rays = native_grid_directions(coords) * coords.cartesian.new_tensor((1, -1, -1))
        else:
            self._gaze_rays = angular_directions(coords.cartesian, coords.fov)
        self._gaze_identity = torch.eye(3, device=coords.device)
        angle = math.radians(self.cfg.image_rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        self._gaze_basis = self._gaze_rays.new_tensor(((c, s, 0), (s, -c, 0), (0, 0, -1)))
