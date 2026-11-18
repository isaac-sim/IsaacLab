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
from fovi.sensing.projection import angular_directions

from .calibrated_geometry import convert_camera_frame_quaternion, resolve_camera_model
from .gaze_kernels import (
    apply_gaze_indexed_kernel,
    apply_gaze_kernel,
    apply_gaze_masked_kernel,
    camera_params,
    sensor_visibility_kernel,
    set_gaze_kernel,
    torch_stream,
)
from .native_grid import GRID_STYLE, native_grid_directions


class VirtualGaze:
    """Gaze shared by the foveated camera backends.

    The camera prim stays where the scene puts it. Gaze turns the retina relative
    to the optical axis of the physical camera it looks through.
    """

    def set_gaze_angles(self, angles_deg: torch.Tensor) -> None:
        """Turn the retina to (tilt, pan) [deg] from the physical camera's optical axis.

        Positive tilt looks up and positive pan looks left. Zero looks along the
        optical axis.

        Args:
            angles_deg: Gaze angles for every environment [deg], shape (num_envs, 2).
        """
        angles = angles_deg.detach().float().contiguous()
        rotation = self._gaze_rotation
        if rotation is None or rotation.shape[0] != len(angles) or rotation.device != angles.device:
            self._gaze_rotation = torch.empty(len(angles), 3, 3, device=angles.device)
            self._gaze_pose_delta = torch.empty(len(angles), 4, device=angles.device)
        with torch_stream(angles.device):
            wp.launch(
                set_gaze_kernel,
                dim=len(angles),
                inputs=[wp.from_torch(angles, dtype=wp.vec2f), *self._gaze_frames],
                outputs=[
                    wp.from_torch(self._gaze_rotation, dtype=wp.mat33f),
                    wp.from_torch(self._gaze_pose_delta, dtype=wp.quatf),
                ],
            )

    def _apply_gaze_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        if self._gaze_rotation is None:
            return
        orientation = self._data.quat_w_world
        orientation = orientation.warp if hasattr(orientation, "warp") else wp.from_torch(orientation, dtype=wp.quatf)
        delta = wp.from_torch(self._gaze_pose_delta, dtype=wp.quatf)
        with torch_stream(self._gaze_pose_delta.device):
            if env_mask is not None:
                wp.launch(apply_gaze_masked_kernel, dim=len(delta), inputs=[orientation, delta, env_mask])
            elif env_ids is None:
                wp.launch(apply_gaze_kernel, dim=len(delta), inputs=[orientation, delta])
            else:
                if not isinstance(env_ids, wp.array):
                    ids = torch.as_tensor(env_ids, dtype=torch.int32, device=self._gaze_pose_delta.device)
                    env_ids = wp.from_torch(ids.contiguous())
                wp.launch(apply_gaze_indexed_kernel, dim=len(env_ids), inputs=[orientation, delta, env_ids])

    def _gaze_validity(self, batch_size: int) -> torch.Tensor:
        """Return which samples the physical camera sees at the current gaze, shape (batch_size, num_samples)."""
        rotation = self._gaze_rotation
        if rotation is None:
            rotation = self._gaze_identity.expand(batch_size, -1, -1).contiguous()
        rays = self._gaze_rays.to(rotation.device)
        valid = torch.empty(batch_size, len(rays), dtype=torch.bool, device=rotation.device)
        # A receptive field can overlap the sensor even when its center is outside it.
        with torch_stream(rotation.device):
            wp.launch(
                sensor_visibility_kernel,
                dim=valid.shape,
                inputs=[
                    wp.from_torch(rays, dtype=wp.vec3f),
                    wp.from_torch(rotation.contiguous(), dtype=wp.mat33f),
                    self._gaze_camera,
                ],
                outputs=[wp.from_torch(valid, dtype=wp.bool)],
            )
        return valid

    def _initialize_gaze(self) -> None:
        self._camera_model = resolve_camera_model(self.cfg)
        self._gaze_camera = camera_params(self._camera_model)
        coords = self._sampling_coords_highres or self._sampling_coords
        if coords.style == GRID_STYLE:
            self._gaze_rays = native_grid_directions(coords) * coords.cartesian.new_tensor((1, -1, -1))
        else:
            self._gaze_rays = angular_directions(coords.cartesian, coords.fov)
        self._gaze_rays = self._gaze_rays.float().contiguous()
        device = self._gaze_rays.device
        self._gaze_identity = torch.eye(3, device=device)
        angle = math.radians(self.cfg.image_rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        # Gaze is defined in the upright image frame; rays include mount roll.
        self._gaze_basis = self._gaze_rays.new_tensor(((c, s, 0), (s, -c, 0), (0, 0, -1)))
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        to_opengl = convert_camera_frame_quaternion(identity, origin="world", target="opengl")[0].tolist()
        to_world = convert_camera_frame_quaternion(identity, origin="opengl", target="world")[0].tolist()
        self._gaze_frames = (
            wp.mat33f(*self._gaze_basis.flatten().tolist()),
            wp.quatf(*to_opengl),
            wp.quatf(*to_world),
        )
