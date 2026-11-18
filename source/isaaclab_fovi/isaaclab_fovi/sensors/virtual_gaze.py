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
    record_launch,
    sensor_visibility_kernel,
    set_gaze_kernel,
    torch_stream,
    warp_stream,
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
            self._gaze_rotation_wp = wp.from_torch(self._gaze_rotation, dtype=wp.mat33f)
            self._gaze_pose_delta_wp = wp.from_torch(self._gaze_pose_delta, dtype=wp.quatf)
            self._gaze_angles_key = None
            self._gaze_pose_launches = {}
        # Callers usually pass the same buffer every step; record its launch once.
        key = (angles.data_ptr(), angles.shape, angles.device)
        if key != self._gaze_angles_key:
            self._gaze_angles_wp = wp.from_torch(angles, dtype=wp.vec2f)
            self._gaze_angles_key = key
            self._set_gaze_launch = record_launch(
                set_gaze_kernel,
                len(angles),
                [self._gaze_angles_wp, *self._gaze_frames],
                [self._gaze_rotation_wp, self._gaze_pose_delta_wp],
                angles.device,
            )
        self._set_gaze_launch.launch(warp_stream(angles.device))

    def _apply_gaze_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        if self._gaze_rotation is None:
            return
        orientation = self._data.quat_w_world
        orientation = orientation.warp if hasattr(orientation, "warp") else wp.from_torch(orientation, dtype=wp.quatf)
        delta = self._gaze_pose_delta_wp
        device = self._gaze_pose_delta.device
        if env_ids is not None and env_mask is None:
            if not isinstance(env_ids, wp.array):
                ids = torch.as_tensor(env_ids, dtype=torch.int32, device=device)
                env_ids = wp.from_torch(ids.contiguous())
            with torch_stream(device):
                wp.launch(apply_gaze_indexed_kernel, dim=len(env_ids), inputs=[orientation, delta, env_ids])
            return
        # Every frame updates the same pose buffer, for all environments or under the camera's update mask.
        launch, sources = self._gaze_pose_launches.get(env_mask is None, (None, None))
        if sources is None or sources[0] is not orientation or sources[1] is not env_mask:
            if env_mask is None:
                launch = record_launch(apply_gaze_kernel, len(delta), [orientation, delta], device=device)
            else:
                launch = record_launch(
                    apply_gaze_masked_kernel, len(delta), [orientation, delta, env_mask], device=device
                )
            self._gaze_pose_launches[env_mask is None] = (launch, (orientation, env_mask))
        launch.launch(warp_stream(device))

    def _gaze_validity(self, batch_size: int) -> torch.Tensor:
        """Return which samples the physical camera sees at the current gaze, shape (batch_size, num_samples)."""
        if self._gaze_rotation is None:
            rotation = self._gaze_identity.expand(batch_size, -1, -1).contiguous()
            rotation_wp = wp.from_torch(rotation, dtype=wp.mat33f)
        else:
            rotation, rotation_wp = self._gaze_rotation, self._gaze_rotation_wp
        device = rotation.device
        rays_wp = self._gaze_rays_wp.get(device)
        if rays_wp is None:
            rays_wp = self._gaze_rays_wp[device] = wp.from_torch(self._gaze_rays.to(device), dtype=wp.vec3f)
        valid = torch.empty(batch_size, len(self._gaze_rays), dtype=torch.bool, device=device)
        # A receptive field can overlap the sensor even when its center is outside it.
        with torch_stream(device):
            wp.launch(
                sensor_visibility_kernel,
                dim=valid.shape,
                inputs=[rays_wp, rotation_wp, self._gaze_camera],
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
        self._gaze_rays_wp: dict[torch.device, wp.array] = {}
        self._gaze_angles_key = None
        self._gaze_pose_launches: dict[bool, tuple[wp.Launch, tuple]] = {}
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
