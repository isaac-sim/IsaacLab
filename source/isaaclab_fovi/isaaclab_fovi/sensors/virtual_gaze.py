# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual gaze: the retina turns inside a fixed physical camera."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import torch
import warp as wp
from fovi.sensing.projection import angular_directions

from .calibrated_geometry import convert_camera_frame_quaternion, resolve_camera_model
from .gaze_kernels import (
    apply_gaze_indexed_kernel,
    apply_gaze_kernel,
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

    _gaze_pose_reads_angles: ClassVar[bool] = False
    """Whether the camera's per-frame pose update turns the gaze from its stored angles, so that a writer of
    :meth:`gaze_angles_buffer` needs no other launch."""

    def gaze_angles_buffer(self) -> torch.Tensor:
        """The camera's stored gaze angles [deg], shape (num_envs, 2).

        A writer that moves the gaze every step can update this buffer in place and pass it to
        :meth:`set_gaze_angles`, which then copies nothing. Writes take effect when :meth:`set_gaze_angles` is called.
        """
        if self._gaze_angles is None:
            self._allocate_gaze(self._view.count, torch.device(self._device))
        return self._gaze_angles

    def set_gaze_angles(self, angles_deg: torch.Tensor) -> None:
        """Turn the retina to (tilt, pan) [deg] from the physical camera's optical axis.

        Positive tilt looks up and positive pan looks left. Zero looks along the
        optical axis.

        Args:
            angles_deg: Gaze angles for every environment [deg], shape (num_envs, 2).
        """
        stored = angles_deg is self._gaze_angles and angles_deg is not None
        if stored and self._gaze_pose_reads_angles and self.cfg.update_latest_camera_pose:
            # The next pose update reads the stored angles; nothing has to happen now.
            return
        angles = angles_deg if stored else angles_deg.detach().float().contiguous()
        if (
            self._gaze_angles is None
            or len(self._gaze_angles) != len(angles)
            or self._gaze_angles.device != angles.device
        ):
            self._allocate_gaze(len(angles), angles.device)
        # Callers usually pass the same buffer every step; record its launch once.
        key = (angles.data_ptr(), angles.shape, angles.device)
        if key != self._gaze_angles_key:
            self._gaze_angles_key = key
            self._set_gaze_launch = record_launch(
                set_gaze_kernel,
                len(angles),
                [wp.from_torch(angles, dtype=wp.vec2f)],
                [self._gaze_angles_wp, self._gaze_rotation_wp],
                angles.device,
            )
        self._set_gaze_launch.launch(warp_stream(angles.device))

    def _allocate_gaze(self, count: int, device: torch.device) -> None:
        """Allocate the stored gaze, on the optical axis, and its rotation."""
        self._gaze_angles = torch.zeros(count, 2, device=device)
        self._gaze_rotation = torch.eye(3, device=device).repeat(count, 1, 1)
        self._gaze_angles_wp = wp.from_torch(self._gaze_angles, dtype=wp.vec2f)
        self._gaze_rotation_wp = wp.from_torch(self._gaze_rotation, dtype=wp.mat33f)
        self._gaze_angles_key = None
        self._gaze_pose_launches: dict[str, tuple[wp.Launch, tuple]] = {}
        self._gaze_no_mask = wp.zeros(1, dtype=wp.bool, device=str(device))

    def _apply_gaze_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Turn the camera's world orientation by the stored gaze, and record the gaze rotations it uses."""
        if self._gaze_angles is None:
            return
        orientation = self._data.quat_w_world
        orientation = orientation.warp if hasattr(orientation, "warp") else wp.from_torch(orientation, dtype=wp.quatf)
        device = self._gaze_angles.device
        frames = (self._gaze_angles_wp, *self._gaze_frames)
        indexed = env_ids is not None and env_mask is None
        if indexed and not isinstance(env_ids, wp.array):
            env_ids = wp.from_torch(torch.as_tensor(env_ids, dtype=torch.int32, device=device).contiguous())
        # Every pose update writes the same pose buffer: for all environments, under the camera's update mask, or
        # for the indexed ones, whose indices each call swaps in.
        mode = "ids" if indexed else ("all" if env_mask is None else "mask")
        launch, sources = self._gaze_pose_launches.get(mode, (None, None))
        if sources is None or sources[0] is not orientation or (mode == "mask" and sources[1] is not env_mask):
            if indexed:
                launch = record_launch(
                    apply_gaze_indexed_kernel,
                    len(env_ids),
                    [*frames, env_ids],
                    [orientation, self._gaze_rotation_wp],
                    device,
                )
            else:
                launch = record_launch(
                    apply_gaze_kernel,
                    len(self._gaze_angles),
                    [*frames, self._gaze_no_mask if env_mask is None else env_mask, int(env_mask is not None)],
                    [orientation, self._gaze_rotation_wp],
                    device,
                )
            self._gaze_pose_launches[mode] = (launch, (orientation, env_mask))
        if indexed:
            if len(env_ids) == 0:
                return
            launch.set_param_by_name("indices", env_ids)
            launch.set_dim(len(env_ids))
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
        # No gaze until one is set: the camera looks along its optical axis untouched.
        self._gaze_angles = None
        self._gaze_rotation = None
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
