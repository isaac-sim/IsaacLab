# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foveated camera sensor and its virtual gaze."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.warp import ProxyArray

from .calibrated_geometry import convert_camera_frame_quaternion, resolve_camera_model, retinal_directions
from .gaze_kernels import (
    CameraParams,
    apply_gaze_indexed_kernel,
    apply_gaze_kernel,
    camera_params,
    publish_grid_kernel,
    record_launch,
    sensor_visibility_kernel,
    set_gaze_kernel,
    warp_stream,
)
from .native_grid import GRID_STYLE, native_grid_directions
from .pooling import ModalPooler, is_segmentation_output, pool_hwc_buffer, pool_segmentation_buffer

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg

OPENGL_TO_ROS_AXES = (1.0, -1.0, -1.0)
"""Axis signs between OpenGL camera-frame directions (x right, y up, -z forward) and ROS ones (x right, y down)."""


class FoveatedCamera(Camera):
    """Camera that samples the scene on a foveated retina instead of a uniform pixel grid.

    The sample layout comes from :class:`fovi.sensing.coords.SamplingCoords`. Constructing this class returns the
    implementation for the renderer selected by :attr:`FoveatedCameraCfg.renderer_cfg`:

    * ``newton_warp``: :class:`~isaaclab_fovi.sensors.foveated_newton_camera.FoveatedNewtonCamera`, which injects
      the foveated rays into the Newton Warp renderer.
    * ``isaac_rtx`` and ``ovrtx``: :class:`~isaaclab_fovi.sensors.foveated_rtx_camera.FoveatedRTXCamera`, which
      authors a lens lookup table on the camera prim.

    Both are subclasses of this class and are imported on first use, so selecting one renderer does not load the
    other's runtime.

    The gaze turns the retina relative to the optical axis of the physical camera it looks through. Newton and OVRTX
    turn the rendered camera pose by the gaze in each pose update; Isaac RTX writes the turned pose into the camera
    prims, which Replicator reads.
    """

    cfg: FoveatedCameraCfg

    _gaze_pose_reads_angles: bool = False
    """Whether the camera turns its pose by the stored gaze in each pose update, rather than writing prim poses when
    the gaze is set. A writer of :meth:`gaze_angles_buffer` then needs no other launch."""

    def __new__(cls, cfg: FoveatedCameraCfg, *args, **kwargs):
        """Create the implementation that matches the configured renderer.

        Args:
            cfg: Camera configuration with a concrete renderer configuration.

        Raises:
            ValueError: The configured renderer has no foveated implementation.
        """
        if cls is not FoveatedCamera:
            return super().__new__(cls)
        renderer_type = cfg.renderer_cfg.renderer_type
        if renderer_type == "newton_warp":
            from .foveated_newton_camera import FoveatedNewtonCamera

            return super().__new__(FoveatedNewtonCamera)
        if renderer_type in ("isaac_rtx", "ovrtx"):
            from .foveated_rtx_camera import FoveatedRTXCamera

            return super().__new__(FoveatedRTXCamera)
        raise ValueError(
            f"Unsupported foveated renderer {renderer_type!r}; select 'newton_warp', 'isaac_rtx', or 'ovrtx'."
        )

    def __init__(self, cfg: FoveatedCameraCfg):
        # The parent's initialisation updates poses before the gaze exists.
        self._gaze_angles: ProxyArray | None = None
        self._gaze_rotation: ProxyArray | None = None
        self._visibility = None
        self._grid_publish = None
        self._published_shape: tuple[int, int] | None = None
        self.mount_moved = False
        """Whether :meth:`set_world_poses` or :meth:`set_world_poses_from_view` has moved the camera's mount since
        construction."""
        super().__init__(cfg)

    # ------------------------------------------------------------------
    #  Gaze
    # ------------------------------------------------------------------

    def gaze_angles_buffer(self) -> torch.Tensor:
        """The camera's stored gaze angles [deg], shape (num_envs, 2).

        A writer that moves the gaze every step can update this buffer in place and pass it to
        :meth:`set_gaze_angles`, which then copies nothing. Writes take effect when :meth:`set_gaze_angles` is called.
        """
        if self._gaze_angles is None:
            self._allocate_gaze()
        return self._gaze_angles.torch

    def set_gaze_angles(self, angles_deg: torch.Tensor) -> None:
        """Turn the retina to (tilt, pan) [deg] from the physical camera's optical axis.

        Positive tilt looks up and positive pan looks left. Zero looks along the
        optical axis.

        Args:
            angles_deg: Gaze angles for every environment [deg], shape (num_envs, 2).

        Raises:
            ValueError: The angles do not hold one row per camera.
        """
        stored = self._gaze_angles is not None and angles_deg is self._gaze_angles.torch
        if stored and self._gaze_pose_reads_angles and self.cfg.update_latest_camera_pose:
            # The next pose update reads the stored angles; nothing has to happen now.
            return
        count = self._view.count
        if tuple(angles_deg.shape) != (count, 2):
            raise ValueError(f"Gaze angles must have shape ({count}, 2), got {tuple(angles_deg.shape)}.")
        gaze = self.gaze_angles_buffer()
        angles = angles_deg if stored else angles_deg.detach().to(device=gaze.device, dtype=torch.float32).contiguous()
        # Callers usually pass the same buffer every step; record its launch once.
        key = angles.data_ptr()
        if key != self._gaze_angles_key:
            self._gaze_angles_key = key
            self._set_gaze_launch = record_launch(
                set_gaze_kernel,
                count,
                [wp.from_torch(angles, dtype=wp.vec2f)],
                [self._gaze_angles, self._gaze_rotation],
                gaze.device,
            )
        self._set_gaze_launch.launch(warp_stream(gaze.device))

    @property
    def mount_quat_w_ros(self) -> torch.Tensor:
        """World orientation (x, y, z, w) of the physical camera before the gaze turns it, ROS convention, shape
        (num_envs, 4).

        Refreshed by each pose update that turns the gaze, so it follows a mount that moves, for example a camera on
        a robot's head. Only cameras whose pose update turns the gaze (Newton and OVRTX) record it.
        """
        if not self._gaze_pose_reads_angles:
            raise RuntimeError("Only cameras whose pose update turns the gaze (Newton, OVRTX) record their mount.")
        self.gaze_angles_buffer()
        return self._mount_quat_ros.torch

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
        """Move the camera's mount; the gaze stays relative to the new mount.

        Args are those of :meth:`isaaclab.sensors.Camera.set_world_poses`.
        """
        super().set_world_poses(positions, orientations, env_ids, convention)
        # The teachers read the camera's live position, so only a turned mount counts as moved.
        if orientations is not None:
            self.mount_moved = True

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        """Point the camera's mount from ``eyes`` at ``targets``; the gaze stays relative to the new mount.

        Args are those of :meth:`isaaclab.sensors.Camera.set_world_poses_from_view`.
        """
        super().set_world_poses_from_view(eyes, targets, env_ids)
        if self._turned_by_view(eyes, targets).any():
            self.mount_moved = True

    def _turned_by_view(self, eyes, targets) -> torch.Tensor:
        """Which rows :meth:`set_world_poses_from_view` turns: the base method skips an eye equal to its target."""
        return (self._as_device_tensor(eyes, 3) - self._as_device_tensor(targets, 3)).norm(dim=-1) > 0

    def _allocate_gaze(self) -> None:
        """Allocate one stored gaze per camera, on the optical axis, its rotation and the mount, on the sensor
        device."""
        count, device = self._view.count, str(self._device)
        self._gaze_angles = ProxyArray(wp.zeros(count, dtype=wp.vec2f, device=device))
        self._gaze_rotation = ProxyArray(
            wp.from_torch(torch.eye(3, device=device).repeat(count, 1, 1), dtype=wp.mat33f)
        )
        self._mount_quat_ros = ProxyArray(wp.zeros(count, dtype=wp.quatf, device=device))
        self._gaze_angles_key = None
        self._gaze_pose_launches: dict[str, tuple[wp.Launch, tuple]] = {}
        self._gaze_no_mask = wp.zeros(1, dtype=wp.bool, device=device)

    def _apply_gaze_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Turn the camera's world orientation by the stored gaze, and record the gaze rotations and the mount."""
        if self._gaze_angles is None:
            return
        orientation = self._data.quat_w_world
        orientation = orientation.warp if hasattr(orientation, "warp") else wp.from_torch(orientation, dtype=wp.quatf)
        device = self._gaze_angles.torch.device
        frames = (self._gaze_angles, *self._gaze_frames)
        outputs = [orientation, self._gaze_rotation, self._mount_quat_ros]
        indexed = env_ids is not None and env_mask is None
        if indexed and not isinstance(env_ids, wp.array):
            env_ids = wp.from_torch(torch.as_tensor(env_ids, dtype=torch.int32, device=device).contiguous())
        # Every pose update writes the same pose buffer: for all environments, under the camera's update mask, or
        # for the indexed ones, whose indices each call swaps in.
        mode = "ids" if indexed else ("all" if env_mask is None else "mask")
        launch, sources = self._gaze_pose_launches.get(mode, (None, None))
        if sources is None or sources[0] is not orientation or (mode == "mask" and sources[1] is not env_mask):
            if indexed:
                launch = record_launch(apply_gaze_indexed_kernel, len(env_ids), [*frames, env_ids], outputs, device)
            else:
                mask = [self._gaze_no_mask if env_mask is None else env_mask, int(env_mask is not None)]
                launch = record_launch(apply_gaze_kernel, self._view.count, [*frames, *mask], outputs, device)
            self._gaze_pose_launches[mode] = (launch, (orientation, env_mask))
        if indexed:
            if len(env_ids) == 0:
                return
            launch.set_param_by_name("indices", env_ids)
            launch.set_dim(len(env_ids))
        launch.launch(warp_stream(device))

    def _update_camera_state(
        self,
        env_ids: wp.array | None = None,
        env_mask: wp.array | None = None,
        pos_src: wp.array | None = None,
        quat_src: wp.array | None = None,
        update_pose: bool = False,
        frame_op: int = 0,
    ):
        """Update the camera state; when the pose update turns the gaze, turn freshly read poses before the renderer
        sees them."""
        super()._update_camera_state(env_ids, env_mask, pos_src, quat_src, update_pose, frame_op)
        if update_pose and self._gaze_pose_reads_angles:
            self._apply_gaze_pose(env_ids, env_mask)

    def _initialize_gaze(self) -> None:
        """Prepare the gaze geometry: the rendered rays in the physical camera's frame and the pose conversions."""
        self._camera_model = resolve_camera_model(self.cfg)
        self._gaze_camera = camera_params(self._camera_model)
        coords = self._sampling_coords_highres or self._sampling_coords
        rotation_deg = self.cfg.image_rotation_deg
        # The rays as rendered, mount roll included, in the ROS frame the camera model projects.
        if coords.style == GRID_STYLE:
            rays = native_grid_directions(coords, image_rotation_deg=rotation_deg)
        else:
            rays = retinal_directions(coords.cartesian, coords.fov, rotation_deg)
        self._gaze_rays = (rays * rays.new_tensor(OPENGL_TO_ROS_AXES)).float().contiguous()
        # No gaze until one is set: the camera looks along its optical axis untouched.
        self._gaze_angles = None
        self._gaze_rotation = None
        self._visibility = None
        device = self._gaze_rays.device
        self._gaze_identity = torch.eye(3, device=device)
        angle = math.radians(rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        # Gaze is defined in the upright image frame; the basis carries it into the rolled OpenGL camera frame.
        self._gaze_basis = self._gaze_rays.new_tensor(((c, s, 0), (s, -c, 0), (0, 0, -1)))
        # The same roll in the ROS frame: a rendered ray r reaches the physical camera at K gaze K^T r.
        self._mask_frame = wp.mat33f(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        opengl = convert_camera_frame_quaternion(identity, origin="world", target="opengl")
        to_opengl = opengl[0].tolist()
        to_world = convert_camera_frame_quaternion(identity, origin="opengl", target="world")[0].tolist()
        to_ros = convert_camera_frame_quaternion(opengl, origin="opengl", target="ros")[0].tolist()
        self._gaze_frames = (
            wp.mat33f(*self._gaze_basis.flatten().tolist()),
            wp.quatf(*to_opengl),
            wp.quatf(*to_world),
            wp.quatf(*to_ros),
        )

    # ------------------------------------------------------------------
    #  Sensor mask and publication
    # ------------------------------------------------------------------

    def _visibility_buffers(self) -> dict:
        """The sensor mask's camera-owned buffers, allocated once: the rendered rays, an unturned rotation per camera,
        each sample's visibility and, for the updated rows, which samples to hide."""
        if self._visibility is None:
            count, device = self._view.count, self._gaze_rays.device
            shape = (count, len(self._gaze_rays))
            self._visibility = {
                "rays": wp.from_torch(self._gaze_rays, dtype=wp.vec3f),
                "identity": wp.from_torch(self._gaze_identity.expand(count, -1, -1).contiguous(), dtype=wp.mat33f),
                "every_env": wp.ones(count, dtype=wp.bool, device=str(device)),
                "valid": ProxyArray(wp.zeros(shape, dtype=wp.bool, device=str(device))),
                "hidden": ProxyArray(wp.zeros(shape, dtype=wp.bool, device=str(device))),
                "launch": None,
            }
        return self._visibility

    def _gaze_validity(self, env_mask: wp.array | None = None) -> torch.Tensor:
        """Which samples the physical camera sees at each updated environment's gaze, shape (num_envs, num_samples).

        Each rendered ray is tested by its center direction. Environments outside ``env_mask`` keep their last
        visibility; None updates every one.
        """
        buffers = self._visibility_buffers()
        rotation = buffers["identity"] if self._gaze_rotation is None else self._gaze_rotation.warp
        updated = buffers["every_env"] if env_mask is None else env_mask
        cached = buffers["launch"]
        if cached is None or cached[0] is not rotation or cached[1] is not updated:
            launch = record_launch(
                sensor_visibility_kernel,
                buffers["valid"].warp.shape,
                [buffers["rays"], rotation, self._mask_frame, self._gaze_camera, updated],
                [buffers["valid"], buffers["hidden"]],
                self._gaze_rays.device,
            )
            buffers["launch"] = cached = (rotation, updated, launch)
        cached[2].launch(warp_stream(self._gaze_rays.device))
        return buffers["valid"].torch

    def _apply_sensor_fov_mask(self, outputs: Mapping[str, torch.Tensor], env_mask: wp.array | None = None) -> None:
        """Zero the samples outside the physical camera at the current gaze in the updated environments' (B, N, 1, C)
        outputs; None updates every environment."""
        if not self.cfg.mask_to_sensor_fov:
            return
        tensors = {name: value.torch if hasattr(value, "torch") else value for name, value in outputs.items()}
        self._sensor_fov_valid = self._gaze_validity(env_mask)
        hidden = self._visibility["hidden"].torch[:, :, None, None]
        for name, tensor in tensors.items():
            if name == "rgb" and "rgba" in tensors:
                continue
            tensor.masked_fill_(hidden, 0)

    def _prepare_grid_publish(self, sources: Mapping[str, torch.Tensor]) -> None:
        """Byte views of each rendered output, shape (num_envs, num_samples, channels), and of its public grid, for
        :func:`publish_grid_kernel`."""
        views = []
        for name, source in sources.items():
            if name == "rgb" and "rgba" in sources:
                continue
            target = self._grid_output[name].view(source.shape[0], -1, source.shape[-1])
            views.append((wp.from_torch(source.view(torch.uint8)), wp.from_torch(target.view(torch.uint8))))
        self._grid_indices_wp = wp.from_torch(self._grid_indices.to(torch.int32).contiguous())
        self._grid_footprint_wp = wp.from_torch(self._ray_coords.valid_mask.contiguous())
        num_envs = views[0][1].shape[0]
        if self.cfg.mask_to_sensor_fov:
            buffers = self._visibility_buffers()
            self._grid_every_env = buffers["every_env"]
            self._grid_sensor = (
                buffers["rays"],
                buffers["identity"],
                self._mask_frame,
                self._gaze_camera,
                buffers["valid"].warp,
            )
        else:
            self._grid_every_env = wp.ones(num_envs, dtype=wp.bool, device=str(self._device))
            # Placeholders: the kernel reads none of these without the sensor mask.
            self._grid_sensor = (
                wp.zeros(1, dtype=wp.vec3f, device=self._device),
                wp.zeros(1, dtype=wp.mat33f, device=self._device),
                wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                CameraParams(),
                wp.zeros((1, 1), dtype=wp.bool, device=self._device),
            )
        self._grid_publish = views
        self._grid_launches = None

    def _publish_grid(self, env_mask: wp.array | None = None) -> None:
        """Mask and lay out every output of the updated environments in one launch each; the result equals masking,
        then copying to the grid. The other environments keep their last published frame; None updates every one."""
        mask = self.cfg.mask_to_sensor_fov
        env_mask = self._grid_every_env if env_mask is None else env_mask
        rays, identity, frame, camera, visible = self._grid_sensor
        rotation = identity if not mask or self._gaze_rotation is None else self._gaze_rotation.warp
        cached = self._grid_launches
        if cached is None or cached[0][0] is not rotation or cached[0][1] is not env_mask:
            launches = [
                record_launch(
                    publish_grid_kernel,
                    (target.shape[0], target.shape[1]),
                    [
                        source,
                        self._grid_indices_wp,
                        self._grid_footprint_wp,
                        env_mask,
                        rays,
                        rotation,
                        frame,
                        camera,
                        int(mask),
                        int(mask and index == 0),
                        visible,
                    ],
                    [target],
                    self._device,
                )
                for index, (source, target) in enumerate(self._grid_publish)
            ]
            self._grid_launches = ((rotation, env_mask), launches)
        stream = warp_stream(self._device)
        for launch in self._grid_launches[1]:
            launch.launch(stream)
        if mask:
            self._sensor_fov_valid = self._visibility["valid"].torch

    def _pool_output(self, name: str, samples: torch.Tensor) -> torch.Tensor:
        """Pool one output's high-resolution samples (B, N_hi, 1, C) to the retina's (B, N_lo, 1, C), in its dtype.

        Segmentation takes each sample's modal label in its native dtype; other outputs are pooled in float.
        """
        if self._pooler_int is not None and is_segmentation_output(name):
            if isinstance(self._pooler_int, ModalPooler):
                return pool_segmentation_buffer(self._pooler_int, samples)
            return pool_hwc_buffer(self._pooler_int, samples.float()).to(samples.dtype)
        return pool_hwc_buffer(self._pooler, samples.float()).to(samples.dtype)

    # ------------------------------------------------------------------
    #  Output description
    # ------------------------------------------------------------------

    def _set_published_shape(self, outputs: Mapping[str, torch.Tensor]) -> None:
        """Report the (height, width) of the published outputs, which differ from the rendered ones."""
        first = next(iter(outputs.values()))
        tensor = first.torch if hasattr(first, "torch") else first
        self._published_shape = (int(tensor.shape[1]), int(tensor.shape[2]))
        self._data.image_shape = self._published_shape

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) of the published outputs: the retina's grid, or (num_samples, 1) for sample sets."""
        return self._published_shape if self._published_shape is not None else super().image_shape

    @property
    def sampling_coords(self):
        """The :class:`SamplingCoords` of the published samples."""
        return self._sampling_coords

    @property
    def image_rotation_deg(self) -> float:
        """Roll [deg] of the rendered rays around the optical axis, which undoes a rolled mount."""
        return float(self.cfg.image_rotation_deg)

    @property
    def num_valid_samples(self) -> int:
        """Number of published foveated samples."""
        return self._num_valid_samples

    @property
    def sensor_fov_valid_fraction(self) -> torch.Tensor | None:
        """Per-environment fraction of samples inside the physical camera, or None before the first masked frame."""
        valid = getattr(self, "_sensor_fov_valid", None)
        return None if valid is None else valid.float().mean(dim=1)
