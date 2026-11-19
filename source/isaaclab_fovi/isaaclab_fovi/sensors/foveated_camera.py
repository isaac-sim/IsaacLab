# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foveated camera sensor and its virtual gaze."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils import clone
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.utils.warp import ProxyArray

from .calibrated_geometry import resolve_camera_model
from .gaze_kernels import (
    CameraParams,
    apply_gaze_indexed_kernel,
    apply_gaze_kernel,
    camera_params,
    publish_grid_kernel,
    record_launch,
    set_gaze_kernel,
    warp_stream,
)
from .native_grid import native_grid_directions

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg

OPENGL_TO_ROS_AXES = (1.0, -1.0, -1.0)
"""Axis signs between OpenGL camera-frame directions (x right, y up, -z forward) and ROS ones (x right, y down)."""


class FoveatedCamera(Camera):
    """Camera that samples the scene on a foveated retina instead of a uniform pixel grid.

    The sample layout comes from :class:`fovi.sensing.coords.SamplingCoords`. Constructing this class returns the
    implementation for the renderer selected by :attr:`FoveatedCameraCfg.renderer_cfg`:

    * ``newton_warp`` and ``newton_optix``: :class:`~isaaclab_fovi.sensors.foveated_newton_camera.FoveatedNewtonCamera`,
      which injects the foveated rays into Newton's renderer.
    * ``isaac_rtx`` and ``ovrtx``: :class:`~isaaclab_fovi.sensors.foveated_rtx_camera.FoveatedRTXCamera`, which
      authors a lens lookup table on the camera prim.

    Both are subclasses of this class and are imported on first use, so selecting one renderer does not load the
    other's runtime.

    The gaze turns the retina relative to the optical axis of the physical camera it looks through. Newton and OVRTX
    turn the rendered camera pose by the gaze in each pose update; Isaac RTX writes the turned pose into the camera
    prims, which Replicator reads.
    """

    cfg: FoveatedCameraCfg

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
        if renderer_type in ("newton_warp", "newton_optix"):
            from .foveated_newton_camera import FoveatedNewtonCamera

            return super().__new__(FoveatedNewtonCamera)
        if renderer_type in ("isaac_rtx", "ovrtx"):
            from .foveated_rtx_camera import FoveatedRTXCamera

            return super().__new__(FoveatedRTXCamera)
        raise ValueError(
            f"Unsupported foveated renderer {renderer_type!r}; select 'newton_warp', 'newton_optix', 'isaac_rtx', or"
            " 'ovrtx'."
        )

    def __init__(self, cfg: FoveatedCameraCfg):
        """Create the camera.

        Raises:
            ValueError: A Newton or OVRTX camera without ``update_latest_camera_pose``, whose renderer would never see
                the gaze.
        """
        self._gaze_pose_reads_angles = self._pose_update_turns_gaze(cfg)
        """Whether the camera turns its pose by the stored gaze in each pose update, rather than writing prim poses
        when the gaze is set."""
        if self._gaze_pose_reads_angles and not cfg.update_latest_camera_pose:
            raise ValueError(
                "Newton and OVRTX turn the rendered pose by the gaze in each pose update; set"
                " update_latest_camera_pose=True so the renderer sees the gaze."
            )
        # The parent's initialisation updates poses before the gaze exists.
        self._gaze_angles: ProxyArray | None = None
        self._gaze_rotation: ProxyArray | None = None
        cfg = clone(cfg)
        cfg.width = cfg.height = cfg.foveated_res
        self._camera_model = resolve_camera_model(cfg)
        fov = self._camera_model.field_of_view(cfg.retinal_window_reference_side, cfg.retinal_window_fraction)
        self._sampling_coords = SamplingCoords(
            fov=fov,
            cmf_a=cfg.cmf_a_deg,
            res=cfg.foveated_res,
            style="warped_cartesian_as_grid",
            field_geometry="spherical",
            device="cpu",
        )
        self._retinal_rays = native_grid_directions(self._sampling_coords, image_rotation_deg=cfg.image_rotation_deg)
        super().__init__(cfg)

    @property
    def sampling_coords(self) -> SamplingCoords:
        """The :class:`SamplingCoords` of the published samples."""
        return self._sampling_coords

    @property
    def camera_model(self) -> CameraModel:
        """The physical camera the retina looks through, as resolved at initialization."""
        return self._camera_model

    @property
    def gaze_angles(self) -> ProxyArray:
        """The camera's stored gaze angles [deg], shape (num_envs,) of ``vec2f`` (Torch view (num_envs, 2)).

        A writer that moves the gaze every step can update this buffer in place and pass its Torch view to
        :meth:`set_gaze_angles`, which then copies nothing. Writes take effect when :meth:`set_gaze_angles` is called.
        """
        if self._gaze_angles is None:
            self._allocate_gaze()
        return self._gaze_angles

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
        if stored and self._gaze_pose_reads_angles:
            # The next pose update reads the stored angles; nothing has to happen now.
            return
        count = self._view.count
        if tuple(angles_deg.shape) != (count, 2):
            raise ValueError(f"Gaze angles must have shape ({count}, 2), got {tuple(angles_deg.shape)}.")
        gaze = self.gaze_angles.torch
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

    def set_intrinsic_matrices(self, matrices, focal_length=None, env_ids=None):
        """Keep the camera as configured and warn the caller.

        A foveated camera's projection comes from its retina and :attr:`camera_model`, which pinhole intrinsics
        cannot describe, so this call changes nothing: the rendered rays and the reported intrinsic matrices stay as
        configured. Stock cameras with an OpenCV lens model handle the call the same way. To change the projection,
        configure :attr:`~isaaclab_fovi.sensors.FoveatedCameraCfg.camera_model` (or ``spawn``) and
        :attr:`~isaaclab_fovi.sensors.FoveatedCameraCfg.cmf_a_deg`.

        Args are those of :meth:`isaaclab.sensors.Camera.set_intrinsic_matrices`.
        """
        warnings.warn(
            "set_intrinsic_matrices() leaves a foveated camera as configured: its retina and camera_model set its"
            " projection.",
            UserWarning,
            stacklevel=2,
        )

    def _initialize_gaze(self) -> None:
        """Move the canonical rays to the sensor device and prepare camera-frame conversions."""
        self._sampling_coords = self._sampling_coords.to(device=self._device)
        self._retinal_rays = self._retinal_rays.to(device=self._device)
        self._gaze_angles = None
        self._gaze_rotation = None
        rotation_deg = self.cfg.image_rotation_deg
        angle = math.radians(rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        # Gaze is defined in the upright image frame; the basis carries it into the rolled OpenGL camera frame.
        self._gaze_basis = self._retinal_rays.new_tensor(((c, s, 0), (s, -c, 0), (0, 0, -1)))
        # The same roll in the ROS frame: a rendered ray r reaches the physical camera at K gaze K^T r.
        self._mask_frame = wp.mat33f(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
        # Fixed right-multiplied frame changes, on the CPU in float64, where no reduced-precision matmul applies.
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        to_opengl = convert_camera_frame_orientation_convention(identity, origin="world", target="opengl")[0].tolist()
        to_world = convert_camera_frame_orientation_convention(identity, origin="opengl", target="world")[0].tolist()
        self._gaze_frames = (
            wp.mat33f(*self._gaze_basis.flatten().tolist()),
            wp.quatf(*to_opengl),
            wp.quatf(*to_world),
        )

    def _allocate_gaze(self) -> None:
        """Allocate one stored gaze per camera, on the optical axis, and its rotation on the sensor device."""
        count, device = self._view.count, str(self._device)
        self._gaze_angles = ProxyArray(wp.zeros(count, dtype=wp.vec2f, device=device))
        self._gaze_rotation = ProxyArray(
            wp.from_torch(torch.eye(3, device=device).repeat(count, 1, 1), dtype=wp.mat33f)
        )
        self._gaze_angles_key = None
        self._gaze_pose_launches: dict[str, tuple[wp.Launch, tuple]] = {}
        self._gaze_no_mask = wp.zeros(1, dtype=wp.bool, device=device)

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

    def _apply_gaze_pose(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Turn the camera's world orientation by the stored gaze, and record the gaze rotations."""
        if self._gaze_angles is None:
            return
        orientation = self._data.quat_w_world.warp
        device = self._gaze_angles.torch.device
        frames = (self._gaze_angles, *self._gaze_frames)
        outputs = [orientation, self._gaze_rotation]
        indexed = env_ids is not None and env_mask is None
        if indexed:
            # The kernel takes contiguous int32 indices; this converts sequences, slices, tensors and Warp arrays.
            env_ids = self._resolve_env_ids_wp(env_ids)
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

    def _prepare_grid_publish(self, sources: Mapping[str, torch.Tensor]) -> None:
        """Allocate upright public buffers and bind byte views of the renderer's native grid."""
        coords = self._sampling_coords
        indices = torch.arange(len(coords), dtype=torch.int32, device=self._device)
        self._grid_indices_wp = wp.from_torch(coords.as_grid(indices).reshape(-1).contiguous())
        self._grid_footprint_wp = wp.from_torch(coords.valid_mask.contiguous())
        # Rendered OpenGL rays become physical-camera ROS rays for the calibrated mask.
        rays = self._retinal_rays * self._retinal_rays.new_tensor(OPENGL_TO_ROS_AXES)
        self._grid_rays_wp = wp.from_torch(rays, dtype=wp.vec3f)
        self._grid_camera = camera_params(self._camera_model) if self.cfg.mask_to_sensor_fov else CameraParams()
        self._grid_identity = wp.from_torch(
            torch.eye(3, device=self._device).repeat(self._view.count, 1, 1), dtype=wp.mat33f
        )

        outputs = {}
        self._grid_publish = []
        for name, source in sources.items():
            if name == "rgb" and "rgba" in sources:
                continue
            count, _, _, channels = source.shape
            target = torch.zeros_like(source, memory_format=torch.contiguous_format)
            self._grid_publish.append(
                (
                    wp.from_torch(source.view(count, -1, channels).view(torch.uint8)),
                    wp.from_torch(target.view(count, -1, channels).view(torch.uint8)),
                )
            )
            outputs[name] = target
        if "rgb" in sources and "rgba" in sources:
            outputs["rgb"] = outputs["rgba"][..., :3]
        self._data._output = {name: ProxyArray(wp.from_torch(value)) for name, value in outputs.items()}
        self._grid_launches = None

    def _publish_grid(self, env_mask: wp.array) -> None:
        """Mask and publish the upright grid for the updated environments."""
        rotation = self._grid_identity if self._gaze_rotation is None else self._gaze_rotation.warp
        cached = self._grid_launches
        if cached is None or cached[0] is not rotation or cached[1] is not env_mask:
            inputs = [
                self._grid_indices_wp,
                self._grid_footprint_wp,
                env_mask,
                self._grid_rays_wp,
                rotation,
                self._mask_frame,
                self._grid_camera,
                int(self.cfg.mask_to_sensor_fov),
            ]
            launches = [
                record_launch(publish_grid_kernel, target.shape[:2], [source, *inputs], [target], self._device)
                for source, target in self._grid_publish
            ]
            self._grid_launches = cached = (rotation, env_mask, launches)
        stream = warp_stream(self._device)
        for launch in cached[2]:
            launch.launch(stream)

    @staticmethod
    def _pose_update_turns_gaze(cfg: FoveatedCameraCfg) -> bool:
        """Whether this implementation turns its rendered pose by the stored gaze in each pose update."""
        return False
