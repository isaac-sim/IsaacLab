# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foveated camera sensor and its virtual gaze."""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch
import warp as wp
from fovi.sensing.coords import WARPED_CARTESIAN_STYLES, SamplingCoords
from fovi.sensing.projection import CameraModel

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_error_magnitude
from isaaclab.utils.warp import ProxyArray

from .calibrated_geometry import camera_model_from_spawn, retinal_directions
from .gaze_controls import add_viewer_gaze_controls
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
from .native_grid import GRID_STYLE, buffer_tensor, native_grid_directions
from .pooling import (
    build_pooler,
    build_retinal_coords,
    is_segmentation_output,
    pool_hwc_buffer,
    pool_segmentation_buffer,
    segmentation_neighbor_indices,
)

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg

MOUNT_TURN_TOLERANCE_RAD = 1e-5
"""Smallest change of a camera's mount orientation [rad] that counts as a turn; smaller ones are float rounding."""

OPENGL_TO_ROS_AXES = (1.0, -1.0, -1.0)
"""Axis signs between OpenGL camera-frame directions (x right, y up, -z forward) and ROS ones (x right, y down)."""


class _Retina(NamedTuple):
    """One CMF constant's retina, cached by :meth:`FoveatedCamera.set_cmf_a`."""

    coords: SamplingCoords
    """Sampling coordinates on the sensor device."""
    coords_highres: SamplingCoords | None
    """High-resolution ray coordinates, when the retina pools them."""
    rays: torch.Tensor
    """Rendered rays in the ROS camera frame, shape (N, 3)."""
    render_rays: Any
    """The renderer's form of :attr:`rays`, from :meth:`FoveatedCamera._render_rays`."""
    pooler: Any
    """The pooler of :attr:`coords`, or None."""


def _fixed_count_style(style: str) -> bool:
    """Whether a sampling style keeps its sample count for every CMF constant: the warped-Cartesian styles, whose
    count is the grid's, and the ``_fixn`` isotropic ones, which fix it."""
    return style in WARPED_CARTESIAN_STYLES or "fixn" in style


def _runtime_cmf_a_limitation(camera: FoveatedCamera) -> str:
    """Why ``camera`` cannot change its CMF constant at runtime, for messages and the viewer panel."""
    if not camera._renders_switchable_rays:
        return "OVRTX and Isaac RTX read the retina's lens textures once, when the renderer loads the stage"
    if not _fixed_count_style(camera.cfg.foveated_style):
        return f"a {camera.cfg.foveated_style!r} retina's sample count depends on cmf_a"
    return "the retina can change its cmf_a at runtime"


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

    _renders_switchable_rays: ClassVar[bool] = False
    """Whether the renderer traces rays the camera can replace while it runs, as :meth:`set_cmf_a` needs."""

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

    def __init__(self, cfg: FoveatedCameraCfg, *, turns_pose_by_gaze: bool = False):
        """Create the camera.

        Raises:
            ValueError: A Newton or OVRTX camera without ``update_latest_camera_pose``, whose renderer would never see
                the gaze.
        """
        self._gaze_pose_reads_angles = turns_pose_by_gaze
        """Whether the camera turns its pose by the stored gaze in each pose update, rather than writing prim poses
        when the gaze is set; see :attr:`turns_pose_by_gaze`."""
        if self._gaze_pose_reads_angles and not cfg.update_latest_camera_pose:
            raise ValueError(
                "Newton and OVRTX turn the rendered pose by the gaze in each pose update; set"
                " update_latest_camera_pose=True so the renderer sees the gaze."
            )
        # The parent's initialisation updates poses before the gaze exists.
        self._gaze_written = False
        self._gaze_angles: ProxyArray | None = None
        self._gaze_rotation: ProxyArray | None = None
        self._visibility = None
        self._grid_publish = None
        self._published_shape: tuple[int, int] | None = None
        self._segmentation_neighbors: torch.Tensor | None = None
        self._sensor_fov_valid: torch.Tensor | None = None
        self._pooler = None
        self._retinas: dict[float, _Retina] = {}
        """Retinas built for :meth:`set_cmf_a`, by CMF constant."""
        self.mount_turns = 0
        """How many times :meth:`set_world_poses` or :meth:`set_world_poses_from_view` has turned the camera's mount."""
        super().__init__(cfg)
        # Each Newton viewer window draws the controls while it shows this camera.
        add_viewer_gaze_controls(self)

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) of the published outputs: the retina's grid, or (num_samples, 1) for sample sets."""
        return self._published_shape if self._published_shape is not None else super().image_shape

    @property
    def sampling_coords(self):
        """The :class:`SamplingCoords` of the published samples."""
        return self._sampling_coords

    @property
    def num_valid_samples(self) -> int:
        """Number of published foveated samples."""
        return self._num_valid_samples

    @property
    def sensor_fov_valid_fraction(self) -> torch.Tensor | None:
        """Per-environment fraction of samples inside the physical camera, or None before the first masked frame."""
        return None if self._sensor_fov_valid is None else self._sensor_fov_valid.float().mean(dim=1)

    @property
    def camera_model(self) -> CameraModel:
        """The physical camera the retina looks through, as resolved at initialization."""
        return self._camera_model

    @property
    def projection_params(self) -> CameraParams:
        """:attr:`camera_model` as a Warp struct, for kernels that project camera-frame directions."""
        return self._gaze_camera

    @property
    def image_frame(self) -> wp.mat33f:
        """Rotation from the upright image frame, in which gazes are measured, to the physical camera's frame (ROS
        convention); it undoes ``cfg.image_rotation_deg``."""
        return self._mask_frame

    @property
    def turns_pose_by_gaze(self) -> bool:
        """Whether each pose update turns the rendered pose by the stored gaze (Newton and OVRTX). A writer of
        :attr:`gaze_angles` then needs no other launch."""
        return self._gaze_pose_reads_angles

    @property
    def gaze_angles(self) -> ProxyArray:
        """The camera's stored gaze angles [deg], shape (num_envs,) of ``vec2f`` (Torch view (num_envs, 2)).

        A writer that moves the gaze every step can update this buffer in place and pass its Torch view to
        :meth:`set_gaze_angles`, which then copies nothing. Writes take effect when :meth:`set_gaze_angles` is called.
        """
        if self._gaze_angles is None:
            self._allocate_gaze()
        return self._gaze_angles

    @property
    def mount_quat_w_ros(self) -> ProxyArray:
        """World orientation (x, y, z, w) of the physical camera before the gaze turns it, ROS convention, shape
        (num_envs,) of ``quatf``.

        Refreshed by each pose update that turns the gaze, so it follows a mount that moves, for example a camera on
        a robot's head. Only cameras whose pose update turns the gaze (Newton and OVRTX) record it.
        """
        if not self._gaze_pose_reads_angles:
            raise RuntimeError("Only cameras whose pose update turns the gaze (Newton, OVRTX) record their mount.")
        if self._gaze_angles is None:
            self._allocate_gaze()
        return self._mount_quat_ros

    @property
    def supports_runtime_cmf_a(self) -> bool:
        """Whether :meth:`set_cmf_a` can change this camera's foveation while it runs: on Newton, for retinas whose
        sample count does not depend on the CMF constant (the warped-Cartesian and ``_fixn`` isotropic styles)."""
        return self._renders_switchable_rays and _fixed_count_style(self.cfg.foveated_style)

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
        if stored and self.turns_pose_by_gaze:
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
        self._gaze_written = True

    def ungazed_quat_w_ros(self) -> torch.Tensor:
        """World orientation (x, y, z, w) of the physical camera now, before the gaze turns it, ROS convention, shape
        (num_envs, 4)."""
        if self._gaze_pose_reads_angles and self._gaze_angles is not None:
            # The record starts at the pose before any gaze and follows each pose update.
            return self._mount_quat_ros.torch.clone()
        if self._gaze_written:
            # The prims hold the gaze (Isaac RTX); get_world_poses reports the mount without it.
            return self.get_world_poses(convention="ros")[1]
        return self._data.quat_w_ros.torch.clone()

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
        """Move the camera's mount; the gaze stays relative to the new mount.

        Args are those of :meth:`isaaclab.sensors.Camera.set_world_poses`.
        """
        if orientations is None:
            # A position-only move keeps every mount's orientation.
            super().set_world_poses(positions, orientations, env_ids, convention)
            return
        before = self._mount_orientations(env_ids)
        super().set_world_poses(positions, orientations, env_ids, convention)
        self._mount_written(env_ids)
        self._count_mount_turn(before, env_ids)

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        """Point the camera's mount from ``eyes`` at ``targets``; the gaze stays relative to the new mount.

        Args are those of :meth:`isaaclab.sensors.Camera.set_world_poses_from_view`.
        """
        before = self._mount_orientations(env_ids)
        super().set_world_poses_from_view(eyes, targets, env_ids)
        # Rows whose eye equals their target are skipped by the base method and keep their poses.
        turned = (self._as_device_tensor(eyes, 3) - self._as_device_tensor(targets, 3)).norm(dim=-1) > 0
        self._mount_written(self._env_indices(env_ids)[turned])
        self._count_mount_turn(before, env_ids)

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

    def set_cmf_a(self, cmf_a_deg: float) -> None:
        """Re-sample the retina for a new CMF constant [deg], keeping the gaze, the sample count and every buffer.

        Each constant's retina is built once and cached until the camera is initialized again, so switching back to
        one reuses it.

        Raises:
            RuntimeError: The camera is not initialized.
            NotImplementedError: The renderer or sampling style cannot change the retina at runtime; see
                :attr:`supports_runtime_cmf_a`.
        """
        if not self.is_initialized:
            raise RuntimeError("Initialize the foveated camera (reset the simulation) before changing its cmf_a.")
        if not self.supports_runtime_cmf_a:
            raise NotImplementedError(_runtime_cmf_a_limitation(self))
        cmf_a_deg = float(cmf_a_deg)
        if cmf_a_deg == self.cfg.cmf_a_deg:
            return
        if not self._retinas:
            rays = self._gaze_rays.clone()
            self._retinas[self.cfg.cmf_a_deg] = _Retina(
                self._sampling_coords, self._sampling_coords_highres, rays, self._render_rays(rays), self._pooler
            )
        if cmf_a_deg not in self._retinas:
            self._retinas[cmf_a_deg] = self._build_retina(cmf_a_deg)
        retina = self._retinas[cmf_a_deg]
        self._sampling_coords, self._sampling_coords_highres, self._pooler = (
            retina.coords,
            retina.coords_highres,
            retina.pooler,
        )
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        # The sensor mask, the grid layout and the renderer read the rays in place, so they follow without new launches.
        self._gaze_rays.copy_(retina.rays)
        self._write_render_rays(retina.render_rays)
        if self._segmentation_neighbors is not None:
            self._prepare_segmentation_pooling(["semantic_segmentation"])
        self.cfg.cmf_a_deg = cmf_a_deg

    def _prepare_sampling(self) -> None:
        """Resolve physical calibration and retinal coordinates before the renderer sizes its buffers."""
        calibration = self.cfg.camera_model
        if calibration is None:
            if self.cfg.spawn is None:
                raise ValueError("A foveated camera on an existing prim needs camera_model.")
            calibration = camera_model_from_spawn(self.cfg.spawn)
        self._camera_model = CameraModel(**calibration)
        self._sampling_coords, self._sampling_coords_highres = build_retinal_coords(self.cfg, self._camera_model, "cpu")
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._native_grid = self.cfg.foveated_style == GRID_STYLE
        self._num_valid_samples = len(self._sampling_coords)
        self._num_ray_samples = len(self._ray_coords)

    def _initialize_sampling(self) -> None:
        """Move retinal coordinates to the sensor device and prepare the gaze rays and frame conversions."""
        # Retinas cached by set_cmf_a hold the previous initialization's coordinates, rays and poolers.
        self._retinas = {}
        self._sampling_coords = self._sampling_coords.to(device=self._device)
        if self._sampling_coords_highres is not None:
            self._sampling_coords_highres = self._sampling_coords_highres.to(device=self._device)
        self._ray_coords = self._sampling_coords_highres or self._sampling_coords
        self._gaze_camera = camera_params(self._camera_model)
        rotation_deg = self.cfg.image_rotation_deg
        self._gaze_rays = self._retina_rays(self._ray_coords)
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
        # Fixed right-multiplied frame changes, on the CPU in float64, where no reduced-precision matmul applies.
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        to_opengl = convert_camera_frame_orientation_convention(identity, origin="world", target="opengl")[0].tolist()
        to_world = convert_camera_frame_orientation_convention(identity, origin="opengl", target="world")[0].tolist()
        to_ros = convert_camera_frame_orientation_convention(identity, origin="world", target="ros")[0].tolist()
        self._gaze_frames = (
            wp.mat33f(*self._gaze_basis.flatten().tolist()),
            wp.quatf(*to_opengl),
            wp.quatf(*to_world),
            wp.quatf(*to_ros),
        )

    def _retina_rays(self, coords) -> torch.Tensor:
        """The rendered rays of ``coords``, mount roll included, in the ROS frame the camera model projects, shape
        (N, 3)."""
        rotation_deg = self.cfg.image_rotation_deg
        if coords.style == GRID_STYLE:
            rays = native_grid_directions(coords, image_rotation_deg=rotation_deg)
        else:
            rays = retinal_directions(coords.cartesian, coords.fov, rotation_deg)
        return (rays * rays.new_tensor(OPENGL_TO_ROS_AXES)).float().contiguous()

    def _build_retina(self, cmf_a_deg: float) -> _Retina:
        """The retina at ``cmf_a_deg``: its coordinates on the sensor device, its rays, the renderer's copy of them,
        and its pooler."""
        cfg = self.cfg.replace(cmf_a_deg=cmf_a_deg)
        # fovi reports its isotropic ring fitting on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            lowres, highres = build_retinal_coords(cfg, self._camera_model, "cpu")
        lowres = lowres.to(device=self._device)
        highres = None if highres is None else highres.to(device=self._device)
        ray_coords = highres or lowres
        if len(ray_coords) != self._num_ray_samples or not torch.equal(
            ray_coords.valid_mask, self._ray_coords.valid_mask
        ):
            raise NotImplementedError(
                f"cmf_a {cmf_a_deg:g} changes this retina's samples; {_runtime_cmf_a_limitation(self)}"
            )
        pooler, _ = build_pooler(cfg, lowres, highres, device=str(self._device))
        rays = self._retina_rays(ray_coords)
        return _Retina(lowres, highres, rays, self._render_rays(rays), pooler)

    def _render_rays(self, rays: torch.Tensor) -> Any:
        """The renderer's form of ``rays`` (ROS frame), which :meth:`_write_render_rays` hands it; only renderers that
        trace the camera's rays support it."""
        raise NotImplementedError(_runtime_cmf_a_limitation(self))

    def _write_render_rays(self, render_rays: Any) -> None:
        """Hand the renderer rays made by :meth:`_render_rays`."""
        raise NotImplementedError(_runtime_cmf_a_limitation(self))

    def _allocate_gaze(self) -> None:
        """Allocate one stored gaze per camera, on the optical axis, its rotation and the mount, on the sensor
        device."""
        count, device = self._view.count, str(self._device)
        self._gaze_angles = ProxyArray(wp.zeros(count, dtype=wp.vec2f, device=device))
        self._gaze_rotation = ProxyArray(
            wp.from_torch(torch.eye(3, device=device).repeat(count, 1, 1), dtype=wp.mat33f)
        )
        # No gaze has turned the pose yet, so the current orientation is the mount.
        mount = self._data.quat_w_ros.torch.clone()
        self._mount_quat_ros = ProxyArray(wp.from_torch(mount, dtype=wp.quatf))
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
        """Turn the camera's world orientation by the stored gaze, and record the gaze rotations and the mount."""
        if self._gaze_angles is None:
            return
        orientation = self._data.quat_w_world.warp
        device = self._gaze_angles.torch.device
        frames = (self._gaze_angles, *self._gaze_frames)
        outputs = [orientation, self._gaze_rotation, self._mount_quat_ros]
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

    def _mount_orientations(self, env_ids) -> torch.Tensor:
        """The selected cameras' mount orientations (x, y, z, w), world convention, as :meth:`get_world_poses` reports
        them."""
        return self.get_world_poses(env_ids, convention="world")[1].clone()

    def _mount_written(self, env_ids) -> None:
        """Called after a pose write turned the selected cameras' mounts, before the turn is measured. Cameras that
        hold the gaze in their prims write it onto the new mounts here."""

    def _count_mount_turn(self, before: torch.Tensor, env_ids) -> None:
        """Count a mount turn if any selected camera's mount now differs from ``before``.

        The teachers read the camera's live position, so only orientation changes count, and writing back the reported
        poses does not.
        """
        turned = quat_error_magnitude(before, self._mount_orientations(env_ids)) > MOUNT_TURN_TOLERANCE_RAD
        if bool(turned.any()):
            self.mount_turns += 1

    def _env_indices(self, env_ids) -> torch.Tensor:
        """The camera indices that ``env_ids`` selects, as a tensor."""
        if env_ids is None:
            return torch.arange(self._view.count, device=self._device)
        if isinstance(env_ids, slice):
            return torch.arange(self._view.count, device=self._device)[env_ids]
        if isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long).reshape(-1)

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
        tensors = {name: buffer_tensor(value) for name, value in outputs.items()}
        self._sensor_fov_valid = self._gaze_validity(env_mask)
        hidden = self._visibility["hidden"].torch[:, :, None, None]
        for name, tensor in tensors.items():
            if name == "rgb" and "rgba" in tensors:
                continue
            tensor.masked_fill_(hidden, 0)

    def _prepare_segmentation_pooling(self, names) -> None:
        """Take the modal-label neighbourhoods of pooled segmentation outputs among ``names`` from the main pooler."""
        self._segmentation_neighbors = (
            segmentation_neighbor_indices(self._pooler) if any(is_segmentation_output(name) for name in names) else None
        )

    def _pool_output(self, name: str, samples: torch.Tensor) -> torch.Tensor:
        """Pool one output's high-resolution samples (B, N_hi, 1, C) to the retina's (B, N_lo, 1, C), in its dtype.

        Segmentation takes each sample's modal label in its native dtype; other outputs are pooled in float with
        ``pool_mode``. With :attr:`~FoveatedCameraCfg.mask_to_sensor_fov`, photoreceptors outside the physical camera's
        image at the current gaze are left out, so a pooled sample at the image edge is not darkened by them.
        """
        visible = self._visibility["valid"].torch if self.cfg.mask_to_sensor_fov else None
        if is_segmentation_output(name):
            return pool_segmentation_buffer(self._segmentation_neighbors, samples, visible)
        pooled = pool_hwc_buffer(self._pooler, samples.float(), visible)
        if not samples.dtype.is_floating_point:
            # Round to the nearest value the output can hold, e.g. 0-255 for RGB, rather than truncating.
            limits = torch.iinfo(samples.dtype)
            pooled = pooled.round().clamp(limits.min, limits.max)
        return pooled.to(samples.dtype)

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

    def _set_published_shape(self, outputs: Mapping[str, torch.Tensor]) -> None:
        """Report the (height, width) of the published outputs, which differ from the rendered ones."""
        tensor = next(iter(outputs.values())).torch
        self._published_shape = (int(tensor.shape[1]), int(tensor.shape[2]))
        self._data.image_shape = self._published_shape
