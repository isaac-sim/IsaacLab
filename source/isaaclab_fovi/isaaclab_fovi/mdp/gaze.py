# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active gaze for a foveated camera: a gaze action, its observation, and a teacher's gaze labels.

Gaze angles are (tilt, pan) [deg] from the physical camera's optical axis: positive tilt looks up and positive pan
looks left, as for :meth:`~isaaclab_fovi.sensors.FoveatedCamera.set_gaze_angles`. The policy moves and observes the
gaze in field radii, half the retina's field of view. Camera-frame directions follow the ROS (OpenCV) convention:
x right, y down, z forward.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import torch
import warp as wp

from isaaclab.managers import ActionTerm, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import ProxyArray

from ..sensors.calibrated_geometry import convert_camera_frame_quaternion
from ..sensors.gaze_kernels import (
    DEG_TO_RAD,
    CameraParams,
    project_direction,
    project_directions,
    record_launch,
    warp_stream,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .gaze_cfg import GazeActionCfg

RAD_TO_DEG = 180.0 / math.pi


@wp.func
def teach(
    point: wp.vec3f,
    position: wp.vec3f,
    mount: wp.quatf,
    g: wp.vec2f,
    frame: wp.mat33f,
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
) -> wp.vec4f:
    """The teacher's action (x, y) toward a point, 1 for a valid target or else 0, and the gaze's error [deg].

    ``frame`` carries the upright image frame, in which gazes are measured, into the physical camera's frame.
    """
    d = wp.normalize(wp.quat_rotate_inv(mount, point - position))
    # The fixation reaches the physical camera at frame * gaze * z, so the gaze looks along frame^T d.
    u = wp.transpose(frame) * d
    tilt = -wp.atan2(u[1], wp.sqrt(u[0] * u[0] + u[2] * u[2])) * wp.float32(RAD_TO_DEG)
    pan = -wp.atan2(u[0], u[2]) * wp.float32(RAD_TO_DEG)
    desired = wp.vec2f(tilt, pan) / field_radius_deg
    ok = project_direction(d, camera)[2] > 0.5 and wp.length(desired) <= gaze_limit
    step = wp.vec2f(0.0, 0.0)
    if ok:
        step = wp.vec2f(
            wp.clamp((desired[0] - g[0]) / step_size, -1.0, 1.0), wp.clamp((desired[1] - g[1]) / step_size, -1.0, 1.0)
        )
    g_tilt = g[0] * field_radius_deg * wp.float32(DEG_TO_RAD)
    g_pan = g[1] * field_radius_deg * wp.float32(DEG_TO_RAD)
    looking = wp.vec3f(-wp.cos(g_tilt) * wp.sin(g_pan), -wp.sin(g_tilt), wp.cos(g_tilt) * wp.cos(g_pan))
    # atan2 stays accurate near perfect alignment, unlike acos of the dot product.
    error = wp.atan2(wp.length(wp.cross(looking, u)), wp.dot(looking, u)) * wp.float32(RAD_TO_DEG)
    return wp.vec4f(step[0], step[1], wp.where(ok, 1.0, 0.0), error)


@wp.kernel(enable_backward=False)
def teacher_labels_kernel(
    targets: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    offset: wp.vec3f,
    rotate_offset: wp.int32,
    positions: wp.array(dtype=wp.vec3f),
    mounts: wp.array(dtype=wp.quatf),
    state: wp.array(dtype=wp.vec4f),
    frame: wp.mat33f,
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
    labels: wp.array2d(dtype=wp.float32),
    log: wp.array(dtype=wp.float32),
):
    """Labels (action x, action y, valid) toward a point fixed to each target; and the mean gaze error to valid
    targets [deg] with the fraction of valid targets.

    ``mounts`` holds each camera's mount (ROS convention); ``state`` each gaze, in field radii, and its velocity.
    One block labels every environment and reduces the logs itself, so the launch needs no zeroed accumulators.
    """
    _block, lane = wp.tid()
    count = targets.shape[0]
    error_sum = wp.float32(0.0)
    valid_count = wp.float32(0.0)
    i = lane
    while i < count:
        point = targets[i]
        if rotate_offset != 0:
            point = point + wp.quat_rotate(orientations[i], offset)
        s = state[i]
        gaze = wp.vec2f(s[0], s[1])
        result = teach(point, positions[i], mounts[i], gaze, frame, field_radius_deg, step_size, gaze_limit, camera)
        labels[i, 0] = result[0]
        labels[i, 1] = result[1]
        labels[i, 2] = result[2]
        if result[2] > 0.5:
            error_sum += result[3]
            valid_count += 1.0
        i += wp.block_dim()
    errors = wp.tile_sum(wp.tile(error_sum))
    valid = wp.tile_sum(wp.tile(valid_count))
    if lane == 0:
        log[0] = errors[0] / wp.max(valid[0], 1.0)
        log[1] = valid[0] / wp.float32(count)


def teacher_block_dim(device: torch.device | str) -> int:
    """Threads of the one block that runs :func:`teacher_labels_kernel`; Warp runs CPU blocks as one thread."""
    return 1024 if torch.device(device).type == "cuda" else 1


@wp.kernel(enable_backward=False)
def gaze_reset_kernel(
    ids: wp.array(dtype=wp.int32),
    use_ids: wp.int32,
    start: wp.array(dtype=wp.vec2f),
    use_start: wp.int32,
    field_radius_deg: wp.float32,
    angles_deg: wp.array(dtype=wp.vec2f),
    state: wp.array(dtype=wp.vec4f),
):
    """Put each reset environment's gaze at its start, or on the optical axis, at rest.

    ``state`` holds each gaze and its velocity, in field radii and field radii per second.
    """
    k = wp.tid()
    i = k
    if use_ids != 0:
        i = ids[k]
    g = wp.vec2f(0.0, 0.0)
    if use_start != 0:
        g = start[k]
    angles_deg[i] = g * field_radius_deg
    state[i] = wp.vec4f(g[0], g[1], 0.0, 0.0)


@wp.kernel(enable_backward=False)
def gaze_step_kernel(
    actions: wp.array(dtype=wp.vec2f),
    state: wp.array(dtype=wp.vec4f),
    angles_deg: wp.array(dtype=wp.vec2f),
    step_size: wp.float32,
    gaze_limit: wp.float32,
    step_dt: wp.float32,
    field_radius_deg: wp.float32,
):
    """Move each gaze by its clamped action and keep it within the limit; record the gaze, velocity and angles.

    ``state`` holds each gaze and its velocity, in field radii and field radii per second.
    """
    i = wp.tid()
    a = actions[i]
    s = state[i]
    previous = wp.vec2f(s[0], s[1])
    moved = previous + wp.vec2f(wp.clamp(a[0], -1.0, 1.0), wp.clamp(a[1], -1.0, 1.0)) * step_size
    current = moved * (gaze_limit / wp.max(wp.length(moved), gaze_limit))
    rate = (current - previous) / step_dt
    angles_deg[i] = current * field_radius_deg
    state[i] = wp.vec4f(current[0], current[1], rate[0], rate[1])


def angles_to_direction(angles_deg: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit directions of gaze angles [deg], shape (..., 2) to (..., 3)."""
    tilt, pan = angles_deg.deg2rad().unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def direction_to_angles(direction: torch.Tensor) -> torch.Tensor:
    """Gaze angles [deg] of camera-frame directions, shape (..., 3) to (..., 2)."""
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1).rad2deg()


def clamp_gaze(gaze: torch.Tensor, limit: float) -> torch.Tensor:
    """Scale gazes back onto the circle of radius ``limit`` when they leave it, shape (..., 2)."""
    return gaze * (limit / gaze.norm(dim=-1, keepdim=True).clamp_min(limit))


def random_gaze(count: int, limit: float, device: torch.device | str | None = None) -> torch.Tensor:
    """Gazes drawn uniformly over the disc of radius ``limit``, shape (count, 2)."""
    radius = limit * torch.rand(count, device=device).sqrt()
    angle = 2 * math.pi * torch.rand(count, device=device)
    return torch.stack((radius * angle.sin(), radius * angle.cos()), dim=-1)


class GazeAction(ActionTerm):
    """Moves a foveated camera's gaze by bounded steps in (tilt, pan) order.

    Actions, the limit and the observed gaze are in field radii. An action of 1 changes a coordinate by
    ``step_size``, and the gaze stays within ``gaze_limit`` of the optical axis. The gaze starts each episode on the
    optical axis, or anywhere the physical camera sees within the limit when ``random_reset`` is set. Gazes are measured
    in the upright image frame, which :attr:`~isaaclab_fovi.sensors.FoveatedCameraCfg.image_rotation_deg` rolls
    against a rolled mount.

    With ``mount="fixed"`` the camera keeps the orientation it had when the term was created, and the teacher uses
    that mount; turning the camera's mount raises. With ``mount="tracked"`` the camera may ride a moving body, for
    example a robot's head, and the teacher reads each camera's current mount, so software gaze adds to hardware gaze.
    """

    cfg: GazeActionCfg

    is_graphable: ClassVar[bool] = True
    """The gaze moves in :meth:`process_actions`, once per environment step; :meth:`apply_actions` does nothing, so
    physics may record the other terms' actions and its decimation loop into a CUDA graph."""

    def __init__(self, cfg: GazeActionCfg, env: ManagerBasedEnv):
        cfg.validate_config()
        super().__init__(cfg, env)
        self.camera = self._asset
        camera = self.camera
        self.tracked_mount = cfg.mount == "tracked"
        if self.tracked_mount and not (camera._gaze_pose_reads_angles and camera.cfg.update_latest_camera_pose):
            raise ValueError(
                "A tracked gaze mount needs a camera that turns its pose by the gaze every frame: Newton or OVRTX,"
                " with update_latest_camera_pose."
            )
        if not self.tracked_mount and camera.cfg.prim_path.rpartition("/")[0] not in (
            "{ENV_REGEX_NS}",
            env.scene.env_regex_ns,
        ):
            # A body's motion turns its camera without set_world_poses, which the fixed mount could not detect.
            raise ValueError(
                f"A fixed gaze mount needs a camera mounted on the environment, got {camera.cfg.prim_path!r};"
                " use mount='tracked' for a camera on a moving body."
            )
        if not hasattr(camera, "_gaze_camera"):
            raise RuntimeError(f"The gaze camera {cfg.asset_name!r} must be initialized before its gaze action.")
        # The camera's own geometry, so the teacher and the sensor mask share one projection.
        self.camera_model = camera._camera_model
        self._camera = camera._gaze_camera
        self._frame = camera._mask_frame
        self._frame_torch = torch.tensor(
            [[self._frame[row][col] for col in range(3)] for row in range(3)], device=self.device
        )
        self.field_radius_deg = camera.sampling_coords.fov / 2
        """Half the retina's field of view [deg]."""
        self._fixed_mount = None
        if not self.tracked_mount:
            # The camera's ungazed world orientation now, ROS convention; the fixed-mount teacher keeps it.
            orientation = camera._data.quat_w_world
            orientation = orientation.torch if hasattr(orientation, "torch") else orientation
            opengl = convert_camera_frame_quaternion(orientation.float(), origin="world", target="opengl")
            ros = convert_camera_frame_quaternion(opengl, origin="opengl", target="ros")
            self._fixed_mount = ProxyArray(wp.from_torch(ros.contiguous(), dtype=wp.quatf))
        self._raw: torch.Tensor | None = None
        """Gaze actions of the last step, as supplied to :meth:`process_actions`."""
        self._state = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec4f, device=self.device))
        """Gaze and its velocity, the gaze observation; the gaze and velocity are views of it."""
        self._gaze = self._state.torch[:, :2]
        self._velocity = self._state.torch[:, 2:]
        self._step_launch: tuple[tuple, wp.Launch] | None = None
        self._labels = ProxyArray(wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device))
        self._log = ProxyArray(wp.zeros(2, dtype=wp.float32, device=self.device))
        """Mean gaze error to valid teacher targets [deg] and the fraction of valid targets."""
        self._no_orientations = wp.zeros(1, dtype=wp.quatf, device=self.device)
        self._labels_launch = None
        # Each reset copies its environments, and their start gazes if random, into these and replays one launch.
        self._reset_ids = ProxyArray(wp.zeros(self.num_envs, dtype=wp.int32, device=self.device))
        self._reset_start = ProxyArray(wp.zeros(self.num_envs, dtype=wp.vec2f, device=self.device))
        self._reset_launches: dict[tuple[bool, bool], wp.Launch] = {}

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw if self._raw is not None else torch.zeros_like(self._gaze)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._gaze

    @property
    def gaze(self) -> torch.Tensor:
        """Current gaze in field radii, shape (num_envs, 2)."""
        return self._gaze

    @property
    def velocity(self) -> torch.Tensor:
        """Gaze velocity in field radii per second, shape (num_envs, 2)."""
        return self._velocity

    @property
    def state(self) -> torch.Tensor:
        """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
        return self._state.torch

    @property
    def angles_deg(self) -> torch.Tensor:
        """Current gaze angles [deg], shape (num_envs, 2)."""
        return self._gaze * self.field_radius_deg

    def process_actions(self, actions: torch.Tensor):
        """Move the gaze by the supplied actions, shape (num_envs, 2).

        The launch reads ``actions`` in place. The action manager hands the term the same slice of its buffer each
        step, so the launch is recorded once; another buffer records a launch that reads it.
        """
        if actions.dtype != torch.float32:
            actions = actions.float()
        key = (actions.data_ptr(), actions.stride(), actions.device)
        if self._step_launch is None or self._step_launch[0] != key:
            launch = record_launch(
                gaze_step_kernel,
                self.num_envs,
                [
                    wp.from_torch(actions, dtype=wp.vec2f),
                    self._state,
                    self._camera_angles(),
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._env.step_dt,
                    self.field_radius_deg,
                ],
                device=self.device,
            )
            self._step_launch = (key, launch)
        self._raw = actions
        self._step_launch[1].launch(warp_stream(self.device))
        self.camera.set_gaze_angles(self._camera_angles().torch)

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        # The action manager zeroes the reset environments' actions itself.
        indexed = env_ids is not None
        count = self.num_envs
        if indexed:
            if isinstance(env_ids, slice):
                env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
            count = len(env_ids)
            if count == 0:
                return
            self._reset_ids.torch[:count] = torch.as_tensor(env_ids, device=self.device)
        random_start = self.cfg.random_reset
        if random_start:
            gaze = random_gaze(count, self.cfg.gaze_limit, self.device)
            visible = self.visible(angles_to_direction(gaze * self.field_radius_deg))
            self._reset_start.torch[:count] = torch.where(visible[:, None], gaze, 0.0)
        angles = self._camera_angles()
        launch = self._reset_launches.get((indexed, random_start))
        if launch is None:
            launch = self._reset_launches[(indexed, random_start)] = record_launch(
                gaze_reset_kernel,
                self.num_envs,
                [self._reset_ids, int(indexed), self._reset_start, int(random_start), self.field_radius_deg],
                [angles, self._state],
                self.device,
            )
        launch.set_dim(count)
        launch.launch(warp_stream(self.device))
        # The camera keeps reading the same buffer, which every other environment already holds.
        self.camera.set_gaze_angles(angles.torch)

    def _camera_angles(self) -> ProxyArray:
        """The camera's stored gaze angles [deg], which the gaze kernels write in place."""
        self.camera.gaze_angles_buffer()
        return self.camera._gaze_angles

    def visible(self, direction: torch.Tensor) -> torch.Tensor:
        """Whether directions in the upright gaze frame fall inside the physical camera's image."""
        return project_directions(direction @ self._frame_torch.T, self._camera)[1]

    def _teacher_labels(
        self, targets: wp.array, orientations: wp.array | None = None, offset: wp.vec3f | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher labels toward a point fixed to each target, in one launch.

        Args:
            targets: Target positions in the world frame, one per environment.
            orientations: Target orientations (x, y, z, w), to place ``offset`` in each target's frame.
            offset: The point in each target's frame [m].

        Returns:
            The labels (action x, action y, valid), shape (num_envs, 3); and the mean gaze error to valid targets
            [deg] with the fraction of valid targets, shape (2,). Both are buffers the next call overwrites.

        Raises:
            RuntimeError: The camera's mount turned while the gaze assumes a fixed mount.
        """
        if self.tracked_mount:
            self.camera.gaze_angles_buffer()
            mounts = self.camera._mount_quat_ros
        elif self.camera.mount_moved:
            raise RuntimeError(
                "The gaze camera's mount turned, so the fixed-mount teacher would aim from its first orientation;"
                " use GazeActionCfg.mount='tracked'."
            )
        else:
            mounts = self._fixed_mount
        positions = self.camera.data.pos_w.warp
        sources = (targets, orientations, positions, offset, mounts)
        if self._labels_launch is None or any(a is not b for a, b in zip(self._labels_launch[0], sources)):
            block = teacher_block_dim(self.device)
            launch = record_launch(
                teacher_labels_kernel,
                (1, block),
                [
                    targets,
                    self._no_orientations if orientations is None else orientations,
                    wp.vec3f(0.0, 0.0, 0.0) if offset is None else offset,
                    int(orientations is not None),
                    positions,
                    mounts,
                    self._state,
                    self._frame,
                    self.field_radius_deg,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._camera,
                ],
                [self._labels, self._log],
                self.device,
                block_dim=block,
            )
            self._labels_launch = (sources, launch)
        self._labels_launch[1].launch(warp_stream(self.device))
        return self._labels.torch, self._log.torch


def gaze_state(env: ManagerBasedEnv, action_name: str = "gaze") -> torch.Tensor:
    """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
    term: GazeAction = env.action_manager.get_term(action_name)
    return term.state


def entity_point(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] | torch.Tensor = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """World position of a point fixed to an asset's root, or to its first selected body, shape (num_envs, 3).

    Args:
        asset_cfg: The asset, and optionally one body by ``body_names``.
        offset: The point in the root's or the body's frame [m].
    """
    data = env.scene[asset_cfg.name].data
    if asset_cfg.body_names is None:
        position, orientation = data.root_pos_w.torch, data.root_quat_w.torch
    else:
        body = asset_cfg.body_ids[0]
        position, orientation = data.body_pos_w.torch[:, body], data.body_quat_w.torch[:, body]
    offset = torch.as_tensor(offset, dtype=position.dtype, device=position.device)
    return position + quat_apply(orientation, offset.expand_as(position))


def teacher_labels(env: ManagerBasedEnv, points: torch.Tensor, action_name: str = "gaze") -> torch.Tensor:
    """The teacher's gaze action toward world points and its validity, shape (num_envs, 3).

    A training label, never a policy input. Tasks whose target changes, for example from an object to a goal once
    the object is grasped, choose the points and call this.
    """
    term: GazeAction = env.action_manager.get_term(action_name)
    labels, values = term._teacher_labels(wp.from_torch(points.float().contiguous(), dtype=wp.vec3f))
    _log_teacher(env, values)
    return labels


def _log_teacher(env: ManagerBasedEnv, values: torch.Tensor) -> None:
    """Log the mean gaze error to valid targets and the fraction of valid targets."""
    # Loggers keep each step's entries, so they get a copy rather than the buffer the next step overwrites.
    error, fraction = values.clone()
    log = env.extras.setdefault("log", {})
    log["Gaze/target_error_deg"] = error
    log["Gaze/valid_target_fraction"] = fraction


class gaze_teacher_labels(ManagerTermBase):
    """The teacher's gaze action toward a point fixed to an asset, and its validity, shape (num_envs, 3).

    Args:
        target_cfg: The asset, and optionally one body by ``body_names``.
        offset: The point in the root's or the body's frame [m].
        action_name: The gaze action term.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._offset = wp.vec3f(*cfg.params.get("offset", (0.0, 0.0, 0.0)))
        self._sources = None
        self._targets = self._orientations = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_cfg: SceneEntityCfg,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        action_name: str = "gaze",
    ) -> torch.Tensor:
        data = env.scene[target_cfg.name].data
        # Reading the properties refreshes the asset's pose buffers; their Warp views are wrapped once.
        if target_cfg.body_names is None:
            positions, orientations = data.root_pos_w, data.root_quat_w
        else:
            positions, orientations = data.body_pos_w, data.body_quat_w
        if self._sources != (id(positions), id(orientations)):
            self._sources = (id(positions), id(orientations))
            if target_cfg.body_names is None:
                self._targets, self._orientations = positions.warp, orientations.warp
            else:
                body = target_cfg.body_ids[0]
                self._targets = wp.from_torch(positions.torch[:, body], dtype=wp.vec3f)
                self._orientations = wp.from_torch(orientations.torch[:, body], dtype=wp.quatf)
        term: GazeAction = env.action_manager.get_term(action_name)
        labels, values = term._teacher_labels(self._targets, self._orientations, self._offset)
        _log_teacher(env, values)
        return labels


def motor_action_l2(env: ManagerBasedRLEnv, gaze_action_name: str = "gaze") -> torch.Tensor:
    """Squared L2 norm of the actions other than the gaze, shape (num_envs,).

    Matches :func:`isaaclab.envs.mdp.action_l2` on the task's own actions, so moving the gaze costs nothing.
    """
    gaze = env.action_manager.get_term(gaze_action_name).raw_actions
    return env.action_manager.action.square().sum(-1) - gaze.square().sum(-1)
