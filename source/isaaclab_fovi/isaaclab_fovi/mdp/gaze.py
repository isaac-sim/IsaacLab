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
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_apply, quat_apply_inverse

from ..sensors.calibrated_geometry import resolve_camera_model, sampling_geometry
from ..sensors.gaze_kernels import (
    DEG_TO_RAD,
    CameraParams,
    camera_params,
    project_direction,
    project_directions,
    torch_stream,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

RAD_TO_DEG = 180.0 / math.pi


@wp.func
def teach(
    point: wp.vec3f,
    position: wp.vec3f,
    mount: wp.quatf,
    g: wp.vec2f,
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
) -> wp.vec4f:
    """The teacher's action (x, y) toward a point, 1 for a valid target or else 0, and the gaze's error [deg]."""
    d = wp.normalize(wp.quat_rotate_inv(mount, point - position))
    tilt = -wp.atan2(d[1], wp.sqrt(d[0] * d[0] + d[2] * d[2])) * wp.float32(RAD_TO_DEG)
    pan = -wp.atan2(d[0], d[2]) * wp.float32(RAD_TO_DEG)
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
    error = wp.atan2(wp.length(wp.cross(looking, d)), wp.dot(looking, d)) * wp.float32(RAD_TO_DEG)
    return wp.vec4f(step[0], step[1], wp.where(ok, 1.0, 0.0), error)


@wp.kernel(enable_backward=False)
def teacher_kernel(
    points: wp.array(dtype=wp.vec3f),
    positions: wp.array(dtype=wp.vec3f),
    mount: wp.quatf,
    gaze: wp.array(dtype=wp.vec2f),
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
    action: wp.array(dtype=wp.vec2f),
    valid: wp.array(dtype=wp.bool),
    error_deg: wp.array(dtype=wp.float32),
):
    """The teacher's action toward each point, whether the point is a valid target, and the gaze's error to it."""
    i = wp.tid()
    result = teach(points[i], positions[i], mount, gaze[i], field_radius_deg, step_size, gaze_limit, camera)
    action[i] = wp.vec2f(result[0], result[1])
    valid[i] = result[2] > 0.5
    error_deg[i] = result[3]


@wp.kernel(enable_backward=False)
def teacher_labels_kernel(
    targets: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    offset: wp.vec3f,
    rotate_offset: wp.int32,
    positions: wp.array(dtype=wp.vec3f),
    mount: wp.quatf,
    gaze: wp.array(dtype=wp.vec2f),
    field_radius_deg: wp.float32,
    step_size: wp.float32,
    gaze_limit: wp.float32,
    camera: CameraParams,
    labels: wp.array2d(dtype=wp.float32),
    sums: wp.array(dtype=wp.float32),
):
    """Labels (action x, action y, valid) toward a point fixed to each target, and the error and count sums of valid
    targets."""
    i = wp.tid()
    point = targets[i]
    if rotate_offset != 0:
        point = point + wp.quat_rotate(orientations[i], offset)
    result = teach(point, positions[i], mount, gaze[i], field_radius_deg, step_size, gaze_limit, camera)
    labels[i, 0] = result[0]
    labels[i, 1] = result[1]
    labels[i, 2] = result[2]
    if result[2] > 0.5:
        wp.atomic_add(sums, 0, result[3])
        wp.atomic_add(sums, 1, 1.0)


@wp.kernel(enable_backward=False)
def gaze_step_kernel(
    actions: wp.array(dtype=wp.vec2f),
    gaze: wp.array(dtype=wp.vec2f),
    velocity: wp.array(dtype=wp.vec2f),
    angles_deg: wp.array(dtype=wp.vec2f),
    step_size: wp.float32,
    gaze_limit: wp.float32,
    step_dt: wp.float32,
    field_radius_deg: wp.float32,
):
    """Move each gaze by its clamped action, keep it within the limit, and record its velocity and angles."""
    i = wp.tid()
    a = actions[i]
    previous = gaze[i]
    moved = previous + wp.vec2f(wp.clamp(a[0], -1.0, 1.0), wp.clamp(a[1], -1.0, 1.0)) * step_size
    current = moved * (gaze_limit / wp.max(wp.length(moved), gaze_limit))
    gaze[i] = current
    velocity[i] = (current - previous) / step_dt
    angles_deg[i] = current * field_radius_deg


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
    optical axis, or anywhere the physical camera sees within the limit when ``random_reset`` is set.

    The camera must be mounted on the environment, so that its offset is its pose in the environment frame.
    """

    cfg: GazeActionCfg

    def __init__(self, cfg: GazeActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.camera = self._asset
        camera_cfg = self.camera.cfg
        if camera_cfg.prim_path.rpartition("/")[0] not in ("{ENV_REGEX_NS}", env.scene.env_regex_ns):
            raise ValueError(f"Gaze needs a camera mounted on the environment, got {camera_cfg.prim_path!r}.")
        self.camera_model = resolve_camera_model(camera_cfg)
        self._camera = camera_params(self.camera_model)
        self.field_radius_deg = sampling_geometry(camera_cfg)[0] / 2
        """Half the retina's field of view [deg]."""
        offset = camera_cfg.offset
        rotation = torch.tensor(offset.rot, dtype=torch.float32, device=self.device)[None]
        self.mount_rotation = convert_camera_frame_orientation_convention(
            rotation, origin=offset.convention, target="ros"
        )[0]
        """Orientation (x, y, z, w) of the physical camera in the environment frame, ROS convention."""
        self._mount = wp.quatf(*self.mount_rotation.tolist())
        self._raw = torch.zeros(self.num_envs, 2, device=self.device)
        self._gaze = torch.zeros_like(self._raw)
        self._velocity = torch.zeros_like(self._raw)
        self._angles = torch.zeros_like(self._raw)
        """Gaze angles [deg] of the last step, the buffer the camera reads."""
        self._raw_wp, self._gaze_wp, self._velocity_wp, self._angles_wp = (
            wp.from_torch(buffer, dtype=wp.vec2f) for buffer in (self._raw, self._gaze, self._velocity, self._angles)
        )
        self._label_sums = torch.zeros(2, device=self.device)
        self._label_sums_wp = wp.from_torch(self._label_sums)
        self._no_orientations = wp.from_torch(torch.zeros(1, 4, device=self.device), dtype=wp.quatf)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

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
    def angles_deg(self) -> torch.Tensor:
        """Current gaze angles [deg], shape (num_envs, 2)."""
        return self._gaze * self.field_radius_deg

    def process_actions(self, actions: torch.Tensor):
        self._raw.copy_(actions)
        with torch_stream(self.device):
            wp.launch(
                gaze_step_kernel,
                dim=self.num_envs,
                inputs=[
                    self._raw_wp,
                    self._gaze_wp,
                    self._velocity_wp,
                    self._angles_wp,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._env.step_dt,
                    self.field_radius_deg,
                ],
            )
        self.camera.set_gaze_angles(self._angles)

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._raw[ids] = 0.0
        self._velocity[ids] = 0.0
        self._gaze[ids] = 0.0
        if self.cfg.random_reset and len(ids):
            gaze = random_gaze(len(ids), self.cfg.gaze_limit, self.device)
            visible = self.visible(angles_to_direction(gaze * self.field_radius_deg))
            self._gaze[ids] = torch.where(visible[:, None], gaze, 0.0)
        self.camera.set_gaze_angles(self.angles_deg)

    def directions(self, points: torch.Tensor) -> torch.Tensor:
        """Camera-frame unit directions from the camera to world points, shape (num_envs, 3)."""
        position = self.camera.data.pos_w.torch
        local = quat_apply_inverse(self.mount_rotation.expand(len(points), -1), points - position)
        return torch.nn.functional.normalize(local, dim=-1)

    def visible(self, direction: torch.Tensor) -> torch.Tensor:
        """Whether camera-frame directions fall inside the physical camera's image."""
        return project_directions(direction, self._camera)[1]

    def teacher(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The teacher's action toward one world point per environment, in one launch.

        A point is a valid target when the physical camera sees it and the gaze can reach it within the limit; the
        action toward an invalid target is zero.

        Returns:
            The actions, shape (num_envs, 2); whether each point is a valid target, shape (num_envs,); and the angle
            between the gaze and each point [deg], shape (num_envs,).
        """
        points = points.float().contiguous()
        action = torch.empty(len(points), 2, device=self.device)
        valid = torch.empty(len(points), dtype=torch.bool, device=self.device)
        error = torch.empty(len(points), device=self.device)
        with torch_stream(self.device):
            wp.launch(
                teacher_kernel,
                dim=len(points),
                inputs=[
                    wp.from_torch(points, dtype=wp.vec3f),
                    self.camera.data.pos_w.warp,
                    self._mount,
                    self._gaze_wp,
                    self.field_radius_deg,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._camera,
                ],
                outputs=[wp.from_torch(action, dtype=wp.vec2f), wp.from_torch(valid), wp.from_torch(error)],
            )
        return action, valid, error

    def _teacher_labels(
        self, targets: wp.array, orientations: wp.array | None = None, offset: wp.vec3f | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher labels toward a point fixed to each target, in one launch.

        Args:
            targets: Target positions in the world frame, one per environment.
            orientations: Target orientations (x, y, z, w), to place ``offset`` in each target's frame.
            offset: The point in each target's frame [m].

        Returns:
            The labels (action x, action y, valid), shape (num_envs, 3); and the sum of the gaze's errors to valid
            targets [deg] with their count, shape (2,), a buffer the next call overwrites.
        """
        labels = torch.empty(self.num_envs, 3, device=self.device)
        self._label_sums.zero_()
        with torch_stream(self.device):
            wp.launch(
                teacher_labels_kernel,
                dim=self.num_envs,
                inputs=[
                    targets,
                    self._no_orientations if orientations is None else orientations,
                    wp.vec3f(0.0, 0.0, 0.0) if offset is None else offset,
                    int(orientations is not None),
                    self.camera.data.pos_w.warp,
                    self._mount,
                    self._gaze_wp,
                    self.field_radius_deg,
                    self.cfg.step_size,
                    self.cfg.gaze_limit,
                    self._camera,
                ],
                outputs=[wp.from_torch(labels), self._label_sums_wp],
            )
        return labels, self._label_sums


@configclass
class GazeActionCfg(ActionTermCfg):
    """Configuration of :class:`GazeAction`."""

    class_type: type = GazeAction
    asset_name: str = MISSING
    """Scene name of the foveated camera whose gaze the action moves."""
    step_size: float = 0.1
    """Change of each gaze coordinate per step at an action of 1, in field radii."""
    gaze_limit: float = 1.0
    """Largest distance of the gaze from the camera's optical axis, in field radii."""
    random_reset: bool = False
    """Start episodes at a random gaze the camera sees; otherwise on the optical axis."""


def gaze_state(env: ManagerBasedEnv, action_name: str = "gaze") -> torch.Tensor:
    """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
    term: GazeAction = env.action_manager.get_term(action_name)
    return torch.cat((term.gaze, term.velocity), dim=-1)


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
    labels, sums = term._teacher_labels(wp.from_torch(points.float().contiguous(), dtype=wp.vec3f))
    _log_teacher(env, sums, len(labels))
    return labels


def _log_teacher(env: ManagerBasedEnv, sums: torch.Tensor, count: int) -> None:
    """Log the mean gaze error to valid targets and the fraction of valid targets."""
    log = env.extras.setdefault("log", {})
    log["Gaze/target_error_deg"] = sums[0] / sums[1].clamp_min(1)
    log["Gaze/valid_target_fraction"] = sums[1] / count


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
        labels, sums = term._teacher_labels(self._targets, self._orientations, self._offset)
        _log_teacher(env, sums, len(labels))
        return labels


def motor_action_l2(env: ManagerBasedRLEnv, gaze_action_name: str = "gaze") -> torch.Tensor:
    """Squared L2 norm of the actions other than the gaze, shape (num_envs,).

    Matches :func:`isaaclab.envs.mdp.action_l2` on the task's own actions, so moving the gaze costs nothing.
    """
    gaze = env.action_manager.get_term(gaze_action_name).raw_actions
    return env.action_manager.action.square().sum(-1) - gaze.square().sum(-1)
