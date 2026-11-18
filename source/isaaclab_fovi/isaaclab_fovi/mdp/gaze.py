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

from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_apply, quat_apply_inverse

from ..sensors.calibrated_geometry import resolve_camera_model, sampling_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


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
        self.field_radius_deg = sampling_geometry(camera_cfg)[0] / 2
        """Half the retina's field of view [deg]."""
        offset = camera_cfg.offset
        rotation = torch.tensor(offset.rot, dtype=torch.float32, device=self.device)[None]
        self.mount_rotation = convert_camera_frame_orientation_convention(
            rotation, origin=offset.convention, target="ros"
        )[0]
        """Orientation (x, y, z, w) of the physical camera in the environment frame, ROS convention."""
        self._raw = torch.zeros(self.num_envs, 2, device=self.device)
        self._gaze = torch.zeros_like(self._raw)
        self._velocity = torch.zeros_like(self._raw)

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
        previous = self._gaze.clone()
        self._gaze.copy_(clamp_gaze(self._gaze + actions.clamp(-1, 1) * self.cfg.step_size, self.cfg.gaze_limit))
        self._velocity.copy_((self._gaze - previous) / self._env.step_dt)
        self.camera.set_gaze_angles(self.angles_deg)

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
        return self.camera_model.project(direction)[1]

    def teacher_action(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The action that moves the gaze toward world points, and whether each point is a valid target.

        A point is valid when the physical camera sees it and the gaze can reach it within the limit.
        """
        direction = self.directions(points)
        desired = direction_to_angles(direction) / self.field_radius_deg
        valid = self.visible(direction) & (desired.norm(dim=-1) <= self.cfg.gaze_limit)
        action = ((desired - self._gaze) / self.cfg.step_size).clamp(-1, 1)
        return torch.where(valid[:, None], action, 0.0), valid

    def fixation_error_deg(self, points: torch.Tensor) -> torch.Tensor:
        """Angle [deg] between the gaze and the direction to world points, shape (num_envs,)."""
        direction = self.directions(points)
        gaze = angles_to_direction(self.angles_deg)
        # atan2 stays accurate near perfect alignment, unlike acos of the dot product.
        return torch.atan2(torch.cross(gaze, direction, dim=-1).norm(dim=-1), (gaze * direction).sum(-1)).rad2deg()


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
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
    return position + quat_apply(orientation, torch.tensor(offset, device=env.device).expand_as(position))


def teacher_labels(env: ManagerBasedEnv, points: torch.Tensor, action_name: str = "gaze") -> torch.Tensor:
    """The teacher's gaze action toward world points and its validity, shape (num_envs, 3).

    A training label, never a policy input. Tasks whose target changes, for example from an object to a goal once
    the object is grasped, choose the points and call this.
    """
    term: GazeAction = env.action_manager.get_term(action_name)
    action, valid = term.teacher_action(points)
    error = term.fixation_error_deg(points)
    log = env.extras.setdefault("log", {})
    log["Gaze/target_error_deg"] = torch.where(valid, error, 0.0).sum() / valid.sum().clamp_min(1)
    log["Gaze/valid_target_fraction"] = valid.float().mean()
    return torch.cat((action, valid[:, None].float()), dim=-1)


def gaze_teacher_labels(
    env: ManagerBasedEnv,
    target_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    action_name: str = "gaze",
) -> torch.Tensor:
    """The teacher's gaze action toward a point fixed to an asset, and its validity, shape (num_envs, 3)."""
    return teacher_labels(env, entity_point(env, target_cfg, offset), action_name)


def motor_action_l2(env: ManagerBasedRLEnv, gaze_action_name: str = "gaze") -> torch.Tensor:
    """Squared L2 norm of the actions other than the gaze, shape (num_envs,).

    Matches :func:`isaaclab.envs.mdp.action_l2` on the task's own actions, so moving the gaze costs nothing.
    """
    gaze = env.action_manager.get_term(gaze_action_name).raw_actions
    return env.action_manager.action.square().sum(-1) - gaze.square().sum(-1)
