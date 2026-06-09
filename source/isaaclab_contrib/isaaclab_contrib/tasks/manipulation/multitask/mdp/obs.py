# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for multi-task environments.

Group-local terms are wrapped with :func:`~.utils.scatterable`: they index
asset and command data with ``SceneEntityCfg.view_ids`` / ``env_ids`` and
return ``(env_ids, result)``; the decorator scatters ``result`` into a full
``(num_envs, ...)`` tensor.

:class:`~.utils.scatter_term` can compose multiple scatterable children into
one buffer.  :func:`generated_commands` uses layout groups and manages its
own full-env buffer.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

from .utils import ScatterResult, _any_group_disabled, scatterable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg


@scatterable(output_dim=7)
def ee_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> ScatterResult:
    """EE pose in robot root frame [m, -]. Returns shape ``(num_envs, 7)``."""
    robot = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    root_pos = wp.to_torch(robot.data.root_pos_w)[asset_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[asset_cfg.view_ids]
    body_pos = wp.to_torch(robot.data.body_pos_w)[asset_cfg.view_ids, body_idx]
    body_quat = wp.to_torch(robot.data.body_quat_w)[asset_cfg.view_ids, body_idx]
    pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, body_pos, body_quat)
    return asset_cfg.env_ids, torch.cat([pos_b, quat_b], dim=-1)


@scatterable(output_dim=3)
def ee_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "ee_pose") -> ScatterResult:
    """EE position error ``(target - current)`` in root frame [m]. Returns shape ``(num_envs, 3)``."""
    robot = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    root_pos = wp.to_torch(robot.data.root_pos_w)[asset_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[asset_cfg.view_ids]
    body_pos = wp.to_torch(robot.data.body_pos_w)[asset_cfg.view_ids, body_idx]
    cmd_pos = env.command_manager.get_command(command_name)[asset_cfg.env_ids, :3]
    cur_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, body_pos)
    return asset_cfg.env_ids, cmd_pos - cur_b


@scatterable(output_dim=3)
def object_pos_in_robot_frame(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
) -> ScatterResult:
    """Object position in robot root frame [m]. Returns shape ``(num_envs, 3)``."""
    robot = env.scene[robot_cfg.name]
    rigid_object = env.scene[object_cfg.name]
    root_pos = wp.to_torch(robot.data.root_pos_w)[robot_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[robot_cfg.view_ids]
    obj_pos = wp.to_torch(rigid_object.data.root_pos_w)[object_cfg.view_ids, :3]
    obj_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, obj_pos)
    return robot_cfg.env_ids, obj_b


@scatterable(output_dim=3)
def ee_object_pos_error(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg
) -> ScatterResult:
    """Position error ``(object - ee)`` in robot root frame [m]. Returns shape ``(num_envs, 3)``."""
    robot = env.scene[robot_cfg.name]
    rigid_object = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    root_pos = wp.to_torch(robot.data.root_pos_w)[robot_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[robot_cfg.view_ids]
    obj_pos = wp.to_torch(rigid_object.data.root_pos_w)[object_cfg.view_ids, :3]
    ee_pos = wp.to_torch(ee.data.target_pos_w)[ee_frame_cfg.view_ids, 0, :]
    obj_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, obj_pos)
    ee_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, ee_pos)
    return robot_cfg.env_ids, obj_b - ee_b


@scatterable(output_dim=3)
def object_target_pos_error(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, command_name: str = "ee_pose"
) -> ScatterResult:
    """Position error ``(target - object)`` in robot root frame [m]. Returns shape ``(num_envs, 3)``."""
    robot = env.scene[robot_cfg.name]
    rigid_object = env.scene[object_cfg.name]
    root_pos = wp.to_torch(robot.data.root_pos_w)[robot_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[robot_cfg.view_ids]
    obj_pos = wp.to_torch(rigid_object.data.root_pos_w)[object_cfg.view_ids, :3]
    cmd_pos = env.command_manager.get_command(command_name)[robot_cfg.env_ids, :3]
    obj_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, obj_pos)
    return robot_cfg.env_ids, cmd_pos - obj_b


@scatterable
def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> ScatterResult:
    """Relative joint positions [rad or m]. Returns shape ``(num_envs, num_joints)``."""
    robot = env.scene[asset_cfg.name]
    pos_data = wp.to_torch(robot.data.joint_pos)[asset_cfg.view_ids]
    default_data = wp.to_torch(robot.data.default_joint_pos)[asset_cfg.view_ids]
    if asset_cfg.joint_ids != slice(None):
        pos_data = pos_data[:, asset_cfg.joint_ids]
        default_data = default_data[:, asset_cfg.joint_ids]
    return asset_cfg.env_ids, pos_data - default_data


@scatterable
def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> ScatterResult:
    """Joint velocities [rad/s or m/s]. Returns shape ``(num_envs, num_joints)``."""
    robot = env.scene[asset_cfg.name]
    vel_data = wp.to_torch(robot.data.joint_vel)[asset_cfg.view_ids]
    if asset_cfg.joint_ids != slice(None):
        vel_data = vel_data[:, asset_cfg.joint_ids]
    return asset_cfg.env_ids, vel_data


@scatterable
def generated_commands(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "ee_pose"
) -> ScatterResult:
    """Generated commands. Returns shape ``(num_envs, cmd_dim)``."""
    cmd = env.command_manager.get_command(command_name)
    return asset_cfg.env_ids, cmd[asset_cfg.env_ids]


@scatterable
def cabinet_joint_pos(env: ManagerBasedRLEnv, cabinet_asset_cfg: SceneEntityCfg) -> ScatterResult:
    """Relative cabinet joint positions [rad or m]. Returns shape ``(num_envs, num_joints)``."""
    cabinet = env.scene[cabinet_asset_cfg.name]
    pos_data = wp.to_torch(cabinet.data.joint_pos)[cabinet_asset_cfg.view_ids]
    default_data = wp.to_torch(cabinet.data.default_joint_pos)[cabinet_asset_cfg.view_ids]
    if cabinet_asset_cfg.joint_ids != slice(None):
        pos_data = pos_data[:, cabinet_asset_cfg.joint_ids]
        default_data = default_data[:, cabinet_asset_cfg.joint_ids]
    return cabinet_asset_cfg.env_ids, pos_data - default_data


@scatterable
def cabinet_joint_vel(env: ManagerBasedRLEnv, cabinet_asset_cfg: SceneEntityCfg) -> ScatterResult:
    """Cabinet joint velocities [rad/s or m/s]. Returns shape ``(num_envs, num_joints)``."""
    cabinet = env.scene[cabinet_asset_cfg.name]
    vel_data = wp.to_torch(cabinet.data.joint_vel)[cabinet_asset_cfg.view_ids]
    if cabinet_asset_cfg.joint_ids != slice(None):
        vel_data = vel_data[:, cabinet_asset_cfg.joint_ids]
    return cabinet_asset_cfg.env_ids, vel_data


@scatterable(output_dim=3)
def cabinet_rel_ee_drawer_distance(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg, cabinet_frame_cfg: SceneEntityCfg
) -> ScatterResult:
    """Drawer-handle minus EE TCP position [m]. Returns shape ``(num_envs, 3)``."""
    ee_pos = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 0, :]
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    return ee_frame_cfg.env_ids, handle_pos - ee_pos


def zero_obs(env: ManagerBasedRLEnv, dim: int = 1) -> torch.Tensor:
    """Zero-filled observation placeholder [--]. Returns shape ``(num_envs, dim)``."""
    return torch.zeros(env.num_envs, dim, device=env.device)


def _with_fallback_dim(func, dim: int):
    """Wrap a :func:`~.utils.scatterable` function with an explicit disabled-group shape.

    When any :class:`~isaaclab.managers.SceneEntityCfg` kwarg has an empty
    ``env_ids`` tensor (i.e. the group is disabled), the wrapper returns a
    ``(num_envs, dim)`` zero buffer without calling *func*.  This gives
    variable-dim scatterable functions a known output shape even when their
    group is weight-zero, without needing :func:`apply_task_filter` to replace
    the term at config time.

    Args:
        func: A ``@scatterable``-decorated callable.
        dim: The trailing output dimension to use for the zero buffer.

    Returns:
        A wrapped callable with ``output_dim`` set to *dim*.
    """

    @functools.wraps(func)
    def wrapper(env, *args, _out=None, **kwargs):
        if _any_group_disabled(*args, **kwargs):
            if _out is not None:
                return _out
            if not hasattr(wrapper, "_buf"):
                wrapper._buf = torch.zeros(env.num_envs, dim, device=env.device)
            wrapper._buf.zero_()
            return wrapper._buf
        return func(env, *args, _out=_out, **kwargs)

    wrapper.output_dim = dim
    return wrapper


@configclass
class MultiTaskObsTerm(ObsTerm):
    """ObsTerm that declares its output dimension for variable-dim group obs.

    Use this instead of plain :class:`ObsTerm` for group-specific observation
    terms whose output size depends on asset configuration (e.g. joint count).
    Setting ``dim`` causes :meth:`__post_init__` to wrap :attr:`func` so that
    a ``(num_envs, dim)`` zero buffer is returned automatically when the group
    is disabled — no call to :func:`apply_task_filter` needed.
    """

    dim: int | None = None

    def __post_init__(self):
        if self.dim is not None and callable(self.func):
            self.func = _with_fallback_dim(self.func, self.dim)


class multi_task_onehot(ManagerTermBase):
    """One-hot encoding of selector term for every environment.
    Returns shape ``(num_envs, num_selectors)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        selector = env.scene.selector
        groups = selector.selector_names
        group_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for i, name in enumerate(groups):
            env_to_view_map = selector[name]
            group_idx[env_to_view_map.env_ids] = i
        self._onehot = torch.nn.functional.one_hot(group_idx, num_classes=len(groups)).float()

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return self._onehot
