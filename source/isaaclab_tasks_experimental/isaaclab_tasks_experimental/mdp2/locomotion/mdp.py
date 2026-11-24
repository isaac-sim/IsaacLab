# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Ant and Humanoid manager workflows' locomotion terms."""

from __future__ import annotations

import math

import torch
import warp as wp
from isaaclab_experimental.mdp2 import Implementation, State, Term, TermContext
from isaaclab_experimental.mdp2.mdp.observations import projected_gravity_b, projected_gravity_torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.string import resolve_matching_names_values


@wp.func
def _wrap_to_pi(angle: float) -> float:
    offset = angle + math.pi
    wrapped = offset - 2.0 * math.pi * wp.floor(offset / (2.0 * math.pi))
    return math.pi if wrapped == 0.0 and angle > 0.0 else wrapped - math.pi


@wp.func
def _yaw(quat: wp.quatf) -> float:
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@wp.func
def _base_height(world: int, root_link_pose_w: wp.array(dtype=wp.transformf)) -> float:
    return wp.transform_get_translation(root_link_pose_w[world])[2]


base_height = Term(torch=lambda root_link_pose_w: root_link_pose_w[:, 2], warp=_base_height)


@wp.func
def _base_yaw_roll(world: int, axis: int, root_link_pose_w: wp.array(dtype=wp.transformf)) -> float:
    quat = wp.transform_get_rotation(root_link_pose_w[world])
    if axis == 0:
        return _wrap_to_pi(_yaw(quat))
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return _wrap_to_pi(wp.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def _base_yaw_roll_torch(root_link_pose_w: torch.Tensor) -> torch.Tensor:
    roll, _, yaw = math_utils.euler_xyz_from_quat(root_link_pose_w[:, 3:7])
    return math_utils.wrap_to_pi(torch.stack((yaw, roll), dim=1))


base_yaw_roll = Term(torch=_base_yaw_roll_torch, warp=_base_yaw_roll, shape=("worlds", 2))


@wp.func
def _target_delta(root_link_pose: wp.transformf, origin: wp.vec3f, target_x: float, target_y: float) -> wp.vec2f:
    pos = wp.transform_get_translation(root_link_pose)
    return wp.vec2f(origin[0] + target_x - pos[0], origin[1] + target_y - pos[1])


def _target_delta_torch(
    root_link_pose_w: torch.Tensor, env_origins: torch.Tensor, target_x: float, target_y: float
) -> torch.Tensor:
    x = env_origins[:, 0] + target_x - root_link_pose_w[:, 0]
    y = env_origins[:, 1] + target_y - root_link_pose_w[:, 1]
    return torch.stack((x, y), dim=1)


@wp.func
def _base_angle_to_target(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    target_x: float,
    target_y: float,
) -> float:
    pose = root_link_pose_w[world]
    delta = _target_delta(pose, env_origins[world], target_x, target_y)
    return _wrap_to_pi(wp.atan2(delta[1], delta[0]) - _yaw(wp.transform_get_rotation(pose)))


def _base_angle_to_target_torch(root_link_pose_w, env_origins, target_x, target_y):
    delta = _target_delta_torch(root_link_pose_w, env_origins, target_x, target_y)
    _, _, yaw = math_utils.euler_xyz_from_quat(root_link_pose_w[:, 3:7])
    return math_utils.wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - yaw)


base_angle_to_target = Term(torch=_base_angle_to_target_torch, warp=_base_angle_to_target)


@wp.func
def _base_heading_proj(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    target_x: float,
    target_y: float,
) -> float:
    pose = root_link_pose_w[world]
    delta = _target_delta(pose, env_origins[world], target_x, target_y)
    heading = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3f(1.0, 0.0, 0.0))
    direction = delta / wp.max(wp.length(delta), 1.0e-9)
    return wp.dot(wp.vec2f(heading[0], heading[1]), direction)


def _base_heading_proj_torch(root_link_pose_w, env_origins, target_x, target_y):
    delta = _target_delta_torch(root_link_pose_w, env_origins, target_x, target_y)
    quat = root_link_pose_w[:, 3:7]
    forward = torch.zeros_like(quat[:, :3])
    forward[:, 0] = 1.0
    heading = math_utils.quat_apply(quat, forward)
    return (heading[:, :2] * math_utils.normalize(delta)).sum(dim=1)


base_heading_proj = Term(torch=_base_heading_proj_torch, warp=_base_heading_proj)


@wp.func
def _base_up_proj(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), gravity_w: wp.array(dtype=wp.vec3f)
) -> float:
    return -projected_gravity_b(world, root_link_pose_w, gravity_w)[2]


def _base_up_proj_torch(root_link_pose_w, gravity_w):
    return -projected_gravity_torch(root_link_pose_w, gravity_w)[:, 2]


base_up_proj = Term(torch=_base_up_proj_torch, warp=_base_up_proj)


@wp.func
def _upright_posture_bonus(world: int, up: wp.array(dtype=float), threshold: float) -> float:
    return float(up[world] > threshold)


upright_posture_bonus = Term(torch=lambda up, threshold: (up > threshold).float(), warp=_upright_posture_bonus)


@wp.func
def _move_to_target_bonus(world: int, heading: wp.array(dtype=float), threshold: float) -> float:
    return wp.min(heading[world] / threshold, 1.0)


move_to_target_bonus = Term(
    torch=lambda heading, threshold: (heading / threshold).clamp(max=1.0),
    warp=_move_to_target_bonus,
)


@wp.func
def _progress(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    potential: wp.array(dtype=float),
    target_x: float,
    target_y: float,
    step_dt: float,
) -> float:
    previous = potential[world]
    delta = _target_delta(root_link_pose_w[world], env_origins[world], target_x, target_y)
    current = -wp.length(delta) / step_dt
    potential[world] = current
    return current - previous


@wp.func
def _reset_progress(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    potential: wp.array(dtype=float),
    target_x: float,
    target_y: float,
    step_dt: float,
):
    delta = _target_delta(root_link_pose_w[world], env_origins[world], target_x, target_y)
    potential[world] = -wp.length(delta) / step_dt


def _progress_torch(root_link_pose_w, env_origins, potential, target_x, target_y, step_dt):
    current = -_target_delta_torch(root_link_pose_w, env_origins, target_x, target_y).norm(dim=1) / step_dt
    change = current - potential
    potential.copy_(current)
    return change


def _reset_progress_torch(mask, root_link_pose_w, env_origins, potential, target_x, target_y, step_dt):
    current = -_target_delta_torch(root_link_pose_w, env_origins, target_x, target_y).norm(dim=1) / step_dt
    torch.where(mask, current, potential, out=potential)


progress_reward = Term(
    torch=Implementation(compute=_progress_torch, reset=_reset_progress_torch),
    warp=Implementation(compute=_progress, reset=_reset_progress),
    state={"potential": State()},
)


def _init_gears(ctx: TermContext, gear_ratio: dict[str, float], **_) -> None:
    names = ctx.scene_data.schema.entities[ctx.selectors["joint_ids"].entity].names["joints"]
    selected_names = [names[index] for index in ctx.inputs["joint_ids"].tolist()]
    indices, _, values = resolve_matching_names_values(gear_ratio, selected_names)
    state = ctx.state["scaled_gears"]
    gears = state.new_ones(state.shape)
    gears[indices] = state.new_tensor(values)
    state.copy_(gears / gears.max())


@wp.func
def _power_consumption(
    world: int,
    actions: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    scaled_gears: wp.array(dtype=float),
) -> float:
    cost = float(0.0)
    for j in range(joint_ids.shape[0]):
        cost += wp.abs(actions[world, j] * joint_vel[world, joint_ids[j]] * scaled_gears[j])
    return cost


power_consumption = Term(
    torch=lambda actions, joint_vel, joint_ids, scaled_gears: (actions * joint_vel[:, joint_ids] * scaled_gears)
    .abs()
    .sum(dim=1),
    warp=_power_consumption,
    init=_init_gears,
    state={"scaled_gears": State(("joint_ids",), persistent=True)},
)


@wp.func
def _joint_pos_limits_penalty_ratio(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    joint_ids: wp.array(dtype=wp.int32),
    scaled_gears: wp.array(dtype=float),
    threshold: float,
) -> float:
    cost = float(0.0)
    for j in range(joint_ids.shape[0]):
        index = joint_ids[j]
        limit = soft_joint_pos_limits[world, index]
        position = 2.0 * (joint_pos[world, index] - 0.5 * (limit[0] + limit[1])) / (limit[1] - limit[0])
        violation = wp.abs(position) - threshold
        if violation > 0.0:
            cost += violation / (1.0 - threshold) * scaled_gears[j]
    return cost


def _joint_pos_limits_penalty_ratio_torch(joint_pos, soft_joint_pos_limits, joint_ids, scaled_gears, threshold):
    limits = soft_joint_pos_limits[:, joint_ids]
    position = math_utils.scale_transform(joint_pos[:, joint_ids], limits[..., 0], limits[..., 1])
    violation = (position.abs() - threshold).clamp(min=0.0) / (1.0 - threshold)
    return (violation * scaled_gears).sum(dim=1)


joint_pos_limits_penalty_ratio = Term(
    torch=_joint_pos_limits_penalty_ratio_torch,
    warp=_joint_pos_limits_penalty_ratio,
    init=_init_gears,
    state={"scaled_gears": State(("joint_ids",), persistent=True)},
)


@wp.func
def _terminated_penalty(world: int, terminated: wp.array(dtype=wp.bool), step_dt: float) -> float:
    return float(terminated[world]) / step_dt


terminated_penalty = Term(torch=lambda terminated, step_dt: terminated.float() / step_dt, warp=_terminated_penalty)


@wp.func
def _survival(world: int, time_out: wp.array(dtype=wp.bool)) -> float:
    return float(time_out[world])


survival = Term(torch=lambda time_out: time_out.float(), warp=_survival)
