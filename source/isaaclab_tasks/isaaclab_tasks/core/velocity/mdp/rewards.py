# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the velocity-tracking locomotion environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from isaaclab.utils.warp import raycast_mesh

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_variance(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, max_time: float = 0.5
) -> torch.Tensor:
    """Penalize an uneven swing/stance split between the feet.

    The penalty is the variance across feet of the last completed swing and stance durations [s],
    each clipped at ``max_time`` [s], and is zero for near-zero commands.

    Raises:
        RuntimeError: If ``sensor_cfg`` resolves to fewer than two bodies, where :func:`torch.var`
            would return ``nan``.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]
    # ``body_ids`` may be a slice, so count the bodies the selection actually resolved to
    num_feet = last_air_time.shape[1]
    if num_feet < 2:
        raise RuntimeError(
            f"'feet_air_time_variance' needs at least two bodies to take a variance across feet, but"
            f" '{sensor_cfg.name}' resolved to {num_feet}. Widen 'body_names' on its 'sensor_cfg' to"
            " cover every foot."
        )
    penalty = torch.var(torch.clip(last_air_time, max=max_time), dim=1) + torch.var(
        torch.clip(last_contact_time, max=max_time), dim=1
    )
    # no penalty for zero command
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return penalty


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = (
        contact_sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity-aligned robot frame with an exp kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.linalg.norm(command[:, :2], dim=1) < command_threshold)


def pelvis_height_deficit_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Squared shortfall of the root below ``target_height`` above the ground beneath it."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    clearance = asset.data.root_pos_w.torch[:, 2] - ground
    return torch.clamp(target_height - clearance, min=0.0).square()


def joint_deviation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared deviation of the selected joints from their default angles."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def body_ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared world-frame roll/pitch angular velocity of a named body."""
    asset = env.scene[asset_cfg.name]
    omega = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(omega[:, :2]), dim=1)


def root_downward_speed_l2(
    env, speed_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize world-frame downward root speed exceeding the limit [m/s]."""
    velocity_z = env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, 2]
    return torch.clamp(-velocity_z - speed_limit, min=0.0).square()


def sole_edge_distances(sole_points, mesh, height_tolerance=0.04, max_distance=0.25):
    """Estimate horizontal sole-corner overhang [m] beyond the supporting terrain plane."""
    starts = sole_points.clone()
    starts[..., 2] += 0.5
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.0
    hits, _, normals, _ = raycast_mesh(starts, directions, mesh, max_dist=2.0, return_normal=True)
    valid = torch.isfinite(hits).all(-1) & torch.isfinite(normals).all(-1)
    support = valid & ((hits[..., 2] - sole_points[..., 2]).abs() < height_tolerance) & (normals[..., 2] > 0.7)
    support_index = torch.where(support, hits[..., 2], -torch.inf).argmax(-1)
    gather = support_index[..., None, None].expand(*support_index.shape, 1, 3)
    anchor = torch.nan_to_num(hits.gather(-2, gather).squeeze(-2), posinf=0.0, neginf=0.0)
    normal = torch.nan_to_num(normals.gather(-2, gather).squeeze(-2), posinf=0.0, neginf=0.0)
    corner_indices = [0, 2, 6, 8]
    corners = sole_points[..., corner_indices, :]
    ground = hits[..., corner_indices, 2]
    plane_z = anchor[..., None, 2] - ((corners[..., :2] - anchor[..., None, :2]) * normal[..., None, :2]).sum(
        -1
    ) / normal[..., None, 2].clamp_min(0.7)
    outside = (plane_z - ground > height_tolerance) | ~valid[..., corner_indices]
    angles = torch.arange(16, device=sole_points.device, dtype=sole_points.dtype) * (2 * math.pi / 16)
    horizontal = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), -1)
    ray_start = corners.clone()
    ray_start[..., 2] = plane_z - 0.01
    ray_start = ray_start[..., None, :].expand(*corners.shape[:-1], 16, 3).contiguous()
    ray_direction = horizontal.expand_as(ray_start).contiguous()
    _, distances, _, _ = raycast_mesh(
        ray_start.reshape(-1, 16, 3),
        ray_direction.reshape(-1, 16, 3),
        mesh,
        max_dist=max_distance,
        return_distance=True,
    )
    distance = distances.amin(-1).reshape(*corners.shape[:-1]).clamp_max(max_distance)
    return torch.where(support.any(-1)[..., None], torch.where(outside, distance, 0.0), max_distance)


def feet_support_edge_distance(env, sensor_cfg, asset_cfg, sole_size, sole_offset) -> torch.Tensor:
    """Penalize mean corner overhang [m] for feet carrying more than 20 N."""
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    positions = robot.data.body_pos_w.torch[:, asset_cfg.body_ids]
    quaternions = robot.data.body_quat_w.torch[:, asset_cfg.body_ids]
    xy = torch.cartesian_prod(
        torch.linspace(-0.5, 0.5, 3, device=env.device), torch.linspace(-0.5, 0.5, 3, device=env.device)
    )
    points = torch.cat((xy, torch.full_like(xy[:, :1], -0.5)), -1)
    points = points * torch.tensor(sole_size, device=env.device) + torch.tensor(sole_offset, device=env.device)
    points = points.expand(*positions.shape[:-1], 9, 3)
    points = quat_apply(quaternions[..., None, :].expand(*points.shape[:-1], 4), points) + positions[..., None, :]
    scanner = env.scene["height_scanner"]
    key = (scanner.cfg.mesh_prim_paths[0], scanner.device)
    if key not in scanner.meshes:
        from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster

        BaseRayCaster._initialize_warp_meshes(scanner)
    mesh = scanner.meshes[key]
    distance = sole_edge_distances(points, mesh).mean(-1)
    contact = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > 20.0
    return (distance * contact).sum(-1)
