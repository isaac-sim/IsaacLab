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


def feet_lateral_distance_l2(env: ManagerBasedRLEnv, target_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize squared error from the desired signed lateral foot separation [m^2].

    ``asset_cfg`` must select exactly two feet in left-to-right order. Separation is measured
    in the root's yaw frame against ``target_distance`` [m], leaving forward stride and
    terrain-dependent foot heights unconstrained. Crossed feet have negative separation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, :]
    if foot_pos.shape[1] != 2:
        raise RuntimeError("'feet_lateral_distance_l2' requires exactly two feet in left-to-right order.")
    separation = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), foot_pos[:, 0] - foot_pos[:, 1])
    return torch.square(separation[:, 1] - target_distance)


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
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    overstay_scale: float = 0.0,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    A positive ``overstay_scale`` subtracts the total ongoing air time beyond ``threshold``.
    This cost also applies during double flight and can make the reward negative. Its default
    of zero preserves the positive-only reward.

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
    if overstay_scale > 0.0:
        reward -= overstay_scale * (air_time - threshold).clamp(min=0.0).sum(dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_flight(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize simultaneous flight of all selected feet.

    Unlike swing-duration rewards, this term also penalizes hopping in place. The contact
    sensor must track air time, and ``sensor_cfg`` must select the robot's feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.all(air_time > 0.0, dim=1).float()


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


def stance_knee_extension_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    terrain_cfg: SceneEntityCfg | None,
    max_flexion: float = 0.35,
    contact_delay: float = 0.08,
    terrain_radius: float = 0.35,
    relief_range: tuple[float, float] = (0.02, 0.12),
    standing_scale: float = 0.0,
    standing_max_flexion: float | None = None,
) -> torch.Tensor:
    """Penalize excessive knee flexion during established stance on locally level ground.

    Knee joints and contact bodies must be ordered identically. ``max_flexion`` [rad]
    leaves a soft knee; ``contact_delay`` [s] allows landing absorption. Terrain relief
    within ``terrain_radius`` [m] fades the penalty between ``relief_range`` [m].
    Fewer than three valid local hits disables it. Swing knees remain unconstrained.
    ``standing_scale`` applies a fraction of this cost when translation and yaw commands are small.
    ``standing_max_flexion`` optionally sets a separate threshold [rad] for these commands.
    """
    asset = env.scene[asset_cfg.name]
    contact_time = env.scene[sensor_cfg.name].data.current_contact_time.torch[:, sensor_cfg.body_ids]
    stance = contact_time > contact_delay
    flexion = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)
    walking = command[:, :2].norm(dim=1) > 0.1
    stationary = (command[:, :2].norm(dim=1) < 0.1) & (command[:, 2].abs() < 0.15)
    threshold = max_flexion
    if standing_max_flexion is not None:
        threshold = torch.where(stationary, standing_max_flexion, max_flexion).unsqueeze(1)
    cost = (torch.clamp(flexion - threshold, min=0.0).square() * stance).sum(dim=1)
    cost = cost / stance.sum(dim=1).clamp(min=1)

    terrain_weight = 1.0
    if terrain_cfg is not None:
        hits = env.scene[terrain_cfg.name].data.ray_hits_w.torch
        distance = (hits[..., :2] - asset.data.root_pos_w.torch[:, None, :2]).square().sum(dim=-1)
        valid = torch.isfinite(hits).all(dim=-1) & (distance <= terrain_radius**2)
        highest = torch.where(valid, hits[..., 2], -torch.inf).amax(dim=1)
        lowest = torch.where(valid, hits[..., 2], torch.inf).amin(dim=1)
        relief = highest - lowest
        terrain_weight = ((relief_range[1] - relief) / (relief_range[1] - relief_range[0])).clamp(0.0, 1.0)
        terrain_weight = torch.where(valid.sum(dim=1) >= 3, terrain_weight, 0.0)
    return cost * terrain_weight * torch.where(stationary, standing_scale, walking.float())


def walking_yaw_momentum_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    turn_std: float = 0.3,
) -> torch.Tensor:
    """Penalize whole-body centroidal yaw momentum during walking.

    Returns squared momentum [kg^2 m^4/s^2], including orbital and spin components.
    Body inertias are expressed about each COM in the body link frame.
    Current masses and inertias include dynamics randomization.
    ``turn_std`` [rad/s] attenuates the penalty during commanded turns, allowing
    heading control to generate angular momentum. All bodies contribute.
    """
    data = env.scene[asset_cfg.name].data
    mass = data.body_mass.torch.unsqueeze(-1)
    pos = data.body_com_pos_w.torch
    vel = data.body_com_lin_vel_w.torch
    total_mass = mass.sum(dim=1, keepdim=True)
    com = (mass * pos).sum(dim=1, keepdim=True) / total_mass
    com_vel = (mass * vel).sum(dim=1, keepdim=True) / total_mass
    orbital = torch.cross(pos - com, mass * (vel - com_vel), dim=-1)

    quat = data.body_link_quat_w.torch
    omega_local = quat_apply_inverse(quat, data.body_com_ang_vel_w.torch)
    inertia = data.body_inertia.torch.reshape(*pos.shape[:2], 3, 3)
    spin_local = (inertia @ omega_local.unsqueeze(-1)).squeeze(-1)
    spin = quat_apply(quat, spin_local)
    momentum_z = (orbital + spin).sum(dim=1)[:, 2]
    command = env.command_manager.get_command(command_name)
    walking = command[:, :2].norm(dim=1) > 0.1
    turn_weight = torch.exp(-(command[:, 2] / turn_std).square())
    return momentum_z.square() * walking * turn_weight
