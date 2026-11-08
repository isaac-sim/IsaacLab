# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared obstacle footprints, clearances and requested-path geometry."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class ObstacleFootprintCfg:
    """Static planar collision footprint in the scene asset's local yaw frame.

    Specify exactly one of full box ``size`` [m] or circle ``radius`` [m].
    ``center`` accounts for USD roots that are not at the footprint center.
    Mesh assets require a measured/authored footprint; this is not automatic mesh fitting.
    """

    size: tuple[float, float] | None = None
    radius: float | None = None
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if (self.size is None) == (self.radius is None):
            raise ValueError("Specify exactly one box size or circle radius.")
        dimensions = self.size if self.size is not None else (self.radius,)
        if (self.size is not None and len(self.size) != 2) or not all(
            math.isfinite(value) and value > 0.0 for value in dimensions
        ):
            raise ValueError("Footprint dimensions must be finite and positive.")
        if len(self.center) != 2 or not all(math.isfinite(value) for value in self.center):
            raise ValueError("Footprint center must contain two finite coordinates.")


def obstacle_geometry(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World centers/yaws and local bounds in configured critic-observation order."""
    centers, rotations, half_sizes, radii = [], [], [], []
    for name, footprint in env.cfg.obstacle_footprints.items():
        obstacle = env.scene[name]
        rotation = yaw_quat(obstacle.data.root_quat_w.torch)
        offset = torch.tensor((*footprint.center, 0.0), device=env.device).expand_as(obstacle.data.root_pos_w.torch)
        centers.append(obstacle.data.root_pos_w.torch + quat_apply(rotation, offset))
        rotations.append(rotation)
        half_sizes.append(
            tuple(value / 2 for value in footprint.size) if footprint.size is not None else (footprint.radius,) * 2
        )
        radii.append(footprint.radius or 0.0)
    return (
        torch.stack(centers, dim=1),
        torch.stack(rotations, dim=1),
        torch.tensor(half_sizes, device=env.device),
        torch.tensor(radii, device=env.device),
    )


def obstacle_point_clearances(env: ManagerBasedRLEnv, points: torch.Tensor, robot_radius: float) -> torch.Tensor:
    """Signed planar clearance for each (environment, sample point, obstacle)."""
    centers, rotations, half_sizes, radii = obstacle_geometry(env)
    # Pair each sampled point with every obstacle before transforming to obstacle-local XY.
    point_offsets_w = points[:, :, None, :] - centers[:, None, :, :]
    obstacle_rotations = rotations[:, None].expand(-1, points.shape[1], -1, -1)
    points_in_obstacle = quat_apply_inverse(obstacle_rotations, point_offsets_w)[..., :2]
    distance_to_box_faces = points_in_obstacle.abs() - half_sizes
    outside_distance = distance_to_box_faces.clamp_min(0.0).norm(dim=-1)
    inside_distance = distance_to_box_faces.amax(dim=-1).clamp_max(0.0)
    box_distance = outside_distance + inside_distance
    circle_distance = points_in_obstacle.norm(dim=-1) - radii
    clearance = torch.where(radii > 0.0, circle_distance, box_distance) - robot_radius
    return torch.nan_to_num(clearance, nan=-robot_radius, posinf=4.0, neginf=-robot_radius)


def obstacle_clearances(env: ManagerBasedRLEnv, robot_radius: float = 0.12) -> torch.Tensor:
    """Return signed footprint-to-obstacle clearances [m], shape (N, number of configured obstacles)."""
    return obstacle_point_clearances(env, env.scene["robot"].data.root_link_pos_w.torch[:, None, :], robot_radius)[:, 0]


def requested_path_clearance(
    env: ManagerBasedRLEnv,
    robot_radius: float = 0.12,
    lookahead_distance: float = 0.5,
    num_samples: int = 6,
) -> torch.Tensor:
    """Conservative clearance [m] along the requested constant body-twist motion.

    Samples span a fixed arc length, including the requested yaw rate. Subtracting
    half the arc spacing accounts for obstacles between samples: signed distance
    is 1-Lipschitz along the path. Stationary requests, including pure turns, use
    the current circular footprint.
    """
    if num_samples < 2 or lookahead_distance <= 0.0:
        raise ValueError("Path clearance needs at least two samples and a positive lookahead distance.")
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    travel_direction_w = torch.zeros_like(command)
    speed = command[:, :2].norm(dim=-1)
    moving = speed > 1.0e-6
    travel_direction_w[:, :2] = torch.where(moving[:, None], command[:, :2] / speed.clamp_min(1.0e-6)[:, None], 0.0)
    travel_direction_w = quat_apply(yaw_quat(robot.data.root_link_quat_w.torch), travel_direction_w)
    arc_distances = torch.linspace(0.0, lookahead_distance, num_samples, device=env.device)
    yaw_per_metre = torch.where(moving, command[:, 2] / speed.clamp_min(1.0e-6), 0.0)
    # Integrate the body-twist command at each arc distance: (environment, sample).
    half_turn_angle = 0.5 * yaw_per_metre[:, None] * arc_distances[None, :]
    cos_half, sin_half = half_turn_angle.cos(), half_turn_angle.sin()
    path_offsets_w = torch.stack(
        (
            travel_direction_w[:, 0, None] * cos_half - travel_direction_w[:, 1, None] * sin_half,
            travel_direction_w[:, 0, None] * sin_half + travel_direction_w[:, 1, None] * cos_half,
            torch.zeros_like(half_turn_angle),
        ),
        dim=-1,
    )
    # The half-angle sinc form stays finite and becomes a straight line at zero yaw.
    path_offsets_w *= (arc_distances[None, :] * torch.sinc(half_turn_angle / torch.pi))[..., None]
    path_points_w = robot.data.root_link_pos_w.torch[:, None, :] + path_offsets_w
    clearance = obstacle_point_clearances(env, path_points_w, robot_radius).flatten(start_dim=1).amin(dim=-1)
    sampling_margin = moving * (lookahead_distance / (2 * (num_samples - 1)))
    return clearance - sampling_margin
