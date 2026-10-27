# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset randomization for UR10 particle push."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .ur10_particle_push_env_cfg import PILE_LATTICE_RESOLUTION

if TYPE_CHECKING:
    from .ur10_particle_push_env_cfg import UR10ParticlePushEnvCfg


@dataclass(frozen=True)
class PushResetPoseBank:
    """Collision-screened robot starts paired with nearby pile centers."""

    joint_position: torch.Tensor
    pile_center_xy: torch.Tensor
    curriculum_level: torch.Tensor

    @property
    def row_count(self) -> int:
        """Number of reset poses."""
        return int(self.joint_position.shape[0])


def build_reset_targets(
    cfg: UR10ParticlePushEnvCfg,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic, paired paddle and pile targets for reset-time sampling.

    Newton IK is intentionally solved once during environment construction. At reset time, the
    Isaac Lab event term samples this collision-screened bank while particle shape, yaw, and
    sub-cell positions remain continuously randomized.
    """
    count = cfg.reset_pose_count
    level_count = len(cfg.reset_randomization_scales)
    poses_per_level = count // level_count
    rng = np.random.default_rng(cfg.reset_seed)
    curriculum_level = np.arange(count, dtype=np.int64) // poses_per_level
    randomization_scale = np.asarray(cfg.reset_randomization_scales, dtype=np.float32)[curriculum_level]
    pile_center_x = np.asarray(cfg.reset_pile_center_x, dtype=np.float32)[curriculum_level]
    paddle_center = np.broadcast_to(np.asarray(cfg.paddle_reset_center, dtype=np.float32), (count, 3)).copy()
    paddle_center[:, 0] += pile_center_x - np.float32(cfg.pile_nominal_center[0])
    paddle_position = paddle_center.copy()
    pile_center_xy = np.empty((count, 2), dtype=np.float32)
    paddle_yaw = np.zeros(count, dtype=np.float32)

    paddle_position[:, 0] += randomization_scale * rng.uniform(
        *cfg.reset_paddle_longitudinal_offset_range,
        count,
    ).astype(np.float32)
    paddle_position[:, 1] += randomization_scale * rng.uniform(
        -cfg.reset_paddle_max_lateral_offset,
        cfg.reset_paddle_max_lateral_offset,
        count,
    ).astype(np.float32)
    paddle_yaw[:] = randomization_scale * rng.uniform(
        -cfg.reset_paddle_max_yaw,
        cfg.reset_paddle_max_yaw,
        count,
    ).astype(np.float32)
    nominal_distance = np.float32(cfg.pile_nominal_center[0] - cfg.paddle_reset_center[0])
    sampled_distance = rng.uniform(
        *cfg.reset_pile_paddle_distance_range,
        count,
    ).astype(np.float32)
    pile_center_xy[:, 0] = (
        paddle_position[:, 0] + nominal_distance + randomization_scale * (sampled_distance - nominal_distance)
    )
    pile_center_xy[:, 1] = paddle_position[:, 1] + randomization_scale * rng.uniform(
        *cfg.reset_pile_paddle_lateral_offset_range,
        count,
    ).astype(np.float32)

    # Retain one exact pose per level for deterministic smoke tests and playback.
    for level in range(level_count):
        canonical_row = level * poses_per_level
        paddle_position[canonical_row] = paddle_center[canonical_row]
        paddle_yaw[canonical_row] = 0.0
        pile_center_xy[canonical_row] = (pile_center_x[canonical_row], cfg.pile_nominal_center[1])

    # Rz(yaw) @ Ry(pi / 2): local X remains vertical-down and local Z points toward +X at yaw=0.
    half_yaw = 0.5 * paddle_yaw
    sine = np.sin(half_yaw)
    cosine = np.cos(half_yaw)
    half_sqrt_two = np.float32(2.0**-0.5)
    paddle_quaternion = np.stack(
        (
            -half_sqrt_two * sine,
            half_sqrt_two * cosine,
            half_sqrt_two * sine,
            half_sqrt_two * cosine,
        ),
        axis=-1,
    ).astype(np.float32)
    return (
        torch.as_tensor(paddle_position, dtype=torch.float32, device=device),
        torch.as_tensor(paddle_quaternion, dtype=torch.float32, device=device),
        torch.as_tensor(pile_center_xy, dtype=torch.float32, device=device),
        torch.as_tensor(curriculum_level, dtype=torch.long, device=device),
    )


def build_particle_reset(
    template_position_e: torch.Tensor,
    pile_center_xy: torch.Tensor,
    pile_yaw: torch.Tensor,
    particle_jitter: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    *,
    vertical_cell_count: torch.Tensor,
    footprint_aspect_ratio: torch.Tensor,
) -> torch.Tensor:
    """Pack every environment's fixed particle population into one supported pile."""
    if template_position_e.ndim != 3 or template_position_e.shape[-1] != 3:
        raise ValueError("template_position_e must have shape (num_envs, num_particles, 3).")
    env_count, particle_count, _ = template_position_e.shape
    if particle_count != math.prod(PILE_LATTICE_RESOLUTION):
        raise ValueError("template_position_e must contain the complete emitted particle lattice.")
    if pile_center_xy.shape != (env_count, 2):
        raise ValueError(f"pile_center_xy must have shape {(env_count, 2)}.")
    if pile_yaw.shape != (env_count,):
        raise ValueError(f"pile_yaw must have shape {(env_count,)}.")
    if particle_jitter.shape != template_position_e.shape:
        raise ValueError(f"particle_jitter must have shape {tuple(template_position_e.shape)}.")
    if vertical_cell_count.shape != (env_count,) or footprint_aspect_ratio.shape != (env_count,):
        raise ValueError("Pile packing profiles must contain one value per environment.")

    spawn = cfg.scene.media.spawn
    extent = template_position_e.new_tensor(
        tuple(upper - lower for lower, upper in zip(spawn.lower, spawn.upper, strict=True))
    )
    spacing = extent / template_position_e.new_tensor(PILE_LATTICE_RESOLUTION)
    rank = torch.arange(particle_count, device=template_position_e.device).expand(env_count, -1)
    vertical_cell_count = vertical_cell_count.to(dtype=torch.long, device=template_position_e.device)
    footprint_aspect_ratio = footprint_aspect_ratio.to(
        dtype=template_position_e.dtype,
        device=template_position_e.device,
    )
    footprint_count = torch.div(
        particle_count + vertical_cell_count - 1,
        vertical_cell_count,
        rounding_mode="floor",
    )
    x_cell_count = torch.ceil(torch.sqrt(footprint_count.float() * footprint_aspect_ratio)).long().clamp_min(1)
    y_cell_count = torch.div(
        footprint_count + x_cell_count - 1,
        x_cell_count,
        rounding_mode="floor",
    ).clamp_min(1)

    vertical_index = rank.remainder(vertical_cell_count[:, None])
    planar_index = torch.div(rank, vertical_cell_count[:, None], rounding_mode="floor")
    y_index = planar_index.remainder(y_cell_count[:, None])
    x_index = torch.div(planar_index, y_cell_count[:, None], rounding_mode="floor")
    local_x = (x_index - 0.5 * (x_cell_count[:, None] - 1)) * spacing[0] + particle_jitter[..., 0]
    local_y = (y_index - 0.5 * (y_cell_count[:, None] - 1)) * spacing[1] + particle_jitter[..., 1]
    cosine = torch.cos(pile_yaw)[:, None]
    sine = torch.sin(pile_yaw)[:, None]
    support_z = template_position_e[..., 2].amin(dim=1)
    position_e = torch.stack(
        (
            pile_center_xy[:, None, 0] + cosine * local_x - sine * local_y,
            pile_center_xy[:, None, 1] + sine * local_x + cosine * local_y,
            support_z[:, None] + vertical_index * spacing[2] + particle_jitter[..., 2],
        ),
        dim=-1,
    )

    # Jitter is symmetric, so translate each pile back onto its physical support plane.
    position_e[..., 2] += (support_z - position_e[..., 2].amin(dim=1))[:, None]
    return position_e
