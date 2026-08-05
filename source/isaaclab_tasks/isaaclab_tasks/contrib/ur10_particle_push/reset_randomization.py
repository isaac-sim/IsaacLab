# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, deployment-facing reset randomization for UR10 particle push."""

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
    """Collision-screened robot poses sampled once at environment construction."""

    joint_position: torch.Tensor
    paddle_position_e: torch.Tensor
    curriculum_level: torch.Tensor
    source_pile_index: torch.Tensor

    @property
    def row_count(self) -> int:
        """Number of reset poses."""
        return int(self.joint_position.shape[0])


@dataclass(frozen=True)
class PushParticleResetState:
    """Particle state and semantic masks generated for one reset batch."""

    position_e: torch.Tensor
    focused_source_mask: torch.Tensor


def build_reset_pose_curriculum_levels(
    cfg: UR10ParticlePushEnvCfg,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the curriculum level assigned to every deterministic reset-pose row."""
    level_count = len(cfg.curriculum_pile_center_x)
    poses_per_level = cfg.reset_pose_count // level_count
    return torch.arange(level_count, device=device, dtype=torch.long).repeat_interleave(poses_per_level)


def build_reset_pose_source_pile_indices(
    cfg: UR10ParticlePushEnvCfg,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the source pile targeted by each deterministic reset-pose row."""
    levels = build_reset_pose_curriculum_levels(cfg, device="cpu").numpy()
    indices = np.zeros(cfg.reset_pose_count, dtype=np.int64)
    for level, pile_count in enumerate(cfg.curriculum_source_pile_count):
        level_rows = np.flatnonzero(levels == level)
        indices[level_rows] = np.arange(level_rows.size) % pile_count
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def build_reset_paddle_targets(
    cfg: UR10ParticlePushEnvCfg,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample deterministic upright paddle poses for every curriculum level.

    Each level shifts the paddle with its particle pile, preserving the same short approach
    distance. The first row per level is exact; remaining rows vary position and world-Z yaw by
    that level's randomization scale.
    """
    count = cfg.reset_pose_count
    levels = build_reset_pose_curriculum_levels(cfg, device="cpu").numpy()
    source_pile_indices = build_reset_pose_source_pile_indices(cfg, device="cpu").numpy()
    rng = np.random.default_rng(cfg.reset_seed)
    nominal = np.asarray(cfg.paddle_reset_center, dtype=np.float32)
    positions = np.broadcast_to(nominal, (count, 3)).copy()
    yaw = np.zeros(count, dtype=np.float32)
    for level in range(len(cfg.curriculum_pile_center_x)):
        level_rows = np.flatnonzero(levels == level)
        level_scale = np.float32(cfg.curriculum_randomization_scale[level])
        positions[level_rows, 0] += np.float32(cfg.curriculum_pile_center_x[level] - cfg.pile_nominal_center[0])
        pile_count = cfg.curriculum_source_pile_count[level]
        lateral_offset = np.float32(cfg.curriculum_source_lateral_offset[level])
        paddle_lateral_range = np.float32(
            cfg.reset_paddle_split_max_lateral_offset if pile_count == 2 else cfg.reset_paddle_max_lateral_offset
        )
        if pile_count == 2:
            positions[level_rows, 1] += np.where(source_pile_indices[level_rows] == 0, -lateral_offset, lateral_offset)
        # Keep one exact pose for every focus side and randomize the remaining rows locally.
        for pile_index in range(pile_count):
            pile_rows = level_rows[source_pile_indices[level_rows] == pile_index]
            randomized_rows = pile_rows[1:]
            if randomized_rows.size == 0:
                continue
            positions[randomized_rows, 0] += rng.uniform(
                cfg.reset_paddle_longitudinal_offset_range[0] * level_scale,
                cfg.reset_paddle_longitudinal_offset_range[1] * level_scale,
                randomized_rows.size,
            ).astype(np.float32)
            positions[randomized_rows, 1] += rng.uniform(
                -paddle_lateral_range * level_scale,
                paddle_lateral_range * level_scale,
                randomized_rows.size,
            ).astype(np.float32)
            yaw[randomized_rows] = rng.uniform(
                -cfg.reset_paddle_max_yaw * level_scale,
                cfg.reset_paddle_max_yaw * level_scale,
                randomized_rows.size,
            ).astype(np.float32)

    # Rz(yaw) @ Ry(pi / 2): local X remains vertical-down and local Z points toward +X at yaw=0.
    half_yaw = 0.5 * yaw
    sine = np.sin(half_yaw)
    cosine = np.cos(half_yaw)
    half_sqrt_two = np.float32(2.0**-0.5)
    quaternions = np.stack(
        (
            -half_sqrt_two * sine,
            half_sqrt_two * cosine,
            half_sqrt_two * sine,
            half_sqrt_two * cosine,
        ),
        axis=-1,
    ).astype(np.float32)
    return (
        torch.as_tensor(positions, dtype=torch.float32, device=device),
        torch.as_tensor(quaternions, dtype=torch.float32, device=device),
    )


def sample_correlated_particle_translation(
    paddle_position_xy: torch.Tensor,
    paddle_reset_xy: torch.Tensor,
    translation_lower_bound: torch.Tensor,
    translation_upper_bound: torch.Tensor,
    residual_half_range: torch.Tensor,
) -> torch.Tensor:
    """Sample bounded pile translations while preserving a local paddle-to-pile approach."""
    if paddle_position_xy.ndim != 2 or paddle_position_xy.shape[1] != 2:
        raise ValueError("paddle_position_xy must have shape (num_envs, 2).")
    env_count = paddle_position_xy.shape[0]
    expected_bounds_shape = (env_count, 2)
    if paddle_reset_xy.shape not in ((2,), expected_bounds_shape):
        raise ValueError("paddle_reset_xy must have shape (2,) or (num_envs, 2).")
    for name, value in (
        ("translation_lower_bound", translation_lower_bound),
        ("translation_upper_bound", translation_upper_bound),
        ("residual_half_range", residual_half_range),
    ):
        if value.shape not in ((2,), expected_bounds_shape):
            raise ValueError(f"{name} must have shape (2,) or {expected_bounds_shape}.")

    paddle_translation = paddle_position_xy - paddle_reset_xy
    residual_lower = torch.maximum(-residual_half_range, translation_lower_bound - paddle_translation)
    residual_upper = torch.minimum(residual_half_range, translation_upper_bound - paddle_translation)
    residual = residual_lower + torch.rand_like(paddle_position_xy) * (residual_upper - residual_lower)
    return paddle_translation + residual


def _pack_particle_group(
    rank: torch.Tensor,
    count: torch.Tensor,
    mask: torch.Tensor,
    center_xy: torch.Tensor,
    support_z: torch.Tensor,
    spacing: torch.Tensor,
    particle_jitter: torch.Tensor,
    yaw: torch.Tensor,
    *,
    vertical_cell_count: int | torch.Tensor,
    footprint_aspect_ratio: float | torch.Tensor,
) -> torch.Tensor:
    """Pack one masked group into a compact, bottom-supported particle lattice."""
    vertical_cell_count = torch.as_tensor(vertical_cell_count, dtype=torch.long, device=count.device)
    footprint_aspect_ratio = torch.as_tensor(
        footprint_aspect_ratio,
        dtype=spacing.dtype,
        device=count.device,
    )
    if vertical_cell_count.ndim == 0:
        vertical_cell_count = vertical_cell_count.expand_as(count)
    if footprint_aspect_ratio.ndim == 0:
        footprint_aspect_ratio = footprint_aspect_ratio.expand_as(count)
    if vertical_cell_count.shape != count.shape or footprint_aspect_ratio.shape != count.shape:
        raise ValueError("Per-group packing parameters must be scalar or have one value per environment.")

    safe_rank = rank.clamp_min(0)
    footprint_count = torch.div(
        count + vertical_cell_count - 1,
        vertical_cell_count,
        rounding_mode="floor",
    ).clamp_min(1)
    x_cell_count = torch.ceil(torch.sqrt(footprint_count.float() * footprint_aspect_ratio)).long().clamp_min(1)
    y_cell_count = torch.div(
        footprint_count + x_cell_count - 1,
        x_cell_count,
        rounding_mode="floor",
    ).clamp_min(1)

    vertical_index = safe_rank.remainder(vertical_cell_count[:, None])
    planar_index = torch.div(safe_rank, vertical_cell_count[:, None], rounding_mode="floor")
    y_index = planar_index.remainder(y_cell_count[:, None])
    x_index = torch.div(planar_index, y_cell_count[:, None], rounding_mode="floor")
    local_x = (x_index - 0.5 * (x_cell_count[:, None] - 1)) * spacing[0] + particle_jitter[..., 0]
    local_y = (y_index - 0.5 * (y_cell_count[:, None] - 1)) * spacing[1] + particle_jitter[..., 1]
    cosine = torch.cos(yaw)[:, None]
    sine = torch.sin(yaw)[:, None]
    position = torch.stack(
        (
            center_xy[:, None, 0] + cosine * local_x - sine * local_y,
            center_xy[:, None, 1] + sine * local_x + cosine * local_y,
            support_z[:, None] + vertical_index * spacing[2] + particle_jitter[..., 2],
        ),
        dim=-1,
    )

    # Correct support per group. Delivered and source piles sit on different physical floors.
    infinity = torch.full_like(position[..., 2], torch.inf)
    minimum_z = torch.where(mask, position[..., 2], infinity).amin(dim=1)
    correction = torch.where(mask.any(dim=1), support_z - minimum_z, torch.zeros_like(support_z))
    position[..., 2] += correction[:, None]
    return position


def build_staged_particle_reset(
    template_position_e: torch.Tensor,
    source_center_x: torch.Tensor,
    source_pile_count: torch.Tensor,
    source_lateral_offset: torch.Tensor,
    initial_bin_fraction: torch.Tensor,
    focused_source_pile_index: torch.Tensor,
    source_translation_xy: torch.Tensor,
    source_yaw: torch.Tensor,
    particle_jitter: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    *,
    source_vertical_cell_count: torch.Tensor | None = None,
    source_footprint_aspect_ratio: torch.Tensor | None = None,
) -> PushParticleResetState:
    """Build reverse-curriculum resets from delivered material and one or two source piles.

    Every world reuses the complete emitted lattice. Partial-progress states change only material
    placement, while particle count, spacing, volume, density, and PPC remain fixed.
    """
    if template_position_e.ndim != 3 or template_position_e.shape[-1] != 3:
        raise ValueError("template_position_e must have shape (num_envs, num_particles, 3).")
    env_count, particle_count, _ = template_position_e.shape
    if particle_count != math.prod(PILE_LATTICE_RESOLUTION):
        raise ValueError("template_position_e must contain the complete emitted particle lattice.")
    if particle_jitter.shape != template_position_e.shape:
        raise ValueError(
            f"particle_jitter must have shape {tuple(template_position_e.shape)}, got {tuple(particle_jitter.shape)}."
        )
    for name, value in (
        ("source_center_x", source_center_x),
        ("source_pile_count", source_pile_count),
        ("source_lateral_offset", source_lateral_offset),
        ("initial_bin_fraction", initial_bin_fraction),
        ("focused_source_pile_index", focused_source_pile_index),
        ("source_yaw", source_yaw),
    ):
        if value.shape != (env_count,):
            raise ValueError(f"{name} must have shape {(env_count,)}, got {tuple(value.shape)}.")
    if source_translation_xy.shape != (env_count, 2):
        raise ValueError(
            f"source_translation_xy must have shape {(env_count, 2)}, got {tuple(source_translation_xy.shape)}."
        )

    particle_rank = torch.arange(particle_count, device=template_position_e.device).expand(env_count, -1)
    delivered_count = torch.floor(particle_count * initial_bin_fraction).long().clamp(0, particle_count - 1)
    delivered_mask = particle_rank < delivered_count[:, None]
    source_mask = ~delivered_mask
    source_rank = particle_rank - delivered_count[:, None]
    source_group_index = source_rank.remainder(source_pile_count[:, None])
    source_group_rank = torch.div(source_rank, source_pile_count[:, None], rounding_mode="floor")
    focused_source_mask = source_mask & (source_group_index == focused_source_pile_index[:, None])

    spawn = cfg.scene.media.spawn
    extent = template_position_e.new_tensor(
        tuple(upper - lower for lower, upper in zip(spawn.lower, spawn.upper, strict=True))
    )
    default_source_aspect_ratio = (spawn.upper[0] - spawn.lower[0]) / (spawn.upper[1] - spawn.lower[1])
    lattice_resolution = template_position_e.new_tensor(PILE_LATTICE_RESOLUTION)
    spacing = extent / lattice_resolution
    vertical_cell_count = PILE_LATTICE_RESOLUTION[2]
    if source_vertical_cell_count is None:
        source_vertical_cell_count = torch.full(
            (env_count,),
            vertical_cell_count,
            dtype=torch.long,
            device=template_position_e.device,
        )
    if source_footprint_aspect_ratio is None:
        source_footprint_aspect_ratio = torch.full(
            (env_count,),
            default_source_aspect_ratio,
            dtype=template_position_e.dtype,
            device=template_position_e.device,
        )
    else:
        source_footprint_aspect_ratio = source_footprint_aspect_ratio.to(
            dtype=template_position_e.dtype,
            device=template_position_e.device,
        )
    source_vertical_cell_count = source_vertical_cell_count.to(
        dtype=torch.long,
        device=template_position_e.device,
    )
    if source_vertical_cell_count.shape != (env_count,) or source_footprint_aspect_ratio.shape != (env_count,):
        raise ValueError("Source packing profiles must contain one value per environment.")

    source_support_z = template_position_e[..., 2].amin(dim=1)
    bin_floor = cfg.scene.bin_floor
    bin_support_z = template_position_e.new_full(
        (env_count,),
        bin_floor.init_state.pos[2] + 0.5 * bin_floor.spawn.size[2] + 0.5 * spawn.voxel_size,
    )
    bin_center_xy = template_position_e.new_tensor(
        (
            0.5 * sum(cfg.bin_inner_x_bounds),
            0.5 * sum(cfg.bin_inner_y_bounds),
        )
    ).expand(env_count, -1)
    delivered_position = _pack_particle_group(
        particle_rank,
        delivered_count,
        delivered_mask,
        bin_center_xy,
        bin_support_z,
        spacing,
        particle_jitter,
        torch.zeros_like(source_yaw),
        vertical_cell_count=vertical_cell_count,
        footprint_aspect_ratio=(cfg.bin_inner_x_bounds[1] - cfg.bin_inner_x_bounds[0])
        / (cfg.bin_inner_y_bounds[1] - cfg.bin_inner_y_bounds[0]),
    )

    position_e = torch.where(delivered_mask[..., None], delivered_position, template_position_e)
    source_center_x = source_center_x + source_translation_xy[:, 0]
    source_midpoint_y = source_translation_xy[:, 1]
    source_count = particle_count - delivered_count
    for pile_index in range(2):
        pile_enabled = source_pile_count > pile_index
        pile_mask = source_mask & pile_enabled[:, None] & (source_group_index == pile_index)
        pile_count = torch.div(
            source_count + source_pile_count - 1 - pile_index,
            source_pile_count,
            rounding_mode="floor",
        ).clamp_min(0)
        lateral_sign = -1.0 if pile_index == 0 else 1.0
        pile_center_xy = torch.stack(
            (
                source_center_x,
                source_midpoint_y + lateral_sign * source_lateral_offset,
            ),
            dim=1,
        )
        pile_position = _pack_particle_group(
            source_group_rank,
            pile_count,
            pile_mask,
            pile_center_xy,
            source_support_z,
            spacing,
            particle_jitter,
            source_yaw,
            vertical_cell_count=source_vertical_cell_count,
            footprint_aspect_ratio=source_footprint_aspect_ratio,
        )
        position_e = torch.where(pile_mask[..., None], pile_position, position_e)

    return PushParticleResetState(
        position_e=position_e,
        focused_source_mask=focused_source_mask,
    )
