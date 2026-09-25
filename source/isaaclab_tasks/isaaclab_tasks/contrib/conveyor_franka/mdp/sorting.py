# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parcel-class dispatch for the warehouse's unchanged four-slot transfer policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

from ..conveyor_cube_pool import cube_values
from .commands import ConveyorTransferCommand, ConveyorTransferCommandCfg
from .rewards import physical_cube_acquisition_mask

if TYPE_CHECKING:
    from ..conveyor_franka_warehouse_env import ConveyorFrankaWarehouseEnv


class ConveyorSortCommand(ConveyorTransferCommand):
    """Transfer misplaced cartons and leave correctly sorted inventory circulating.

    Physical parcel IDs carry immutable destination classes. The dispatcher presents
    one wrong-lane arrival through the pretrained policy's existing cube/side command;
    color is a visual class label, not an additional policy observation.
    """

    cfg: ConveyorSortCommandCfg

    def __init__(self, cfg: ConveyorSortCommandCfg, env: ConveyorFrankaWarehouseEnv) -> None:
        super().__init__(cfg, env)
        if not 0.14 <= cfg.pickup_x_range[0] < cfg.pickup_x_range[1] <= 1.02:
            raise ValueError("The sorting pickup window must lie within the original working straight [0.14, 1.02] m.")
        if len(cfg.parcel_destinations) != len(env.conveyor_cube_pool.assets) or set(cfg.parcel_destinations) != {0, 1}:
            raise ValueError("Parcel destinations must match the physical pool and include both conveyor IDs, 0 and 1.")
        if len(cfg.parcel_colors) != len(cfg.parcel_destinations):
            raise ValueError("Each physical parcel requires a color and a destination.")
        for color in set(cfg.parcel_colors):
            destinations = {
                side for shade, side in zip(cfg.parcel_colors, cfg.parcel_destinations, strict=True) if shade == color
            }
            if len(destinations) != 1:
                raise ValueError(f"All {color} parcels must share one destination conveyor.")
        self.parcel_destinations = torch.tensor(cfg.parcel_destinations, device=self.device)
        self.has_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.metrics["sorted_parcels"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["batch_complete"] = torch.zeros(self.num_envs, device=self.device)

    def evaluate(self) -> None:
        """Credit stable placements only while the dispatcher owns an active transfer."""
        self.new_success[~self.has_target] = False
        self.is_success[~self.has_target] = False
        self._last_evaluation_steps[~self.has_target] = self._env.episode_length_buf[~self.has_target]
        super().evaluate()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # Reset metadata remains compatible with the shared reward and observation terms.
        if self._resampling_from_reset:
            super()._resample_command(env_ids)
        self.has_target[env_ids] = False
        self.held_cube_ids[env_ids] = -1
        self.pending_success[env_ids] = False

    def _update_command(self) -> None:
        """Publish slot assignments before the manager computes the next policy observation."""
        env = self._env
        pool = env.conveyor_cube_pool
        positions = cube_values(env, "root_pos_w", all_cubes=True) - env.scene.env_origins[:, None]
        local = env._in_workcell(positions)
        rows = torch.arange(self.num_envs, device=self.device)
        physical_target = pool.slot_ids[rows, self.target_cube_ids]
        held = physical_cube_acquisition_mask(env, command=self) & self.has_target
        self.has_target &= ~self.pending_success & (local[rows, physical_target] | held)
        self.pending_success.zero_()
        wrong_lane = (positions[..., 1] < 0).long() != self.parcel_destinations
        candidates = (
            wrong_lane
            & (positions[..., 0] > self.cfg.pickup_x_range[0])
            & (positions[..., 0] < self.cfg.pickup_x_range[1])
            & (positions[..., 1].abs() > 0.20)
            & (positions[..., 1].abs() < 0.36)
            & (positions[..., 2] > 0.04)
            & (positions[..., 2] < 0.10)
        )
        pool.refresh(positions, local, candidates, self.target_cube_ids, self.has_target)
        available = candidates.gather(1, pool.slot_ids)
        eligible = available.any(dim=1) & ~self.has_target
        choices = torch.where(available, positions[..., 0].gather(1, pool.slot_ids), -torch.inf).argmax(dim=1)
        for slot in range(4):
            env_ids = torch.where(eligible & (choices == slot))[0]
            if env_ids.numel():
                self.set_goal(slot, env_ids)
                self.has_target[env_ids] = True
                self.command_counter[env_ids] += 1
        # A class is counted only after leaving the elevated supply and landing on a loop.
        velocities = cube_values(env, "root_lin_vel_w", all_cubes=True)
        on_loop = (positions[..., 2] > 0.04) & (positions[..., 2] < 0.20) & (velocities[..., 2].abs() < 0.15)
        active = torch.zeros_like(on_loop).scatter_(
            1, pool.slot_ids[rows, self.target_cube_ids, None], self.has_target[:, None]
        )
        on_loop &= ~active
        sorted_parcels = (~wrong_lane & on_loop).sum(dim=1)
        self.metrics["sorted_parcels"].copy_(sorted_parcels)
        self.metrics["batch_complete"].copy_(sorted_parcels == len(pool.assets))


@configclass
class ConveyorSortCommandCfg(ConveyorTransferCommandCfg):
    """Batch sorting with fixed physical classes and checkpoint-compatible commands."""

    class_type: type[ConveyorSortCommand] | str = "{DIR}.sorting:ConveyorSortCommand"

    parcel_destinations: tuple[int, ...] = (0, 1) * 12
    """Destination per physical parcel: 0 is the positive-Y loop, 1 the negative-Y loop."""

    parcel_colors: tuple[str, ...] = ("blue", "orange", "green", "purple") * 6
    """Authored color per physical parcel; blue, orange, green, and purple are available."""

    randomize_arrivals: bool = True
    """Shuffle physical parcel identities across authored start positions at each reset."""

    pickup_x_range: tuple[float, float] = (0.35, 1.02)
    """Longitudinal arrival window in policy workspace coordinates [m]."""
