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

from ..conveyor_warehouse_geometry import warehouse_circulation_edges
from .commands import ConveyorTransferCommand, ConveyorTransferCommandCfg
from .cube_state import cube_values
from .kinematics import end_effector_pose
from .reset_events import CUBE_REST_Z, CUBE_SIZE

if TYPE_CHECKING:
    from ..conveyor_franka_warehouse_env import ConveyorFrankaWarehouseEnv


def reset_inventory(env: ConveyorFrankaWarehouseEnv, env_ids: torch.Tensor) -> None:
    """Restore the home robot and one mixed inventory directly, without racetrack recipes."""
    pool = env.conveyor_cube_pool
    pool.reset(env_ids)
    robot = env.scene["robot"]
    positions = robot.data.default_joint_pos.torch[env_ids]
    velocities = torch.zeros_like(positions)
    robot.write_joint_position_to_sim_index(position=positions, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=velocities, env_ids=env_ids)
    robot.actuators.target_command.set_position_index(value=positions, env_ids=env_ids)
    robot.actuators.target_command.set_velocity_index(value=velocities, env_ids=env_ids)
    if env.cfg.commands.transfer.randomize_arrivals:
        assignments = torch.rand((len(env_ids), len(pool.assets)), device=env.device).argsort(dim=1)
    else:
        assignments = torch.arange(len(pool.assets), device=env.device).expand(len(env_ids), -1)
    for cube_id, cube in enumerate(pool.assets):
        pose = cube.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = env._parcel_start_positions[assignments[:, cube_id]] + env.scene.env_origins[env_ids]
        cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((len(env_ids), 6)), env_ids=env_ids)


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
        edges = torch.tensor(warehouse_circulation_edges(), device=self.device)
        self._deck_starts = edges[:, :3]
        self._deck_vectors = edges[:, 3:6] - self._deck_starts
        self._deck_lengths_xy = self._deck_vectors[:, :2].square().sum(dim=-1).clamp_min(1.0e-8)
        self._deck_half_widths_squared = (0.5 * edges[:, 6]).square()

    def evaluate(self) -> None:
        """Credit stable placements only while the dispatcher owns an active transfer."""
        self.new_success[~self.has_target] = False
        self.is_success[~self.has_target] = False
        self._last_evaluation_steps[~self.has_target] = self._env.episode_length_buf[~self.has_target]
        super().evaluate()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self._resampling_from_reset:
            idle = torch.zeros_like(self.target_cube_ids[env_ids])
            self._assign_goal(env_ids, idle, idle)
            self.subgoal_start_steps[env_ids] = 0
        self.has_target[env_ids] = False
        self.held_cube_ids[env_ids] = -1
        self.pending_success[env_ids] = False

    def _physical_cube_acquisition_mask(self, active_position_w: torch.Tensor) -> torch.Tensor:
        """Pin a lifted, closed-gripper parcel near the tool while the arm transfers it."""
        tool_position, _ = end_effector_pose(self._env)
        finger_positions = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        local_cube_z = active_position_w[:, 2] - self._env.scene.env_origins[:, 2]
        return (
            (local_cube_z >= CUBE_REST_Z + 0.025)
            & (torch.linalg.vector_norm(active_position_w - tool_position, dim=1) <= 0.075)
            & (torch.amax(finger_positions, dim=1) <= 0.030)
        )

    def _update_command(self) -> None:
        """Publish slot assignments before the manager computes the next policy observation."""
        env = self._env
        pool = env.conveyor_cube_pool
        positions_w = cube_values(env, "root_pos_w", all_cubes=True)
        positions = positions_w - env.scene.env_origins[:, None]
        local = env._in_workcell(positions)
        rows = env.scene._ALL_INDICES
        physical_target = pool.slot_ids[rows, self.target_cube_ids]
        held = self._physical_cube_acquisition_mask(positions_w[rows, physical_target]) & self.has_target
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
        env_ids = eligible.nonzero(as_tuple=False).flatten()
        slots = choices[env_ids]
        physical_ids = pool.slot_ids[env_ids, slots]
        self._assign_goal(env_ids, slots, (positions[env_ids, physical_ids, 1] < 0.0).long())
        self.has_target[env_ids] = True
        self.command_counter[env_ids] += 1
        # Correct-side inventory must rest on a circulating deck, including its elevated return.
        velocities = cube_values(env, "root_lin_vel_w", all_cubes=True)
        offsets = positions[..., None, :2] - self._deck_starts[:, :2]
        along = ((offsets * self._deck_vectors[:, :2]).sum(dim=-1) / self._deck_lengths_xy).clamp(0.0, 1.0)
        nearest = self._deck_starts + along[..., None] * self._deck_vectors
        lateral_distance = (positions[..., None, :2] - nearest[..., :2]).square().sum(dim=-1)
        clearance = positions[..., None, 2] - 0.5 * CUBE_SIZE - nearest[..., 2]
        on_loop = ((lateral_distance <= self._deck_half_widths_squared) & (clearance.abs() < 0.01)).any(dim=-1) & (
            velocities[..., 2].abs() < 0.15
        )
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
