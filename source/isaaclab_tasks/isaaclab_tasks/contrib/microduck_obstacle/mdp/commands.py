# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck obstacle-navigation commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand, MicroDuckVelocityCommandCfg

from .events import OBSTACLE_CELLS, OBSTACLE_LAYOUT_NAMES
from .geometry import obstacle_clearances, requested_path_clearance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def navigation_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Actual requested-direction progress, bounded to [-1, 1]; zero for standing."""
    command = env.command_manager.get_command("base_velocity")[:, :2]
    speed = command.norm(dim=-1)
    robot = env.scene["robot"]
    velocity = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch
    )
    progress = (velocity[:, :2] * command).sum(dim=-1) / speed.square().clamp_min(0.05**2)
    return torch.nan_to_num(torch.where(speed >= 0.05, progress.clamp(-1.0, 1.0), 0.0))


class MicroDuckNavigationCommand(MicroDuckVelocityCommand):
    """Operator commands with separate tracking, safety, and progress diagnostics.

    Samples are accumulated only on completed non-terminal control steps. Contact,
    fall, and boundary rates use the terminal flags retained until manager reset;
    scene reset events have already moved the robot by that point.
    """

    def __init__(self, cfg: MicroDuckNavigationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        del self.metrics["success_rate"]
        if not 0.0 <= cfg.rel_destination_envs <= 1.0:
            raise ValueError("rel_destination_envs must be in [0, 1].")
        self.is_destination_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.destination_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.destination_speed = torch.zeros(self.num_envs, device=self.device)
        self.destination_reached = torch.zeros_like(self.is_destination_env)
        self._navigation_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("clearance", "moving", "blocked", "clear", "progress", "stopped", "distance")
        }
        for name in (
            "tracking_success_rate",
            "collision_rate",
            "fall_rate",
            "field_boundary_rate",
            "horizon_survival_rate",
            "mean_clearance_m",
            "blocked_fraction",
            "clear_path_fraction",
            "clear_path_progress_ratio",
            "unnecessary_stop_fraction",
            "requested_direction_distance_m",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        super()._resample_command(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        # Episode choice and world target survive the joystick's 3-6 s resampling.
        fresh = ids[self.command_counter[ids] == 0]
        self.is_destination_env[fresh] = torch.rand(len(fresh), device=self.device) < self.cfg.rel_destination_envs
        self.destination_reached[fresh] = False
        fresh = fresh[self.is_destination_env[fresh]]
        if len(fresh):
            self._sample_destinations(fresh)
        # Make the very first observation contain the destination request too.
        self._update_command()

    def _sample_destinations(self, fresh: torch.Tensor) -> None:
        """Choose fixed forward world goals using the reset layout's endpoint convention."""
        position = self.robot.data.root_link_pos_w.torch[fresh]
        forward = torch.zeros_like(position)
        forward[:, 0] = 1.0
        rotation = math_utils.yaw_quat(self.robot.data.root_link_quat_w.torch[fresh])
        direction = math_utils.quat_apply(rotation, forward)
        layout = self._env.microduck_obstacle_layout[fresh]
        # Aim through the barrier/post, or along the passage. This only sets
        # a request; it never computes a collision-free route for the actor.
        ring = layout == OBSTACLE_LAYOUT_NAMES.index("ring")
        direction[ring] = (self._env.scene["obstacle"].data.root_pos_w.torch[fresh] - position)[ring]
        obstacle_positions = torch.stack(
            [self._env.scene[name].data.root_pos_w.torch[fresh] for name in OBSTACLE_CELLS], dim=1
        )
        front = (
            obstacle_positions[
                torch.arange(len(fresh), device=self.device), self._env.microduck_front_obstacle_index[fresh]
            ]
            - position
        )
        for name in ("barrier", "blocked", "corner"):
            mask = layout == OBSTACLE_LAYOUT_NAMES.index(name)
            direction[mask] = front[mask]
        # The two-obstacle course ends before the straight-ahead goal;
        # shuffled asset names must not redirect the request toward parked props.
        passage = layout == OBSTACLE_LAYOUT_NAMES.index("passage")
        midpoint = 0.5 * (
            self._env.scene["wall_left"].data.root_pos_w.torch[fresh]
            + self._env.scene["wall_right"].data.root_pos_w.torch[fresh]
        )
        direction[passage] = (midpoint - position)[passage]
        # Put the slalom endpoint beside the last obstacle, not inside its
        # footprint. This chooses free goal space, not intermediate waypoints.
        slalom = layout == OBSTACLE_LAYOUT_NAMES.index("slalom")
        local_obstacles = torch.stack(
            [
                math_utils.quat_apply_inverse(rotation, self._env.scene[name].data.root_pos_w.torch[fresh] - position)
                for name in ("crate", "tall_box", "cylinder")
            ],
            dim=1,
        )
        last = local_obstacles[:, :, 0].argmax(dim=1)
        last_y = local_obstacles[torch.arange(len(fresh), device=self.device), last, 1]
        local_goal = forward * self.cfg.destination_distance
        local_goal[:, 1] = torch.where(last_y >= 0.0, -0.5, 0.5)
        direction[slalom] = math_utils.quat_apply(rotation, local_goal)[slalom]
        direction[:, 2] = 0.0
        # Keep reset goals ahead; later turns do not move the world target.
        local_direction = math_utils.quat_apply_inverse(rotation, direction)
        minimum_forward = 1e-3 * direction.norm(dim=-1).clamp_min(1.0)
        adjust = local_direction[:, 0] < minimum_forward
        local_direction[:, 0] = torch.maximum(local_direction[:, 0].abs(), minimum_forward)
        direction[adjust] = math_utils.quat_apply(rotation, local_direction)[adjust]
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.destination_w[fresh] = position + self.cfg.destination_distance * direction
        self.destination_speed[fresh] = torch.empty(len(fresh), device=self.device).uniform_(
            self.cfg.forward_min_speed, self.cfg.ranges.lin_vel_x[1]
        )

    def _update_command(self):
        super()._update_command()
        destination_ids = self.is_destination_env.nonzero().flatten()
        if not len(destination_ids):
            return
        goal_offset_w = self.destination_w[destination_ids] - self.robot.data.root_link_pos_w.torch[destination_ids]
        goal_offset_w[:, 2] = 0.0
        goal_offset_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_link_quat_w.torch[destination_ids]), goal_offset_w
        )
        goal_distance = goal_offset_b[:, :2].norm(dim=-1)
        heading_error = torch.atan2(goal_offset_b[:, 1], goal_offset_b[:, 0])
        self.destination_reached[destination_ids] |= goal_distance <= self.cfg.destination_tolerance
        self.vel_command_b[destination_ids, 0] = (
            self.destination_speed[destination_ids]
            * heading_error.cos().clamp_min(0.0)
            * (goal_distance / 0.5).clamp_max(1.0)
        )
        self.vel_command_b[destination_ids, 1] = 0.0
        self.vel_command_b[destination_ids, 2] = (self.cfg.destination_heading_gain * heading_error).clamp(
            *self.cfg.ranges.ang_vel_z
        )
        self.vel_command_b[destination_ids[self.destination_reached[destination_ids]]] = 0.0

    def _update_metrics(self):
        valid = self._env.episode_length_buf > 0
        robot = self.robot
        velocity = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch
        )
        command = self.command
        self._error_xy_sum += torch.nan_to_num((command[:, :2] - velocity[:, :2]).norm(dim=-1)) * valid
        self._error_yaw_sum += (
            torch.nan_to_num((command[:, 2] - robot.data.root_link_ang_vel_w.torch[:, 2]).abs()) * valid
        )
        self._step_count += valid
        speed = command[:, :2].norm(dim=-1)
        moving = (speed >= 0.05) & valid
        path_clear = requested_path_clearance(self._env) >= 0.25
        clear = moving & path_clear
        progress = navigation_progress(self._env)
        values = {
            "clearance": obstacle_clearances(self._env).amin(dim=-1) * valid,
            "moving": moving.float(),
            "blocked": (moving & ~path_clear).float(),
            "clear": clear.float(),
            "progress": progress * clear,
            "stopped": (clear & (velocity[:, :2].norm(dim=-1) < 0.05)).float(),
            "distance": (velocity[:, :2] * command[:, :2]).sum(dim=-1)
            / speed.clamp_min(0.05)
            * self._env.step_dt
            * moving,
        }
        for name, value in values.items():
            self._navigation_sums[name] += value

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        count = self._step_count[ids].clamp_min(1.0)
        error_xy = self._error_xy_sum[ids] / count
        error_yaw = self._error_yaw_sum[ids] / count
        self.metrics["error_vel_xy"][ids] = error_xy
        self.metrics["error_vel_yaw"][ids] = error_yaw
        self.metrics["tracking_success_rate"][ids] = (
            (self._step_count[ids] > 0)
            & (error_xy < self.cfg.vel_xy_success_threshold)
            & (error_yaw < self.cfg.vel_yaw_success_threshold)
        ).float()
        termination = self._env.termination_manager
        for metric, term in (
            ("collision_rate", "obstacle_collision"),
            ("fall_rate", "fell_over"),
            ("field_boundary_rate", "field_boundary"),
        ):
            self.metrics[metric][ids] = termination.get_term(term)[ids].float()
        self.metrics["horizon_survival_rate"][ids] = (
            termination.get_term("time_out")[ids] & ~termination.terminated[ids]
        ).float()
        sums = self._navigation_sums
        self.metrics["mean_clearance_m"][ids] = sums["clearance"][ids] / count
        self.metrics["blocked_fraction"][ids] = sums["blocked"][ids] / sums["moving"][ids].clamp_min(1.0)
        self.metrics["clear_path_fraction"][ids] = sums["clear"][ids] / sums["moving"][ids].clamp_min(1.0)
        self.metrics["clear_path_progress_ratio"][ids] = sums["progress"][ids] / sums["clear"][ids].clamp_min(1.0)
        self.metrics["unnecessary_stop_fraction"][ids] = sums["stopped"][ids] / sums["clear"][ids].clamp_min(1.0)
        self.metrics["requested_direction_distance_m"][ids] = sums["distance"][ids]
        extras = CommandTerm.reset(self, env_ids)
        self._error_xy_sum[ids] = 0.0
        self._error_yaw_sum[ids] = 0.0
        self._step_count[ids] = 0.0
        for value in sums.values():
            value[ids] = 0.0
        return extras


@configclass
class MicroDuckNavigationCommandCfg(MicroDuckVelocityCommandCfg):
    """MicroDuck command mixture with separately named avoidance diagnostics."""

    class_type: type = MicroDuckNavigationCommand
    rel_destination_envs: float = 0.0
    """Fraction of episodes with persistent world destinations; zero preserves joystick-only requests."""
    destination_distance: float = 2.6
    """Destination distance from reset position [m], beyond the procedural forward obstacles."""
    destination_tolerance: float = 0.2
    """Arrival radius [m]; request a stop for the rest of the episode after arrival."""
    destination_heading_gain: float = 1.5
    """Heading-error gain [1/s], bounded by the existing yaw-rate limits."""
