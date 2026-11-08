# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck obstacle-navigation rewards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import math as math_utils

from .commands import navigation_progress
from .geometry import obstacle_clearances, obstacle_geometry, obstacle_point_clearances, requested_path_clearance
from .observations import applied_velocity_commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class NavigationGoalArrival(ManagerTermBase):
    """Pay once on safe destination entry and revoke that credit on later failure."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._safe = torch.ones_like(self._paid)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._paid[slice(None) if env_ids is None else env_ids] = False
        self._safe[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        command = env.command_manager.get_term("base_velocity")
        position = env.scene["robot"].data.root_link_pos_w.torch
        # Commands update after rewards, so measure entry rather than reading the arrival latch.
        reached = (position[:, :2] - command.destination_w[:, :2]).norm(dim=-1) <= command.cfg.destination_tolerance
        safe = ~env.termination_manager.terminated & ~env.termination_manager.get_term("field_boundary")
        safe &= torch.isfinite(position).all(dim=-1)
        revoke = self._paid & ~safe
        self._safe &= safe
        event = command.is_destination_env & reached & self._safe & ~self._paid
        self._paid |= event
        self._paid &= ~revoke
        return (event.float() - revoke.float()) / env.step_dt


class NavigationObstacleBypass(ManagerTermBase):
    """Pay once for safely passing the nearest obstacle blocking the initial request.

    Eligibility intersects the actual configured footprint; the swept witness uses
    its enclosing rectangle expanded by the robot radius. The witness is conservative
    for circular footprints and rounded corners. Target and entry face stay fixed
    until reset, even when commands change.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._centers, self._rotations, self._half_sizes, self._radii = obstacle_geometry(env)
        self._previous = torch.zeros(env.num_envs, 3, device=env.device)
        self._initial_heading = torch.zeros(env.num_envs, 4, device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._eligible = torch.zeros_like(self._pending)
        self._paid = torch.zeros_like(self._pending)
        self._left = torch.ones_like(self._pending)
        self._right = torch.ones_like(self._pending)
        self._target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._axis = torch.zeros_like(self._target)
        self._sign = torch.ones(env.num_envs, device=env.device)
        self._rows = torch.arange(env.num_envs, device=env.device)

    @property
    def eligible(self) -> torch.Tensor:
        """Episodes with an obstacle blocking the initial requested path (read-only)."""
        return self._eligible

    @property
    def bypassed(self) -> torch.Tensor:
        """Episodes that completed the initial obstacle bypass witness (read-only)."""
        return self._paid

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        centers, rotations, _, _ = obstacle_geometry(self._env)
        self._centers[ids], self._rotations[ids] = centers[ids], rotations[ids]
        robot = self._env.scene["robot"]
        self._previous[ids] = robot.data.root_link_pos_w.torch[ids]
        self._initial_heading[ids] = math_utils.yaw_quat(robot.data.root_link_quat_w.torch[ids])
        self._pending[ids] = True
        self._eligible[ids] = False
        self._paid[ids] = False
        self._left[ids] = True
        self._right[ids] = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # Locate the initial request ray in every obstacle's local frame: (env, obstacle, XY).
        position = env.scene["robot"].data.root_link_pos_w.torch
        previous_local = math_utils.quat_apply_inverse(self._rotations, self._previous[:, None] - self._centers)[
            ..., :2
        ]
        current_local = math_utils.quat_apply_inverse(self._rotations, position[:, None] - self._centers)[..., :2]
        expanded_half_sizes = self._half_sizes + 0.12
        request = env.command_manager.get_command("base_velocity").clone()
        self._select_initial_obstacle(previous_local, request, expanded_half_sizes)

        # A bypass must stay on one lateral side while crossing the obstacle's full length.
        previous_target = previous_local[self._rows, self._target]
        current_target = current_local[self._rows, self._target]
        longitudinal_start = previous_target[self._rows, self._axis] * self._sign
        longitudinal_end = current_target[self._rows, self._axis] * self._sign
        lateral_start = previous_target[self._rows, 1 - self._axis]
        lateral_end = current_target[self._rows, 1 - self._axis]
        half_length = expanded_half_sizes[self._target, self._axis]
        half_width = expanded_half_sizes[self._target, 1 - self._axis]
        longitudinal_motion = longitudinal_end - longitudinal_start
        no_longitudinal_motion = longitudinal_motion.abs() < 1.0e-8
        safe_motion = torch.where(no_longitudinal_motion, 1.0, longitudinal_motion)
        entry_fraction, exit_fraction = (
            (-half_length - longitudinal_start) / safe_motion,
            (half_length - longitudinal_start) / safe_motion,
        )
        segment_entry = torch.where(
            no_longitudinal_motion, 0.0, torch.minimum(entry_fraction, exit_fraction).clamp_min(0.0)
        )
        segment_exit = torch.where(
            no_longitudinal_motion, 1.0, torch.maximum(entry_fraction, exit_fraction).clamp_max(1.0)
        )
        crosses_obstacle_span = torch.where(
            no_longitudinal_motion, longitudinal_start.abs() <= half_length, segment_entry <= segment_exit
        )
        lateral_at_entry, lateral_at_exit = (
            lateral_start + segment_entry * (lateral_end - lateral_start),
            lateral_start + segment_exit * (lateral_end - lateral_start),
        )
        self._left &= ~crosses_obstacle_span | ((lateral_at_entry >= half_width) & (lateral_at_exit >= half_width))
        self._right &= ~crosses_obstacle_span | ((lateral_at_entry <= -half_width) & (lateral_at_exit <= -half_width))
        safe = ~env.termination_manager.terminated & ~env.termination_manager.get_term("field_boundary")
        safe &= torch.isfinite(position).all(dim=-1) & (obstacle_clearances(env).amin(dim=-1) >= 0.0)
        self._eligible &= safe
        event = self._eligible & (self._left | self._right) & (longitudinal_end >= half_length) & ~self._paid
        self._paid |= event
        self._previous.copy_(position)
        return event.float() / env.step_dt

    def _select_initial_obstacle(
        self, previous_local: torch.Tensor, request: torch.Tensor, expanded_half_sizes: torch.Tensor
    ) -> None:
        """Latch the nearest footprint and entry face on the first request after reset."""
        # Command reset follows reward reset. Arm with the first fresh request.
        request[:, 2] = 0.0
        request_direction_w = math_utils.quat_apply(self._initial_heading, request)
        request_direction_local = math_utils.quat_apply_inverse(
            self._rotations, request_direction_w[:, None].expand_as(self._centers)
        )[..., :2]
        # Intersect the ray with box faces. Parallel axes accept only origins inside their bounds.
        parallel_to_face = request_direction_local.abs() < 1.0e-8
        safe_direction = torch.where(parallel_to_face, 1.0, request_direction_local)
        face_time_min = (-self._half_sizes - previous_local) / safe_direction
        face_time_max = (self._half_sizes - previous_local) / safe_direction
        inside_axis_bounds = previous_local.abs() <= self._half_sizes
        parallel_entry = torch.where(inside_axis_bounds, -torch.inf, torch.inf)
        parallel_exit = torch.where(inside_axis_bounds, torch.inf, -torch.inf)
        axis_entry_time = torch.where(parallel_to_face, parallel_entry, torch.minimum(face_time_min, face_time_max))
        axis_exit_time = torch.where(parallel_to_face, parallel_exit, torch.maximum(face_time_min, face_time_max))
        box_entry_time, entry_axis = axis_entry_time.max(dim=-1)
        box_exit_time = axis_exit_time.amin(dim=-1)
        # A bounding-box corner is not a blocked ray for a circle.
        request_speed_squared = request_direction_local.square().sum(dim=-1).clamp_min(1.0e-12)
        ray_projection = (previous_local * request_direction_local).sum(dim=-1)
        discriminant = ray_projection.square() - request_speed_squared * (
            previous_local.square().sum(dim=-1) - self._radii.square()
        )
        circle_entry_time = (-ray_projection - discriminant.clamp_min(0.0).sqrt()) / request_speed_squared
        is_circle = self._radii > 0.0
        entry_time = torch.where(is_circle, circle_entry_time, box_entry_time)
        intersects_ahead = torch.where(is_circle, discriminant >= 0.0, box_exit_time >= box_entry_time) & (
            entry_time >= 0.0
        )
        intersects_ahead &= request[:, :2].norm(dim=-1)[:, None] >= 0.05
        entry_sign = request_direction_local.gather(-1, entry_axis[..., None]).squeeze(-1).sign()
        start_along_entry_axis = previous_local.gather(-1, entry_axis[..., None]).squeeze(-1) * entry_sign
        entry_half_length = expanded_half_sizes.expand_as(previous_local).gather(-1, entry_axis[..., None]).squeeze(-1)
        intersects_ahead &= start_along_entry_axis < -entry_half_length
        # Freeze the first blocking obstacle and its entry face for this episode.
        nearest_obstacle = torch.where(intersects_ahead, entry_time, torch.inf).argmin(dim=-1)
        self._target = torch.where(self._pending, nearest_obstacle, self._target)
        self._axis = torch.where(self._pending, entry_axis[self._rows, nearest_obstacle], self._axis)
        self._sign = torch.where(self._pending, entry_sign[self._rows, nearest_obstacle], self._sign)
        self._eligible = torch.where(self._pending, intersects_ahead[self._rows, nearest_obstacle], self._eligible)
        self._pending.zero_()


def navigation_gated_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward positive progress when measured and executed motion avoid closing risk."""
    raw = navigation_progress(env)
    risk = obstacle_approach_risk(
        env,
        safety_distance=0.2,
        time_horizon=2.0,
        prediction_dt=0.1,
        robot_radius=0.12,
        commanded_risk_weight=0.0,
    )
    return torch.where(raw > 0, raw * (1.0 - risk), raw)


def navigation_velocity_tracking(
    env: ManagerBasedRLEnv, std: float = 0.15, clear_distance: float = 0.25
) -> torch.Tensor:
    """Track velocity during requested-direction progress; preserve explicit stop requests."""
    command = env.command_manager.get_command("base_velocity")[:, :2]
    robot = env.scene["robot"]
    velocity = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch
    )
    tracking = torch.exp(-(command - velocity[:, :2]).square().sum(dim=-1) / std**2)
    gate = (requested_path_clearance(env) / clear_distance).clamp(0.0, 1.0)
    # During a movement request, matching velocity/heading while stopped must not pay.
    gate = torch.where(command.norm(dim=-1) < 0.05, 1.0, gate * navigation_progress(env).clamp(0.0, 1.0))
    return torch.nan_to_num(tracking * gate)


def navigation_yaw_tracking(env: ManagerBasedRLEnv, std: float = 0.35, clear_distance: float = 0.25) -> torch.Tensor:
    """Track yaw during requested-direction progress; preserve explicit in-place turns."""
    command = env.command_manager.get_command("base_velocity")
    actual = env.scene["robot"].data.root_link_ang_vel_w.torch[:, 2]
    tracking = torch.exp(-(command[:, 2] - actual).square() / std**2)
    gate = (requested_path_clearance(env) / clear_distance).clamp(0.0, 1.0)
    gate = torch.where(command[:, :2].norm(dim=-1) < 0.05, 1.0, gate * navigation_progress(env).clamp(0.0, 1.0))
    return torch.nan_to_num(tracking * gate)


def obstacle_clearance_risk(env: ManagerBasedRLEnv, safety_distance: float = 0.2) -> torch.Tensor:
    """Bounded [0, 1] cost for approaching the planar robot safety footprint."""
    clearance = obstacle_clearances(env).amin(dim=-1)
    return ((safety_distance - clearance) / safety_distance).clamp(0.0, 1.0).square()


def obstacle_approach_risk(
    env: ManagerBasedRLEnv,
    safety_distance: float = 0.1,
    time_horizon: float = 2.0,
    prediction_dt: float = 0.1,
    robot_radius: float = 0.12,
    commanded_risk_weight: float = 0.0,
) -> torch.Tensor:
    """Cost in [0, 1 + commanded_risk_weight] for closing within the braking margin.

    Use the larger closing speed from measured motion and the processed planar
    command delivered to the walker. An optional independent command cost retains
    braking credit when measured motion dominates, increasing the total penalty.
    Stopping, retreating, and parallel motion have no cost when neither measured
    nor commanded motion closes on an obstacle.
    Ground is excluded: this uses only the task's obstacle geometry, not ToF.
    """
    if safety_distance < 0.0 or time_horizon <= 0.0 or prediction_dt <= 0.0:
        raise ValueError("Approach risk needs a nonnegative margin and positive time intervals.")
    if not math.isfinite(commanded_risk_weight) or commanded_risk_weight < 0.0:
        raise ValueError("Commanded risk weight must be finite and nonnegative.")
    robot = env.scene["robot"]
    position = robot.data.root_link_pos_w.torch
    predicted = position + robot.data.root_link_lin_vel_w.torch * prediction_dt
    commanded_velocity = torch.zeros_like(position)
    commanded_velocity[:, :2] = applied_velocity_commands(env)[:, :2]
    commanded_velocity = math_utils.quat_apply(
        math_utils.yaw_quat(robot.data.root_link_quat_w.torch), commanded_velocity
    )
    commanded_prediction = position + commanded_velocity * prediction_dt
    clearances = obstacle_point_clearances(
        env, torch.stack((position, predicted, commanded_prediction), dim=1), robot_radius
    )
    current_clearance = clearances[:, 0]
    predicted_clearances = clearances[:, 1:]
    closing_speeds = (current_clearance[:, None] - predicted_clearances) / prediction_dt
    closing_speed = closing_speeds.clamp_min(0.0).amax(dim=1)
    remaining_distance = (current_clearance - safety_distance).clamp_min(0.0)
    # 0.20 m/s is a shaping reference near the nominal command, not a calibrated stopping speed.
    risk = ((closing_speed - remaining_distance / time_horizon) / 0.20).clamp(0.0, 1.0).square()
    risk = torch.where(closing_speed > 1e-4, risk, 0.0).amax(dim=-1)
    if commanded_risk_weight == 0.0:
        return risk
    commanded_clearance = clearances[:, 2]
    commanded_closing_speed = ((current_clearance - commanded_clearance) / prediction_dt).clamp_min(0.0)
    commanded_risk = ((commanded_closing_speed - remaining_distance / time_horizon) / 0.20).clamp(0.0, 1.0).square()
    commanded_risk = torch.where(commanded_closing_speed > 1e-4, commanded_risk, 0.0).amax(dim=-1)
    return risk + commanded_risk_weight * commanded_risk


def navigation_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mean squared change of executed commands, normalized by walking limits."""
    action = env.action_manager.get_term("joint_pos")
    limits = torch.tensor(action.cfg.velocity_limits, device=env.device)
    return ((action.velocity_commands - action.previous_velocity_commands) / limits).square().mean(dim=-1)
