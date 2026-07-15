# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the pour task.

In addition to the standard timeout, instability guards reset non-finite states and finite media
that leave the task workspace before they can expand the allocating sparse grid without bound.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


_GRIPPER_POSITION_TOLERANCE = 1.0e-6


def _state_finite(
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    tcp_body_q: torch.Tensor,
    cup_body_q: torch.Tensor,
    cup_lin_vel: torch.Tensor,
    cup_ang_vel: torch.Tensor,
    particle_pos: torch.Tensor,
) -> torch.Tensor:
    """Return a per-environment finite-state mask using unsanitized simulation tensors."""
    robot_ok = torch.isfinite(robot_joint_pos).all(dim=-1) & torch.isfinite(robot_joint_vel).all(dim=-1)
    tcp_ok = torch.isfinite(tcp_body_q).all(dim=-1)
    cup_ok = (
        torch.isfinite(cup_body_q).all(dim=-1)
        & torch.isfinite(cup_lin_vel).all(dim=-1)
        & torch.isfinite(cup_ang_vel).all(dim=-1)
    )
    media_ok = torch.isfinite(particle_pos).all(dim=(1, 2))
    return robot_ok & tcp_ok & cup_ok & media_ok


def _rigid_state_in_bounds(
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    joint_pos_limits: torch.Tensor,
    tcp_body_q: torch.Tensor,
    cup_body_q: torch.Tensor,
    cup_lin_vel: torch.Tensor,
    cup_ang_vel: torch.Tensor,
    env_origins: torch.Tensor,
    lower_bound: tuple[float, float, float] | torch.Tensor,
    upper_bound: tuple[float, float, float] | torch.Tensor,
    joint_position_margin: float,
    max_joint_velocity: float,
    max_cup_linear_velocity: float,
    max_cup_angular_velocity: float,
) -> torch.Tensor:
    """Return whether every rigid state that feeds actor observations is physically bounded."""
    lower = torch.as_tensor(lower_bound, device=robot_joint_pos.device, dtype=robot_joint_pos.dtype)
    upper = torch.as_tensor(upper_bound, device=robot_joint_pos.device, dtype=robot_joint_pos.dtype)
    joint_lower = joint_pos_limits[..., 0] - float(joint_position_margin)
    joint_upper = joint_pos_limits[..., 1] + float(joint_position_margin)
    joint_position_ok = ((robot_joint_pos >= joint_lower) & (robot_joint_pos <= joint_upper)).all(dim=-1)
    joint_velocity_ok = (torch.abs(robot_joint_vel) <= float(max_joint_velocity)).all(dim=-1)

    tcp_position = tcp_body_q[:, :3] - env_origins
    cup_position = cup_body_q[:, :3] - env_origins
    tcp_position_ok = ((tcp_position >= lower) & (tcp_position <= upper)).all(dim=-1)
    cup_position_ok = ((cup_position >= lower) & (cup_position <= upper)).all(dim=-1)
    tcp_quat_norm = torch.linalg.vector_norm(tcp_body_q[:, 3:7], dim=-1)
    cup_quat_norm = torch.linalg.vector_norm(cup_body_q[:, 3:7], dim=-1)
    pose_ok = (
        torch.isfinite(tcp_body_q).all(dim=-1)
        & torch.isfinite(cup_body_q).all(dim=-1)
        & (torch.abs(tcp_quat_norm - 1.0) <= 0.1)
        & (torch.abs(cup_quat_norm - 1.0) <= 0.1)
    )

    cup_linear_velocity_ok = torch.linalg.vector_norm(cup_lin_vel, dim=-1) <= float(max_cup_linear_velocity)
    cup_angular_velocity_ok = torch.linalg.vector_norm(cup_ang_vel, dim=-1) <= float(max_cup_angular_velocity)
    return (
        joint_position_ok
        & joint_velocity_ok
        & tcp_position_ok
        & cup_position_ok
        & pose_ok
        & cup_linear_velocity_ok
        & cup_angular_velocity_ok
    )


def _particles_in_workspace(
    particle_pos_e: torch.Tensor,
    lower_bound: tuple[float, float, float] | torch.Tensor,
    upper_bound: tuple[float, float, float] | torch.Tensor,
) -> torch.Tensor:
    """Return whether every particle lies inside its environment-local workspace."""
    lower = torch.as_tensor(lower_bound, device=particle_pos_e.device, dtype=particle_pos_e.dtype)
    upper = torch.as_tensor(upper_bound, device=particle_pos_e.device, dtype=particle_pos_e.dtype)
    return ((particle_pos_e >= lower) & (particle_pos_e <= upper)).all(dim=(1, 2))


def _spilled_particle_mask(
    particle_pos_e: torch.Tensor,
    in_source: torch.Tensor,
    in_target: torch.Tensor,
    max_height: float,
) -> torch.Tensor:
    """Classify particles outside both cups that have reached the table [m]."""
    outside_cups = ~in_source & ~in_target
    return outside_cups & (particle_pos_e[..., 2] <= float(max_height))


def _delivered_particle_mask(in_source: torch.Tensor, in_target: torch.Tensor) -> torch.Tensor:
    """Classify particles inside the receiver only after they have left the source cup."""
    return in_target & ~in_source


def nonfinite_failure(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate on non-finite simulation state (instability guard)."""
    return ~env.state_finite()


def extreme_rigid_state(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate finite rigid states before extreme values reach observation normalization."""
    return ~env.rigid_state_in_bounds()


def particle_out_of_bounds(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate an environment when any media particle escapes its workspace."""
    return ~env.particles_in_workspace()


def excessive_spill(env: FrankaPourEnv, terminate: bool = True) -> torch.Tensor:
    """Track when strictly more than the allowed media fraction is spilled."""
    if not isinstance(terminate, bool):
        raise TypeError("terminate must be a bool.")
    spilled = env.spilled_fraction() > float(env.cfg.max_spill_fraction)
    return spilled if terminate else torch.zeros_like(spilled)


def unsuccessful_time_out(env: FrankaPourEnv) -> torch.Tensor:
    """Truncate at the finite-horizon deadline unless success fired on the same step."""
    deadline = env.episode_length_buf >= env.max_episode_length
    return deadline & ~env.episode_succeeded


def lost_lifted_grasp(
    env: FrankaPourEnv,
    dwell_time_s: float = 0.05,
    max_tcp_distance: float = 0.015,
    max_gripper_width_error: float = 0.012,
    max_gripper_command: float = 0.025,
    terminate: bool = True,
) -> torch.Tensor:
    """Track when a demonstrated lift loses the physical cup grasp continuously.

    A short dwell rejects isolated contact-deflection flicker from the coupled rigid solver while
    still allowing the reverse curriculum to terminate a genuinely dropped cup promptly. Static
    reset-dataset training can disable termination while retaining the state monitor for metrics.
    """
    if not math.isfinite(dwell_time_s) or dwell_time_s <= 0.0:
        raise ValueError(f"dwell_time_s must be finite and positive, got {dwell_time_s}.")
    if not isinstance(terminate, bool):
        raise TypeError("terminate must be a bool.")
    _, preloaded_grasp, lifted_grasp = source_grasp_milestones(
        env,
        min_lift_height=max(float(env.cfg.success_min_lift_height), 1.0e-6),
        max_tcp_distance=max_tcp_distance,
        max_gripper_width_error=max_gripper_width_error,
        max_gripper_command=max_gripper_command,
    )
    env._lifted_grasp_seen |= lifted_grasp
    lost = env._lifted_grasp_seen & ~preloaded_grasp
    dwell_steps = max(1, math.ceil(float(dwell_time_s) / max(float(env.step_dt), 1.0e-6)))
    env._lost_grasp_dwell_count[:] = torch.where(
        lost,
        torch.clamp(env._lost_grasp_dwell_count + 1, max=dwell_steps),
        0,
    )
    dwell_qualified_loss = lost & (env._lost_grasp_dwell_count >= dwell_steps)
    return dwell_qualified_loss if terminate else torch.zeros_like(dwell_qualified_loss)


def source_grasp_milestones(
    env: FrankaPourEnv,
    min_lift_height: float,
    max_tcp_distance: float,
    max_gripper_width_error: float,
    max_gripper_command: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reached, actively preloaded, and lifted source-grasp masks."""
    cup_lift = env.cup_pose_e()[:, 2] - float(env.cup_reset_height)
    tcp_distance = torch.linalg.vector_norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    gripper_width_error = torch.abs(env.gripper_width() - float(env.gripper_grasp_width))
    gripper = env.action_manager.get_term("gripper_action")
    gripper_command = gripper.commanded_position[:, 0]
    bilateral_contact = getattr(
        gripper,
        "bilateral_contact",
        torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
    )
    reached_grasp = tcp_distance <= float(max_tcp_distance)
    preloaded_grasp = (
        reached_grasp
        & (gripper_width_error <= float(max_gripper_width_error))
        & (gripper_command <= float(max_gripper_command) + _GRIPPER_POSITION_TOLERANCE)
        & bilateral_contact
    )
    lifted_grasp = preloaded_grasp & (cup_lift >= float(min_lift_height))
    return reached_grasp, preloaded_grasp, lifted_grasp


def stable_pour_success(
    env: FrankaPourEnv,
    dwell_time_s: float = 0.15,
    min_lift_height: float = 0.05,
    max_tcp_distance: float = 0.015,
    max_gripper_width_error: float = 0.012,
    max_gripper_command: float = 0.025,
) -> torch.Tensor:
    """Terminate after a delivered pour remains held, preloaded, and lifted for a dwell interval.

    This is deliberately a plain manager function rather than a class term. Isaac Lab's generic
    record/replay tools remove the success termination from the manager and invoke its configured
    function directly; keeping the per-world counter on the task preserves that standard workflow.
    The success term must follow every failure predicate so ``terminated`` contains them, and must
    precede :func:`unsuccessful_time_out` so a valid deadline transfer is classified only as
    success.
    """
    if not math.isfinite(dwell_time_s) or dwell_time_s <= 0.0:
        raise ValueError(f"dwell_time_s must be finite and positive, got {dwell_time_s}.")
    for name, value in (
        ("min_lift_height", min_lift_height),
        ("max_tcp_distance", max_tcp_distance),
        ("max_gripper_width_error", max_gripper_width_error),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")
    if not math.isfinite(max_gripper_command) or max_gripper_command < 0.0:
        raise ValueError(f"max_gripper_command must be finite and nonnegative, got {max_gripper_command}.")
    dwell_steps = max(1, math.ceil(float(dwell_time_s) / max(float(env.step_dt), 1.0e-6)))
    target_fraction = env.count_in_target() / max(env.num_particles, 1)
    env.ep_max_target_frac[:] = torch.maximum(env.ep_max_target_frac, target_fraction)
    _, _, held_pour = source_grasp_milestones(
        env,
        min_lift_height=min_lift_height,
        max_tcp_distance=max_tcp_distance,
        max_gripper_width_error=max_gripper_width_error,
        max_gripper_command=max_gripper_command,
    )
    env.update_held_delivery_tracker(held_pour)
    held_target_fraction = env.current_held_delivered_mask().sum(dim=1).float() / max(env.num_particles, 1)
    within_spill_limit = env.spilled_fraction() <= float(env.cfg.max_spill_fraction)
    candidate = (
        (held_target_fraction >= env.pour_target_frac)
        & held_pour
        & within_spill_limit
        & ~env.termination_manager.terminated
    )
    env._success_dwell_count[:] = torch.where(
        candidate,
        torch.clamp(env._success_dwell_count + 1, max=dwell_steps),
        0,
    )
    success = candidate & (env._success_dwell_count >= dwell_steps)
    env.episode_succeeded |= success
    return success


def immediate_pour_success(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate immediately when the current target-bowl fraction reaches its threshold.

    Reset-cache particles always start in the source cup, so current target occupancy directly
    measures task progress without also requiring a retained grasp, lift milestone, delivery
    history, or dwell interval. Failure terms run before success and therefore retain precedence.
    """
    target_fraction = env.count_in_target() / max(env.num_particles, 1)
    env.ep_max_target_frac[:] = torch.maximum(env.ep_max_target_frac, target_fraction)
    success = (target_fraction >= env.pour_target_frac) & ~env.termination_manager.terminated
    env._success_dwell_count[:] = success.to(dtype=env._success_dwell_count.dtype)
    env.episode_succeeded |= success
    return success


def nonterminating_stable_pour_success(
    env: FrankaPourEnv,
    dwell_time_s: float = 0.15,
    min_lift_height: float = 0.05,
    max_tcp_distance: float = 0.015,
    max_gripper_width_error: float = 0.012,
    max_gripper_command: float = 0.025,
) -> torch.Tensor:
    """Track stable success without ending training episodes, while remaining replay-compatible."""
    success = stable_pour_success(
        env,
        dwell_time_s=dwell_time_s,
        min_lift_height=min_lift_height,
        max_tcp_distance=max_tcp_distance,
        max_gripper_width_error=max_gripper_width_error,
        max_gripper_command=max_gripper_command,
    )
    # Standard record/replay tools temporarily remove the managed success term and invoke this
    # configured function directly. Return the real predicate in that context; suppress only the
    # live TerminationManager's done signal used by fixed-horizon reset-dataset training.
    if "success" not in env.termination_manager.active_terms:
        return success
    return torch.zeros_like(success)
