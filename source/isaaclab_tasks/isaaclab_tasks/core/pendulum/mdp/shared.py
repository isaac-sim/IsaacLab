# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure tensor calculations shared by the Pendulum MARL workflows."""

from __future__ import annotations

import math

import torch


@torch.jit.script
def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle [rad] to the range ``[-pi, pi]``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def links_upright(pole_pos: torch.Tensor, pendulum_pos: torch.Tensor, max_angle: float) -> torch.Tensor:
    """Return whether both physical links are within the upright angle limit.

    Args:
        pole_pos: Upper-link joint angle [rad], shape ``(num_envs,)``.
        pendulum_pos: Lower-link angle relative to the upper link [rad], shape ``(num_envs,)``.
        max_angle: Maximum permitted absolute world-relative link angle [rad].

    Returns:
        Whether both links are upright, shape ``(num_envs,)``.
    """
    upper_upright = normalize_angle(pole_pos).abs() <= max_angle
    lower_upright = normalize_angle(pole_pos + pendulum_pos).abs() <= max_angle
    return upper_upright & lower_upright


def update_upright_steps(upright_steps: torch.Tensor, upright: torch.Tensor) -> torch.Tensor:
    """Update per-environment consecutive upright control-step counts."""
    return torch.where(upright, upright_steps + 1, torch.zeros_like(upright_steps))


def compute_success(
    time_out: torch.Tensor, terminated: torch.Tensor, upright_steps: torch.Tensor, required_steps: int
) -> torch.Tensor:
    """Return successful episodes from timeout, failure, and upright-window state."""
    return time_out & ~terminated & (upright_steps >= required_steps)


def compute_cart_observation(
    cart_pos: torch.Tensor, cart_vel: torch.Tensor, pole_pos: torch.Tensor, pole_vel: torch.Tensor
) -> torch.Tensor:
    """Return the cart observation in final direct-task order."""
    return torch.cat((cart_pos, cart_vel, pole_pos, pole_vel), dim=-1)


def compute_pendulum_observation(
    pole_pos: torch.Tensor, pendulum_pos: torch.Tensor, pendulum_vel: torch.Tensor
) -> torch.Tensor:
    """Return the lower-pendulum observation in final direct-task order."""
    return torch.cat((pole_pos + pendulum_pos, pendulum_pos, pendulum_vel), dim=-1)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    rew_scale_upright: float,
    rew_scale_action: float,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    upright: torch.Tensor,
    cart_action: torch.Tensor,
    pendulum_action: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    """Compute the final direct Pendulum reward for one control step."""
    lower_angle = normalize_angle(pole_pos + pendulum_pos)
    lower_velocity = pole_vel + pendulum_vel
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.cos(pole_pos)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.cos(lower_angle)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.abs(lower_velocity)
    rew_upright = rew_scale_upright * upright.float()
    rew_action = rew_scale_action * (
        torch.sum(torch.square(cart_action), dim=1) + torch.sum(torch.square(pendulum_action), dim=1)
    )
    return (
        rew_alive
        + rew_termination
        + rew_pole_pos
        + rew_pendulum_pos
        + rew_cart_vel
        + rew_pole_vel
        + rew_pendulum_vel
        + rew_upright
        + rew_action
    ) * step_dt
