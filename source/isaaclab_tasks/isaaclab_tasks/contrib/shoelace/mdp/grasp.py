# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared grasp quality, filtering, and bilateral scoring for manager terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .observations import finger_tail_signed_distance, tail_tcp_relative_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def shoelace_grasp_quality(
    env: ManagerBasedRLEnv,
    contact_std: float,
    relative_speed_std: float,
    open_position: float,
    closed_position: float,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    contact_penetration_tolerance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shared unfiltered grasp quality for acquisition, retention, and success.

    Args:
        env: Shoelace environment with current finger-tail contact observations.
        contact_std: Finger-tail surface-distance width [m].
        relative_speed_std: Tail-to-TCP relative-speed width [m/s].
        open_position: Open finger joint position [m].
        closed_position: Closed finger joint position [m].
        cable_cfgs: Left and right cable scene entities.
        robot_cfgs: Left and right robot entities with resolved finger joint and hand body indices.
        contact_penetration_tolerance: Allowed contact-solver penetration [m].

    Returns:
        Raw grasp qualities in robot-arm order, shape [N, 2], and finite-input flags, shape [N].
        Quality combines two-finger contact, actual closure, and low relative slip, not measured force closure.

    Raises:
        ValueError: If contact penetration tolerance is not finite and nonnegative.
    """
    if not math.isfinite(contact_penetration_tolerance) or contact_penetration_tolerance < 0.0:
        raise ValueError("contact_penetration_tolerance must be finite and nonnegative")
    signed_distance = finger_tail_signed_distance(env).reshape(env.num_envs, 2, 2)
    relative_speed = tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs)
    positions = torch.stack(
        [env.scene[cfg.name].data.joint_pos.torch[:, cfg.joint_ids[0]] for cfg in robot_cfgs], dim=1
    )
    closure = ((open_position - positions) / max(open_position - closed_position, 1.0e-6)).clamp(0.0, 1.0)
    finite = (
        torch.isfinite(signed_distance).all(dim=(1, 2))
        & torch.isfinite(relative_speed).all(dim=1)
        & torch.isfinite(closure).all(dim=1)
    )
    # Tolerate bounded solver penetration under load, without rewarding gaps or deeper penetration.
    contact_error = signed_distance.clamp_min(0.0) + (-signed_distance - contact_penetration_tolerance).clamp_min(0.0)
    finger_contact = torch.exp(-torch.square(contact_error / max(contact_std, 1.0e-6)))
    contact = hamacher_product(finger_contact[:, :, 0], finger_contact[:, :, 1])
    grasp = hamacher_product(contact, closure)
    motion_match = 1.0 - torch.tanh(relative_speed / max(relative_speed_std, 1.0e-6))
    return hamacher_product(grasp, motion_match), finite


def filter_grasps(
    previous: torch.Tensor, grasp: torch.Tensor, finite: torch.Tensor, step_dt: float, time_constant: float
) -> torch.Tensor:
    """Update the filter in place, seeding finite samples and preserving invalid environments.

    Args:
        previous: Filter state to update in place, shape [N, 2]; NaN marks unseeded entries.
        grasp: Current grasp qualities, shape [N, 2].
        finite: Valid-input flags, shape [N].
        step_dt: Policy step duration [s].
        time_constant: Positive filter time constant [s].

    Returns:
        The updated ``previous`` tensor, shape [N, 2].
    """
    alpha = 1.0 - math.exp(-step_dt / time_constant)
    filtered = torch.where(torch.isfinite(previous), previous + alpha * (grasp - previous), grasp)
    previous.copy_(torch.where(finite.unsqueeze(1), filtered, previous))
    return previous


def bilateral_score(per_arm_score: torch.Tensor, bilateral_fraction: float) -> torch.Tensor:
    """Combine independent credit with a symmetric cooperation bonus.

    Args:
        per_arm_score: Left and right scores in [0, 1], shape [N, 2].
        bilateral_fraction: Cooperation share in [0, 1]; the remainder uses the per-arm mean.

    Returns:
        Combined scores, shape [N].
    """
    bilateral = hamacher_product(per_arm_score[:, 0], per_arm_score[:, 1])
    return (1.0 - bilateral_fraction) * per_arm_score.mean(dim=1) + bilateral_fraction * bilateral


def hamacher_product(a: torch.Tensor, b: torch.Tensor | float, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the Hamacher soft-AND of two values in ``[0, 1]``.

    Args:
        a: First input tensor with values in [0, 1].
        b: Second input in [0, 1], broadcastable to ``a``.
        eps: Positive denominator regularization.

    Returns:
        Soft conjunction with the broadcast shape of the inputs.
    """
    return (a * b) / (a + b - a * b + eps)
