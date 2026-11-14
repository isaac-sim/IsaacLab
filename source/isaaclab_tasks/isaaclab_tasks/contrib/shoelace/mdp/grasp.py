# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared grasp quality, filtering, and bilateral scoring for manager terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..shoelace_constants import CONTACT_DISTANCE_CAP
from .observations import tail_tcp_relative_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def shoelace_grasp_quality(
    env: ManagerBasedRLEnv,
    contact_std: float,
    relative_speed_std: float,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    contact_penetration_tolerance: float = 0.0,
    *,
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score opposing inward contacts and low tail speed relative to the fixed TCP.

    Contact distances come from Newton collision candidates over the configured finger shapes
    and free-tail segments. The sensor accepts inward contacts between the finger centres;
    both fingers must touch the same or adjacent segments. No pad landmarks or aperture limits
    are imposed. This is the shared score for acquisition, retention, and success, not a
    force-closure or loaded-pull measurement.

    Args:
        env: Shoelace environment with current finger-tail contact observations.
        contact_std: Finger-tail surface-distance width [m].
        relative_speed_std: Fixed-TCP relative-speed width [m/s].
        cable_cfgs: Left and right cable scene entities.
        contact_penetration_tolerance: Allowed contact-solver penetration [m].
        robot_cfgs: Fixed TCP bodies in left/right arm order, using ``env.cfg.tcp_body_offsets``.

    Returns:
        Raw grasp qualities in robot-arm order, shape [N, 2], and finite-input flags, shape [N].
        Missing contact candidates score zero; gaps, deep penetration, and relative motion reduce quality.

    Raises:
        ValueError: If widths are not finite and positive or penetration tolerance is invalid.
    """
    if not all(math.isfinite(value) and value > 0 for value in (contact_std, relative_speed_std)):
        raise ValueError("Contact and relative-speed widths must be finite and positive")
    if not math.isfinite(contact_penetration_tolerance) or contact_penetration_tolerance < 0:
        raise ValueError("Contact penetration tolerance must be finite and nonnegative")
    signed = env.scene["finger_tail_contacts"].pinch_distance.reshape(env.num_envs, 2, 2, 3)
    error = signed.clamp_min(0.0) + (-signed - contact_penetration_tolerance).clamp_min(0.0)
    observed = signed < CONTACT_DISTANCE_CAP - 1.0e-7
    finger_contact = torch.exp(-(error / contact_std).square()) * observed
    pairs = hamacher_product(finger_contact[:, :, 0, :, None], finger_contact[:, :, 1, None, :])
    # Neighbouring segments can share a physical pinch; separated tail segments cannot.
    pairs[:, :, 0, 2] = pairs[:, :, 2, 0] = 0.0
    contact = pairs.flatten(2).amax(dim=2)
    relative_speed = tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs)
    motion_match = 1.0 - torch.tanh(relative_speed / relative_speed_std)
    finite = torch.isfinite(signed).all(dim=(1, 2, 3)) & torch.isfinite(relative_speed).all(dim=1)
    return hamacher_product(contact, motion_match), finite


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
