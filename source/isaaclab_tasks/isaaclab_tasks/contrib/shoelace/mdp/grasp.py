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
from .observations import finger_point_state_w, finger_tail_signed_distance
from .utils import tail_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def grasp_geometry(
    env: ManagerBasedRLEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    finger_cfgs: tuple[SceneEntityCfg, ...],
    target_diameter: float,
) -> dict[str, torch.Tensor]:
    """Measure a two-surface grasp without assuming a gripper joint type or fixed TCP.

    Capture distance measures the tail centre's distance to the segment joining the surface points.
    Relative speed uses each point separately against the mean free-tail velocity; it is an
    approximation at reference points, not a contact-resolved slip or force-closure measurement.
    ``target_diameter`` is the cable diameter [m] subtracted from reference-point distances.
    Scalar fields have shape [N, 2]; pad distance has shape [N, 2, 2]. Units are metres and m/s.
    """
    if not math.isfinite(target_diameter) or target_diameter <= 0:
        raise ValueError("Target diameter must be finite and positive")
    points, velocities = finger_point_state_w(env, finger_cfgs)
    tails, tail_velocities = tail_state(env, cable_cfgs)
    span = points[:, :, 1] - points[:, :, 0]
    aperture = span.norm(dim=-1)
    axis = span / aperture.unsqueeze(-1).clamp_min(1.0e-6)
    delta = tails - points.mean(dim=2)
    axial = (delta * axis).sum(dim=-1, keepdim=True)
    outside = (axial.abs() - 0.5 * aperture.unsqueeze(-1)).clamp_min(0.0)
    distance = torch.cat((delta - axial * axis, outside), dim=-1).norm(dim=-1)
    slip = (tail_velocities.unsqueeze(2) - velocities).norm(dim=-1).amax(dim=2)
    pad_distance = ((tails.unsqueeze(2) - points).norm(dim=-1) - target_diameter / 2).clamp_min(0.0)
    finite = (aperture > 1.0e-6).all(dim=1)
    for value in (points, velocities, tails, tail_velocities):
        finite &= torch.isfinite(value.flatten(1)).all(dim=1)
    return {
        "aperture": aperture,
        "capture_distance": distance,
        "surface_midpoint_distance": delta.norm(dim=-1),
        "relative_speed": slip,
        "pad_distance": pad_distance,
        "finite": finite,
    }


def _finger_contact_quality(
    env: ManagerBasedRLEnv, contact_std: float, contact_penetration_tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score available contact candidates, tolerating bounded penetration but excluding the no-candidate sentinel."""
    if not math.isfinite(contact_std) or contact_std <= 0:
        raise ValueError("Contact width must be finite and positive")
    if not math.isfinite(contact_penetration_tolerance) or contact_penetration_tolerance < 0:
        raise ValueError("Contact penetration tolerance must be finite and nonnegative")
    signed = finger_tail_signed_distance(env).reshape(env.num_envs, 2, 2)
    error = signed.clamp_min(0.0) + (-signed - contact_penetration_tolerance).clamp_min(0.0)
    observed = signed < CONTACT_DISTANCE_CAP - 1.0e-7
    quality = torch.exp(-(error / contact_std).square()) * observed
    return quality, torch.isfinite(signed).all(dim=(1, 2))


def shoelace_grasp_quality(
    env: ManagerBasedRLEnv,
    contact_std: float,
    relative_speed_std: float,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    finger_cfgs: tuple[SceneEntityCfg, ...],
    contact_penetration_tolerance: float = 0.0,
    *,
    target_diameter: float,
    pad_distance_std: float,
    aperture_std: float,
    capture_std: float,
    geometry: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shared unfiltered grasp quality for acquisition, retention, and success.

    Args:
        env: Shoelace environment with current finger-tail contact observations.
        contact_std: Finger-tail surface-distance width [m].
        relative_speed_std: Worst-finger relative-speed width [m/s].
        cable_cfgs: Left and right cable scene entities.
        finger_cfgs: Four surface reference bodies, in left/right hand and opposing-finger order.
        contact_penetration_tolerance: Allowed contact-solver penetration [m].
        target_diameter: Target cable diameter [m].
        pad_distance_std: Finger reference-point localization width [m].
        aperture_std: Width for excess finger spacing above the target diameter [m].
        capture_std: Width for the tail leaving the opposing-finger segment [m].
        geometry: Geometry for the same state, entity selections, and target diameter, if available.

    Returns:
        Raw grasp qualities in robot-arm order, shape [N, 2], and finite-input flags, shape [N].
        Quality combines two-finger contact, aperture/capture geometry, and worst-finger relative speed.
        It does not measure force closure or certify physical retention.

    Raises:
        ValueError: If widths are not finite and positive or penetration tolerance is invalid.
    """
    if not all(
        math.isfinite(value) and value > 0
        for value in (relative_speed_std, target_diameter, pad_distance_std, aperture_std, capture_std)
    ):
        raise ValueError("Grasp widths and target diameter must be finite and positive")
    finger_contact, contact_finite = _finger_contact_quality(env, contact_std, contact_penetration_tolerance)
    if geometry is None:
        geometry = grasp_geometry(env, cable_cfgs, finger_cfgs, target_diameter)
    finite = geometry["finite"] & contact_finite
    contact = hamacher_product(finger_contact[:, :, 0], finger_contact[:, :, 1])
    aperture_error = (geometry["aperture"] - target_diameter).clamp_min(0.0)
    aperture = torch.exp(-(aperture_error / aperture_std).square())
    capture = torch.exp(-(geometry["capture_distance"] / capture_std).square())
    near_pad = torch.exp(-(geometry["pad_distance"] / pad_distance_std).square())
    shape_quality = hamacher_product(
        hamacher_product(aperture, capture), hamacher_product(near_pad[:, :, 0], near_pad[:, :, 1])
    )
    grasp = hamacher_product(contact, shape_quality)
    motion_match = 1.0 - torch.tanh(geometry["relative_speed"] / relative_speed_std)
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
