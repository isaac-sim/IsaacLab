# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .differential_ik import NewtonDifferentialIKController


@configclass
class NewtonDifferentialIKControllerCfg:
    """Configuration for :class:`NewtonDifferentialIKController`.

    The fields map one-to-one onto :class:`newton.controllers.ControllerDifferentialIKModelFree`, whose defaults
    they mirror. Newton validates the combination at construction, for example rejecting a :attr:`damping`
    for any :attr:`ik_method` other than ``"damped_least_squares"``. A gain left at ``None`` where Newton
    requires one is read live from the corresponding :meth:`~NewtonDifferentialIKController.compute` argument.
    """

    class_type: type[NewtonDifferentialIKController] | str = "{DIR}.differential_ik:NewtonDifferentialIKController"
    """The associated controller class."""

    ik_method: Literal["damped_least_squares", "pseudo_inverse", "transpose", "adaptive_damping", "truncated_svd"] = (
        "damped_least_squares"
    )
    """Inverse-Jacobian solve method. See :class:`newton.controllers.DifferentialIKMethod`."""

    bandwidth: float | None = MISSING
    """Output velocity gain [1/s]. ``None`` reads it per joint from :meth:`~NewtonDifferentialIKController.compute`.

    The joint target is ``joint_pos + dt * bandwidth * dq``, so ``bandwidth = 1 / dt`` applies the full
    correction each step.
    """

    damping: float | None = MISSING
    """Damped-least-squares regularization for ``"damped_least_squares"``; must be ``None`` for other methods.

    With ``"damped_least_squares"``, ``None`` reads it per environment from
    :meth:`~NewtonDifferentialIKController.compute`.
    """

    axis_weight: tuple[float, float, float, float, float, float] | None = None
    """Per-axis task weight ``(x, y, z, roll, pitch, yaw)``. Defaults to all ones.

    Zero-weighted axes are removed from the solve, so ``(1, 1, 1, 0, 0, 0)`` gives position-only IK.
    """

    adaptive_damping_min: float | None = None
    """Damping used away from singularities. Required by ``"adaptive_damping"``."""

    adaptive_damping_max: float | None = None
    """Damping reached at a singularity. Required by ``"adaptive_damping"``."""

    adaptive_damping_threshold: float | None = None
    """Smallest-singular-value threshold below which damping ramps up. Required by ``"adaptive_damping"``."""

    truncated_svd_threshold: float | None = None
    """Singular values below this are dropped. Required by ``"truncated_svd"``."""

    use_joint_limit_avoidance: bool = False
    """Whether to push joints away from their limits in the task null space.

    Requires :meth:`~NewtonDifferentialIKController.set_joint_pos_limits`.
    """

    joint_limit_avoidance_gain: float = 0.0
    """Joint-limit avoidance gain. Must be positive when :attr:`use_joint_limit_avoidance` is enabled."""

    joint_limit_avoidance_margin: float = 0.0
    """Distance from a joint limit [m or rad, depending on joint type] at which avoidance activates."""

    use_null_space_posture_control: bool = False
    """Whether to track a joint posture target in the task null space."""

    null_space_stiffness: float | None = None
    """Posture-control gain. ``None`` reads it per joint from :meth:`~NewtonDifferentialIKController.compute`."""

    null_space_damping: float | None = None
    """Null-space projector regularization, used by joint-limit avoidance and posture control.

    ``None`` reads it per environment from :meth:`~NewtonDifferentialIKController.compute`.
    """

    null_space_axes: tuple[float, float, float, float, float, float] | None = None
    """Task axes the null-space objectives must not disturb. Defaults to :attr:`axis_weight`."""
