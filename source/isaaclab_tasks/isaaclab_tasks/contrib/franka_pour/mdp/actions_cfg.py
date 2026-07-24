# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free configuration for Franka pour reset-relative arm and gripper actions."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import (
        CurriculumGripperPositionAction,
        CurriculumJointPositionAction,
        TrajectoryJointPositionAction,
    )


@configclass
class CurriculumJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`CurriculumJointPositionAction`."""

    alpha: float = 0.2
    """Weight of the current joint target in the exponential moving average."""

    reference_target: tuple[float, ...] = ()
    """Validated absolute joint target used by the early reference projection."""

    project_reference_through_stage: int = -1
    """Last curriculum stage projected onto the reset-to-reference segment, or ``-1`` to disable."""

    reference_action_magnitude: float = 1.0
    """Policy-space scalar magnitude that commands the complete reference segment."""

    reference_action_index: int = 0
    """Fixed policy-action coordinate used as the early-stage reference phase."""

    class_type: type[CurriculumJointPositionAction] | str = "{DIR}.actions:CurriculumJointPositionAction"


@configclass
class TrajectoryJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`TrajectoryJointPositionAction`."""

    joint_names: list[str] = MISSING
    preserve_order: bool = False
    waypoint_count: int = 6
    residual_scale: float | tuple[float, ...] = 0.05
    alpha: float = 0.2
    """Exponential smoothing weight applied only to policy joint residuals."""
    phase_rate: float = 1.0 / 3.5
    approach_phase_rate: float = 1.0 / 3.5
    """Phase rate before the grasp waypoint, kept slower for a contact-safe approach [1/s]."""
    transport_phase_rate: float = 0.5
    """Phase rate after a validated grasp and before the receiver-aligned pour [1/s]."""
    waypoint_phases: tuple[float, ...] = (0.0, 0.12, 0.24, 0.40, 0.62, 1.0)
    approach_waypoint: int = 1
    grasp_waypoint: int = 2
    lift_waypoint: int = 3
    align_waypoint: int = 4
    grasp_gate_stage: int = 3
    approach_max_lateral_distance: float = 0.01
    """Maximum cross-track TCP error perpendicular to the grasp-approach axis [m]."""
    approach_max_joint_error: float = 0.08
    approach_dwell_steps: int = 10
    """Consecutive centered, stationary steps required before the guarded grasp approach."""
    approach_max_linear_velocity: float = 0.01
    approach_max_angular_velocity: float = 0.1
    align_max_distance: float = 0.06
    """Maximum source-grasp-point error from the receiver-side pour pose [m]."""
    grasp_dwell_steps: int = 15
    grasp_max_tcp_distance: float = 0.01
    grasp_max_linear_velocity: float = 0.05
    grasp_max_angular_velocity: float = 0.5
    class_type: type[TrajectoryJointPositionAction] | str = "{DIR}.actions:TrajectoryJointPositionAction"


@configclass
class CurriculumGripperPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`CurriculumGripperPositionAction`."""

    joint_names: list[str] = MISSING
    scale: float = 0.04
    """Per-finger residual or incremental delta per policy-action unit [m]; unused in binary mode."""
    alpha: float = 0.2
    """Interpolation weight applied to the selected finger target."""
    use_incremental_target: bool = False
    """Whether actions increment the target by ``alpha * scale`` so zero action holds its position."""
    binary_threshold: float | None = None
    """Optional threshold selecting filtered close/maximum targets; values below it close."""
    close_position: float = 0.0
    neutral_position: float = 0.025
    """Largest per-finger command accepted from the action [m]."""
    open_position: float = 0.04
    default_position: float | None = None
    """Per-finger residual-mode zero command and initial target [m]. ``None`` uses ``close_position``."""
    limit_to_preload: bool = True
    """Whether task validation restricts action targets to the contact-safe preload interval."""
    contact_min_deflection: float = 0.001
    """Minimum settled position-drive deflection required on each finger [m]."""
    contact_max_velocity: float = 0.005
    """Maximum absolute finger speed accepted as settled bilateral contact [m/s]."""
    force_open_before_phase_stage: int = -1
    """First stage that holds the hand open during the approach phase, or ``-1`` to disable."""
    force_open_before_phase: float = 0.25
    """Reference phase below which configured approach stages force the hand open."""
    capture_max_lateral_distance: float = 0.005
    """Maximum cross-track TCP error perpendicular to the grasp-approach axis [m]."""
    capture_max_vertical_distance: float = 0.008
    """Maximum absolute TCP error along the grasp-approach axis [m].

    The field retains its historical name for configuration compatibility.
    """
    capture_max_joint_error: float = 0.08
    """Maximum reference-to-physical arm-joint error that releases the interlock [rad]."""
    capture_dwell_steps: int = 5
    """Consecutive centered, stationary steps required before finger closure is enabled."""
    capture_max_linear_velocity: float = 0.02
    """Maximum source-cup linear speed during capture qualification [m/s]."""
    capture_max_angular_velocity: float = 0.2
    """Maximum source-cup angular speed during capture qualification [rad/s]."""
    class_type: type[CurriculumGripperPositionAction] | str = "{DIR}.actions:CurriculumGripperPositionAction"
