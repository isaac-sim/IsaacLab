# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-relative arm and continuous symmetric-gripper actions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

_GRIPPER_POSITION_TOLERANCE = 1.0e-6


def _bilateral_gripper_preload(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_target: torch.Tensor,
    *,
    min_deflection: float,
    max_velocity: float,
    max_command: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-finger drive deflection and a stable bilateral-contact mask."""
    finite = torch.isfinite(joint_position) & torch.isfinite(joint_velocity) & torch.isfinite(joint_target)
    deflection = torch.where(finite, torch.clamp(joint_position - joint_target, min=0.0), 0.0)
    bilateral = (
        finite.all(dim=-1)
        & (deflection.amin(dim=-1) >= float(min_deflection))
        & (joint_velocity.abs().amax(dim=-1) <= float(max_velocity))
        & (joint_target.amax(dim=-1) <= float(max_command) + _GRIPPER_POSITION_TOLERANCE)
    )
    return deflection, bilateral


class CurriculumJointPositionAction(JointPositionAction):
    """Reset-relative joint-position action with an optional early reference manifold.

    During the first supplied-grasp curriculum stages, unconstrained independent joint targets can
    leave the physically validated held-pour trajectory and destabilize the light source cup. The
    optional reference projection retains the policy's seven-dimensional interface, but projects
    its command onto the line from the reset offset to a validated joint target. Later stages use
    the ordinary full-rank joint-position action unchanged.
    """

    cfg: CurriculumJointPositionActionCfg

    def __init__(self, cfg: CurriculumJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._alpha = float(cfg.alpha)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        self._project_reference_through_stage = int(cfg.project_reference_through_stage)
        if self._project_reference_through_stage < -1:
            raise ValueError("project_reference_through_stage must be at least -1.")
        self._reference_action_magnitude = float(cfg.reference_action_magnitude)
        if not math.isfinite(self._reference_action_magnitude) or self._reference_action_magnitude <= 0.0:
            raise ValueError("reference_action_magnitude must be finite and positive.")
        self._reference_action_index = int(cfg.reference_action_index)
        if self._reference_action_index < 0 or self._reference_action_index >= self.action_dim:
            raise ValueError(f"reference_action_index must lie in [0, {self.action_dim - 1}].")
        if cfg.reference_target:
            if len(cfg.reference_target) != self.action_dim:
                raise ValueError(
                    f"reference_target must contain {self.action_dim} joint positions, got {len(cfg.reference_target)}."
                )
            if any(not math.isfinite(value) for value in cfg.reference_target):
                raise ValueError("reference_target must contain only finite joint positions.")
            self._reference_target = torch.tensor(cfg.reference_target, device=self.device).repeat(self.num_envs, 1)
        else:
            self._reference_target = None
        if self._project_reference_through_stage >= 0 and self._reference_target is None:
            raise ValueError("reference_target is required when reference projection is enabled.")
        self._previous_target = self._processed_actions.clone()

    @property
    def action_offset(self) -> torch.Tensor:
        """Per-environment joint-position action offset [rad]."""
        if not isinstance(self._offset, torch.Tensor):
            raise RuntimeError("Curriculum joint-position actions require a tensor action offset.")
        return self._offset

    @property
    def action_scale(self) -> torch.Tensor | float:
        """Joint-position displacement represented by one policy-action unit [rad]."""
        return self._scale

    def set_action_offset(
        self,
        offset: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set selected environments' zero-action joint targets [rad].

        Args:
            offset: Joint targets with shape ``(len(env_ids), action_dim)`` [rad].
            env_ids: Environments to update. If omitted, update every environment.

        Raises:
            ValueError: If :paramref:`offset` does not match the selected offset shape.
        """
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self.action_offset[selected].shape
        if offset.shape != expected_shape:
            raise ValueError(f"Action offset shape {tuple(offset.shape)} does not match {tuple(expected_shape)}.")
        offset = offset.to(device=self._offset.device, dtype=self._offset.dtype)
        self._offset[selected] = offset
        if hasattr(self, "_previous_target"):
            self._previous_target[selected] = offset

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map raw actions to low-pass-filtered joint targets [rad]."""
        self._raw_actions.copy_(actions)
        effective_actions = actions
        if self._project_reference_through_stage >= 0:
            projected_worlds = self._env.curriculum_stage <= self._project_reference_through_stage
            scale = self._scale
            reference_action = (self._reference_target - self.action_offset) / scale
            reference_norm_sq = reference_action.square().sum(dim=-1).clamp_min(1.0e-12)
            # A fixed policy coordinate preserves the meaning of "pour" when the curriculum reset
            # pose changes. The action term expands that scalar into the stage-specific correlated
            # joint trajectory; the remaining coordinates are ignored only in these supplied-grasp
            # stages and regain their ordinary joint semantics afterward.
            commanded_phase = actions[:, self._reference_action_index] / self._reference_action_magnitude
            previous_action = (self._previous_target - self.action_offset) / scale
            previous_phase = (previous_action * reference_action).sum(dim=-1) / reference_norm_sq
            # Pouring is one-way in this supplied-grasp stage. Never command a phase behind the
            # observed filtered target: this prevents a stochastic action from reversing the cup
            # after transfer, while requiring continued positive commands to advance the EMA.
            phase = torch.maximum(commanded_phase, previous_phase).clamp(0.0, 1.0)
            projected_actions = reference_action * phase.unsqueeze(-1)
            effective_actions = torch.where(projected_worlds.unsqueeze(-1), projected_actions, actions)
        self._processed_actions = effective_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        self._processed_actions.lerp_(self._previous_target, 1.0 - self._alpha)
        self._previous_target.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear selected raw actions and align their buffered targets with the reset origin."""
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
        self._processed_actions[selected] = self.action_offset[selected]
        self._previous_target[selected] = self.action_offset[selected]


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

    class_type: type[CurriculumJointPositionAction] = CurriculumJointPositionAction


class TrajectoryJointPositionAction(ActionTerm):
    """Filtered joint-position residuals around a monotonic per-environment reference trajectory.

    The first policy coordinate always controls forward progress and the remaining coordinates
    always control joint residuals. Curriculum resets only change the initial progress value and
    reference waypoints; they never repurpose policy coordinates. This avoids the distribution
    shift caused by making a pour-phase action become an ordinary arm-joint action after a stage
    promotion.
    """

    cfg: TrajectoryJointPositionActionCfg

    def __init__(self, cfg: TrajectoryJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names,
            preserve_order=cfg.preserve_order,
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError("Trajectory joint-position action resolved no joints.")
        if int(cfg.waypoint_count) < 2:
            raise ValueError("Trajectory joint-position action requires at least two waypoints.")
        self._waypoint_count = int(cfg.waypoint_count)
        self._alpha = float(cfg.alpha)
        self._phase_rate = float(cfg.phase_rate)
        self._approach_phase_rate = float(cfg.approach_phase_rate)
        self._transport_phase_rate = float(cfg.transport_phase_rate)
        self._waypoint_phases = torch.as_tensor(cfg.waypoint_phases, device=self.device, dtype=torch.float32)
        if self._waypoint_phases.shape != (self._waypoint_count,):
            raise ValueError(f"waypoint_phases must contain {self._waypoint_count} values.")
        if (
            not bool(torch.isfinite(self._waypoint_phases).all())
            or float(self._waypoint_phases[0]) != 0.0
            or float(self._waypoint_phases[-1]) != 1.0
            or not bool(torch.all(self._waypoint_phases[1:] > self._waypoint_phases[:-1]))
        ):
            raise ValueError("waypoint_phases must increase strictly from 0 to 1.")
        milestone_indices = (
            int(cfg.approach_waypoint),
            int(cfg.grasp_waypoint),
            int(cfg.lift_waypoint),
            int(cfg.align_waypoint),
        )
        if not (
            0
            < milestone_indices[0]
            < milestone_indices[1]
            < milestone_indices[2]
            < milestone_indices[3]
            < self._waypoint_count
        ):
            raise ValueError("Approach, grasp, lift, and align waypoints must be strictly ordered interior points.")
        self._approach_phase = float(self._waypoint_phases[milestone_indices[0]])
        self._grasp_phase = float(self._waypoint_phases[milestone_indices[1]])
        self._lift_phase = float(self._waypoint_phases[milestone_indices[2]])
        self._align_phase = float(self._waypoint_phases[milestone_indices[3]])
        self._grasp_gate_stage = int(cfg.grasp_gate_stage)
        self._grasp_dwell_steps = int(cfg.grasp_dwell_steps)
        self._grasp_max_tcp_distance = float(cfg.grasp_max_tcp_distance)
        self._grasp_max_linear_velocity = float(cfg.grasp_max_linear_velocity)
        self._grasp_max_angular_velocity = float(cfg.grasp_max_angular_velocity)
        self._approach_max_lateral_distance = float(cfg.approach_max_lateral_distance)
        self._approach_max_joint_error = float(cfg.approach_max_joint_error)
        self._approach_dwell_steps = int(cfg.approach_dwell_steps)
        self._approach_max_linear_velocity = float(cfg.approach_max_linear_velocity)
        self._approach_max_angular_velocity = float(cfg.approach_max_angular_velocity)
        self._align_max_distance = float(cfg.align_max_distance)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        if not math.isfinite(self._phase_rate) or self._phase_rate <= 0.0:
            raise ValueError("phase_rate must be finite and positive.")
        if not math.isfinite(self._approach_phase_rate) or self._approach_phase_rate <= 0.0:
            raise ValueError("approach_phase_rate must be finite and positive.")
        if not math.isfinite(self._transport_phase_rate) or self._transport_phase_rate <= 0.0:
            raise ValueError("transport_phase_rate must be finite and positive.")
        if self._grasp_dwell_steps <= 0:
            raise ValueError("grasp_dwell_steps must be positive.")
        if self._approach_dwell_steps <= 0:
            raise ValueError("approach_dwell_steps must be positive.")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                self._grasp_max_tcp_distance,
                self._grasp_max_linear_velocity,
                self._grasp_max_angular_velocity,
                self._approach_max_lateral_distance,
                self._approach_max_joint_error,
                self._approach_max_linear_velocity,
                self._approach_max_angular_velocity,
                self._align_max_distance,
            )
        ):
            raise ValueError("Grasp stability limits must be finite and positive.")

        residual_scale = torch.as_tensor(cfg.residual_scale, device=self.device, dtype=torch.float32)
        if residual_scale.ndim == 0:
            residual_scale = residual_scale.repeat(self._num_joints)
        if residual_scale.shape != (self._num_joints,):
            raise ValueError(
                f"residual_scale must be scalar or contain {self._num_joints} values, "
                f"got shape {tuple(residual_scale.shape)}."
            )
        if not bool(torch.isfinite(residual_scale).all()) or bool(torch.any(residual_scale < 0.0)):
            raise ValueError("residual_scale must contain finite nonnegative values.")
        self._residual_scale = residual_scale

        self._raw_actions = torch.zeros((self.num_envs, self._num_joints + 1), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._filtered_residual = torch.zeros_like(self._processed_actions)
        self._reference_waypoints = torch.zeros(
            (self.num_envs, self._waypoint_count, self._num_joints),
            device=self.device,
        )
        self._reference_phase = torch.zeros(self.num_envs, device=self.device)
        self._minimum_phase = torch.zeros(self.num_envs, device=self.device)
        self._grasp_dwell_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._approach_dwell_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._grasp_unlocked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._approach_unlocked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._lift_unlocked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._align_unlocked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._lower_limits = limits[..., 0]
        self._upper_limits = limits[..., 1]

    @property
    def action_dim(self) -> int:
        return self._num_joints + 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def reference_phase(self) -> torch.Tensor:
        """Current monotonic trajectory progress in ``[0, 1]``."""
        return self._reference_phase

    @property
    def reference_target(self) -> torch.Tensor:
        """Joint target on the reference trajectory before policy residuals [rad]."""
        return self._interpolate_reference(self._reference_phase)

    @property
    def reference_error(self) -> torch.Tensor:
        """Applied joint-target error relative to the physical arm joints [rad]."""
        arm_q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        return self._processed_actions - arm_q

    @property
    def milestone_status(self) -> torch.Tensor:
        """Observable task latches plus approach and grasp dwell progress."""
        approach_dwell = self._approach_dwell_count.float() / max(self._approach_dwell_steps, 1)
        dwell = self._grasp_dwell_count.float() / max(self._grasp_dwell_steps, 1)
        return torch.stack(
            (
                self._approach_unlocked.float(),
                self._grasp_unlocked.float(),
                self._lift_unlocked.float(),
                self._align_unlocked.float(),
                torch.clamp(approach_dwell, 0.0, 1.0),
                torch.clamp(dwell, 0.0, 1.0),
            ),
            dim=-1,
        )

    @property
    def lift_unlocked(self) -> torch.Tensor:
        """Whether each environment has demonstrated a held physical lift."""
        return self._lift_unlocked

    @property
    def residual_scale(self) -> torch.Tensor:
        """Joint displacement represented by one residual-action unit [rad]."""
        return self._residual_scale

    def set_reference(
        self,
        waypoints: torch.Tensor,
        phase: torch.Tensor,
        initial_target: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set selected trajectory waypoints, progress, and filtered target.

        Args:
            waypoints: Joint waypoints with shape ``(N, waypoint_count, num_joints)`` [rad].
            phase: Initial trajectory progress with shape ``(N,)``.
            initial_target: Initial filtered joint targets with shape ``(N, num_joints)`` [rad].
            env_ids: Environments to update. If omitted, update every environment.
        """
        selected = slice(None) if env_ids is None else env_ids
        expected_waypoints = self._reference_waypoints[selected].shape
        expected_phase = self._reference_phase[selected].shape
        expected_target = self._processed_actions[selected].shape
        if waypoints.shape != expected_waypoints:
            raise ValueError(f"Waypoint shape {tuple(waypoints.shape)} does not match {tuple(expected_waypoints)}.")
        if phase.shape != expected_phase:
            raise ValueError(f"Phase shape {tuple(phase.shape)} does not match {tuple(expected_phase)}.")
        if initial_target.shape != expected_target:
            raise ValueError(
                f"Initial target shape {tuple(initial_target.shape)} does not match {tuple(expected_target)}."
            )
        if not bool(torch.isfinite(waypoints).all()):
            raise ValueError("Reference waypoints must be finite.")
        if not bool(torch.isfinite(phase).all()) or bool(torch.any((phase < 0.0) | (phase > 1.0))):
            raise ValueError("Reference phase must be finite and lie in [0, 1].")
        self._reference_waypoints[selected] = waypoints
        self._reference_phase[selected] = phase
        self._minimum_phase[selected] = phase
        self._grasp_dwell_count[selected] = 0
        self._approach_dwell_count[selected] = 0
        self._approach_unlocked[selected] = phase > self._approach_phase + 1.0e-6
        # Gated stages must demonstrate fresh stable contact even when a supplied carry reset
        # begins beyond the grasp waypoint; ungated pour-only resets may continue immediately.
        gated_stage = self._env.curriculum_stage[selected] >= self._grasp_gate_stage
        self._grasp_unlocked[selected] = (phase > self._grasp_phase + 1.0e-6) & ~gated_stage
        self._lift_unlocked[selected] = (phase >= self._lift_phase) & ~gated_stage
        self._align_unlocked[selected] = (phase >= self._align_phase) & ~gated_stage
        self._processed_actions[selected] = initial_target
        self._filtered_residual[selected] = 0.0

    def _interpolate_reference(self, phase: torch.Tensor) -> torch.Tensor:
        lower_index = torch.bucketize(phase.contiguous(), self._waypoint_phases[1:-1])
        lower_phase = self._waypoint_phases[lower_index]
        upper_phase = self._waypoint_phases[lower_index + 1]
        fraction = ((phase - lower_phase) / (upper_phase - lower_phase)).unsqueeze(-1)
        # Smooth target velocity at every waypoint without changing the reference path.
        fraction = fraction.square() * (3.0 - 2.0 * fraction)
        env_ids = torch.arange(self.num_envs, device=self.device)
        lower = self._reference_waypoints[env_ids, lower_index]
        upper = self._reference_waypoints[env_ids, lower_index + 1]
        return torch.lerp(lower, upper, fraction)

    def _grasp_ready(self) -> torch.Tensor:
        held = self._held_ready(require_settled=True)
        cup_velocity = self._env.cup_velocity_w()
        return (
            held
            & (torch.linalg.vector_norm(cup_velocity[:, :3], dim=-1) <= self._grasp_max_linear_velocity)
            & (torch.linalg.vector_norm(cup_velocity[:, 3:], dim=-1) <= self._grasp_max_angular_velocity)
        )

    def _approach_ready(self) -> torch.Tensor:
        _, cross_track_error = self._env.grasp_approach_error()
        cup_velocity = self._env.cup_velocity_w()
        return (
            (cross_track_error <= self._approach_max_lateral_distance)
            & (torch.linalg.vector_norm(self.reference_error, dim=-1) <= self._approach_max_joint_error)
            & (torch.linalg.vector_norm(cup_velocity[:, :3], dim=-1) <= self._approach_max_linear_velocity)
            & (torch.linalg.vector_norm(cup_velocity[:, 3:], dim=-1) <= self._approach_max_angular_velocity)
        )

    def _held_ready(self, require_settled: bool = False) -> torch.Tensor:
        """Return whether the hand still geometrically holds the preloaded cup."""
        tcp_distance = torch.linalg.vector_norm(self._env.tcp_pos_e() - self._env.cup_grasp_point_e(), dim=-1)
        width_error = torch.abs(self._env.gripper_width() - float(self._env.gripper_grasp_width))
        gripper = self._env.action_manager.get_term("gripper_action")
        contact = gripper.bilateral_preload if require_settled else gripper.bilateral_contact
        return (
            (tcp_distance <= self._grasp_max_tcp_distance)
            & (width_error <= float(self._env.cfg.success_max_gripper_width_error))
            & contact
            & (
                gripper.commanded_position[:, 0]
                <= float(self._env.cfg.gripper_preload_pos) + _GRIPPER_POSITION_TOLERANCE
            )
        )

    def _lift_ready(self) -> torch.Tensor:
        return self._held_ready() & (
            self._env.cup_pose_e()[:, 2] - float(self._env.cup_reset_height)
            >= float(self._env.cfg.success_min_lift_height)
        )

    def _align_ready(self) -> torch.Tensor:
        source = self._env.cup_grasp_point_e()[:, :2]
        target = self._env.target_pose_e()[:, :2]
        desired = target + source.new_tensor(self._env.cfg.pour_source_offset_xy)
        return self._lift_ready() & (torch.linalg.vector_norm(source - desired, dim=-1) <= self._align_max_distance)

    @staticmethod
    def _phase_speed_command(phase_action: torch.Tensor) -> torch.Tensor:
        """Map a residual phase action to a bounded multiplier around nominal speed."""
        return 1.0 + 0.25 * torch.clamp(phase_action, -1.0, 1.0)

    @staticmethod
    def _monotonic_gate_limit(current_phase: torch.Tensor, gate_phase: float) -> torch.Tensor:
        """Hold at a milestone without rewinding curriculum resets that start after it."""
        return torch.maximum(current_phase, torch.full_like(current_phase, float(gate_phase)))

    def process_actions(self, actions: torch.Tensor) -> None:
        """Advance the reference phase and add bounded joint residuals."""
        self._raw_actions.copy_(actions)
        # Zero action is the validated nominal trajectory. The policy only adjusts timing within a
        # contact-safe range, matching the residual semantics of the seven joint coordinates.
        phase_command = self._phase_speed_command(actions[:, 0])
        approaching = self._reference_phase < self._grasp_phase
        transporting = self._grasp_unlocked & (self._reference_phase < self._align_phase)
        phase_rate = torch.where(
            approaching,
            torch.full_like(self._reference_phase, self._approach_phase_rate),
            torch.where(
                transporting,
                torch.full_like(self._reference_phase, self._transport_phase_rate),
                torch.full_like(self._reference_phase, self._phase_rate),
            ),
        )
        proposed_phase = self._reference_phase + float(self._env.step_dt) * phase_rate * phase_command
        proposed_phase = torch.maximum(proposed_phase, self._minimum_phase).clamp_max_(1.0)
        gated_stage = self._env.curriculum_stage >= self._grasp_gate_stage
        at_approach = self._reference_phase >= self._approach_phase - 1.0e-6
        approach_dwell = gated_stage & at_approach & self._approach_ready() & ~self._approach_unlocked
        self._approach_dwell_count.copy_(
            torch.where(
                approach_dwell,
                self._approach_dwell_count + 1,
                torch.zeros_like(self._approach_dwell_count),
            )
        )
        self._approach_unlocked |= self._approach_dwell_count >= self._approach_dwell_steps
        at_grasp = self._reference_phase >= self._grasp_phase - 1.0e-6
        grasp_ready = self._grasp_ready()
        dwell_active = gated_stage & at_grasp & grasp_ready & ~self._grasp_unlocked
        self._grasp_dwell_count.copy_(
            torch.where(dwell_active, self._grasp_dwell_count + 1, torch.zeros_like(self._grasp_dwell_count))
        )
        self._grasp_unlocked |= self._grasp_dwell_count >= self._grasp_dwell_steps
        self._lift_unlocked |= self._grasp_unlocked & self._lift_ready()
        self._align_unlocked |= self._lift_unlocked & self._align_ready()

        advancing = proposed_phase > self._reference_phase
        approach_gate = advancing & gated_stage & ~self._approach_unlocked & (proposed_phase > self._approach_phase)
        approach_limit = self._monotonic_gate_limit(self._reference_phase, self._approach_phase)
        proposed_phase = torch.where(
            approach_gate,
            approach_limit,
            proposed_phase,
        )
        grasp_gate = advancing & gated_stage & ~self._grasp_unlocked & (proposed_phase > self._grasp_phase)
        grasp_limit = self._monotonic_gate_limit(self._reference_phase, self._grasp_phase)
        proposed_phase = torch.where(grasp_gate, grasp_limit, proposed_phase)
        lift_gate = advancing & ~self._lift_unlocked & (proposed_phase > self._lift_phase)
        lift_limit = torch.maximum(self._reference_phase, torch.full_like(proposed_phase, self._lift_phase))
        proposed_phase = torch.where(lift_gate, lift_limit, proposed_phase)
        align_gate = advancing & ~self._align_unlocked & (proposed_phase > self._align_phase)
        align_limit = torch.maximum(self._reference_phase, torch.full_like(proposed_phase, self._align_phase))
        proposed_phase = torch.where(align_gate, align_limit, proposed_phase)
        self._reference_phase.copy_(proposed_phase)

        residual = torch.tanh(actions[:, 1:]) * self._residual_scale
        # Hold the validated approach and preload geometry stationary while bilateral finger
        # contact settles. Residual control resumes immediately after the grasp latch.
        pending_preload = gated_stage & at_approach & ~self._grasp_unlocked
        residual = torch.where(pending_preload.unsqueeze(-1), torch.zeros_like(residual), residual)
        self._filtered_residual.lerp_(residual, self._alpha)
        target = self._interpolate_reference(self._reference_phase) + self._filtered_residual
        target = torch.clamp(target, min=self._lower_limits, max=self._upper_limits)
        self._processed_actions.copy_(target)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear selected raw residual commands while preserving the reset reference."""
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0


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
    class_type: type[TrajectoryJointPositionAction] = TrajectoryJointPositionAction


class CurriculumGripperPositionAction(ActionTerm):
    """Filtered symmetric finger-position command with residual, incremental, and binary modes."""

    cfg: CurriculumGripperPositionActionCfg

    def __init__(self, cfg: CurriculumGripperPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError("CurriculumGripperPositionAction resolved no joints.")

        self._scale = float(cfg.scale)
        self._alpha = float(cfg.alpha)
        if not isinstance(cfg.use_incremental_target, bool):
            raise TypeError("use_incremental_target must be a bool.")
        self._use_incremental_target = cfg.use_incremental_target
        self._binary_threshold = cfg.binary_threshold
        self._close_position = float(cfg.close_position)
        self._neutral_position = float(cfg.neutral_position)
        self._open_position = float(cfg.open_position)
        self._default_position = self._close_position if cfg.default_position is None else float(cfg.default_position)
        self._contact_min_deflection = float(cfg.contact_min_deflection)
        self._contact_max_velocity = float(cfg.contact_max_velocity)
        self._force_open_stage = int(cfg.force_open_before_phase_stage)
        self._force_open_phase = float(cfg.force_open_before_phase)
        self._capture_max_lateral_distance = float(cfg.capture_max_lateral_distance)
        self._capture_max_vertical_distance = float(cfg.capture_max_vertical_distance)
        self._capture_max_joint_error = float(cfg.capture_max_joint_error)
        self._capture_dwell_steps = int(cfg.capture_dwell_steps)
        self._capture_max_linear_velocity = float(cfg.capture_max_linear_velocity)
        self._capture_max_angular_velocity = float(cfg.capture_max_angular_velocity)
        if not math.isfinite(self._scale) or self._scale <= 0.0:
            raise ValueError("Curriculum gripper action scale must be finite and positive.")
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        if self._binary_threshold is not None:
            if (
                isinstance(self._binary_threshold, bool)
                or not math.isfinite(self._binary_threshold)
                or not -1.0 < self._binary_threshold < 1.0
            ):
                raise ValueError("Binary gripper threshold must be finite and lie strictly between -1 and 1.")
            if self._use_incremental_target:
                raise ValueError("Binary and incremental gripper targets are mutually exclusive.")
        if (
            not math.isfinite(self._close_position)
            or not math.isfinite(self._neutral_position)
            or not math.isfinite(self._open_position)
            or not self._close_position <= self._neutral_position <= self._open_position
        ):
            raise ValueError(
                "Curriculum gripper positions must be finite with close_position <= neutral_position <= open_position."
            )
        if not math.isfinite(self._default_position) or not (
            self._close_position <= self._default_position <= self._neutral_position
        ):
            raise ValueError("default_position must lie in [close_position, neutral_position].")
        if self._force_open_stage < -1:
            raise ValueError("force_open_before_phase_stage must be at least -1.")
        if not 0.0 <= self._force_open_phase <= 1.0:
            raise ValueError("force_open_before_phase must lie in [0, 1].")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self._capture_max_lateral_distance, self._capture_max_vertical_distance)
        ):
            raise ValueError("Capture position limits must be finite and positive.")
        if not math.isfinite(self._capture_max_joint_error) or self._capture_max_joint_error <= 0.0:
            raise ValueError("capture_max_joint_error must be finite and positive.")
        if self._capture_dwell_steps <= 0:
            raise ValueError("capture_dwell_steps must be positive.")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self._capture_max_linear_velocity, self._capture_max_angular_velocity)
        ):
            raise ValueError("Capture velocity limits must be finite and positive.")
        if not math.isfinite(self._contact_min_deflection) or self._contact_min_deflection <= 0.0:
            raise ValueError("contact_min_deflection must be finite and positive.")
        if not math.isfinite(self._contact_max_velocity) or self._contact_max_velocity <= 0.0:
            raise ValueError("contact_max_velocity must be finite and positive.")
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._action_offset = torch.full(
            (self.num_envs, 1),
            self._default_position,
            device=self.device,
        )
        self._processed_actions = self._action_offset.expand(-1, self._num_joints).clone()
        self._capture_unlocked = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._capture_dwell_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def action_offset(self) -> torch.Tensor:
        """Per-environment residual-mode target and initialization position [m]."""
        return self._action_offset

    @property
    def commanded_position(self) -> torch.Tensor:
        """Current symmetric per-finger position target [m]."""
        return self._processed_actions[:, :1]

    @property
    def contact_deflection(self) -> torch.Tensor:
        """Per-finger position-drive deflection caused by contact [m]."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        deflection, _ = _bilateral_gripper_preload(
            joint_position,
            joint_velocity,
            self._processed_actions,
            min_deflection=self._contact_min_deflection,
            max_velocity=self._contact_max_velocity,
            max_command=self._neutral_position,
        )
        return deflection

    @property
    def bilateral_preload(self) -> torch.Tensor:
        """Whether both fingers have settled against the commanded preload."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        _, bilateral = _bilateral_gripper_preload(
            joint_position,
            joint_velocity,
            self._processed_actions,
            min_deflection=self._contact_min_deflection,
            max_velocity=self._contact_max_velocity,
            max_command=self._neutral_position,
        )
        return bilateral

    @property
    def bilateral_contact(self) -> torch.Tensor:
        """Whether both fingers remain deflected against the commanded cup contact."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        deflection, _ = _bilateral_gripper_preload(
            joint_position,
            joint_velocity,
            self._processed_actions,
            min_deflection=self._contact_min_deflection,
            max_velocity=self._contact_max_velocity,
            max_command=self._neutral_position,
        )
        finite = torch.isfinite(joint_position).all(dim=-1) & torch.isfinite(self._processed_actions).all(dim=-1)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return finite & command_valid & (deflection.amin(dim=-1) >= self._contact_min_deflection)

    @property
    def contact_quality(self) -> torch.Tensor:
        """Smooth bilateral-contact quality in ``[0, 1]``."""
        joint_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        deflection = self.contact_deflection
        deflection_quality = torch.clamp(deflection.amin(dim=-1) / self._contact_min_deflection, 0.0, 1.0)
        velocity_quality = torch.clamp(
            1.0 - joint_velocity.abs().amax(dim=-1) / self._contact_max_velocity,
            0.0,
            1.0,
        )
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return deflection_quality * velocity_quality * command_valid.float()

    @property
    def capture_status(self) -> torch.Tensor:
        """Observable gripper-capture latch and dwell progress."""
        dwell = self._capture_dwell_count.float() / max(self._capture_dwell_steps, 1)
        return torch.stack((self._capture_unlocked.float(), torch.clamp(dwell, 0.0, 1.0)), dim=-1)

    @property
    def IO_descriptor(self):
        descriptor = super().IO_descriptor
        descriptor.shape = (1,)
        descriptor.dtype = str(self.raw_actions.dtype)
        descriptor.action_type = "JointAction"
        descriptor.joint_names = self._joint_names
        descriptor.scale = self._scale
        return descriptor

    def set_action_offset(
        self,
        offset: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set selected environments' residual-mode target and initialization position [m]."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._action_offset[selected].shape
        if offset.shape != expected_shape:
            raise ValueError(f"Action offset shape {tuple(offset.shape)} does not match {tuple(expected_shape)}.")
        offset = offset.to(device=self._action_offset.device, dtype=self._action_offset.dtype)
        self._action_offset[selected] = offset
        self._processed_actions[selected] = offset.expand(-1, self._num_joints)

    def set_reset_position(
        self,
        position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Align the filtered target with selected physical reset positions [m]."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._action_offset[selected].shape
        if position.shape != expected_shape:
            raise ValueError(f"Reset-position shape {tuple(position.shape)} does not match {tuple(expected_shape)}.")
        position = position.to(device=self.device, dtype=self._processed_actions.dtype)
        expanded = position.expand(-1, self._num_joints)
        self._processed_actions[selected] = expanded

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        bounded_actions = torch.clamp(actions, -1.0, 1.0)
        if self._binary_threshold is None:
            action_base = self._processed_actions[:, :1] if self._use_incremental_target else self._action_offset
            target = torch.clamp(
                action_base + self._scale * bounded_actions,
                min=self._close_position,
                max=self._neutral_position,
            )
        else:
            target = torch.where(
                bounded_actions < self._binary_threshold,
                torch.full_like(bounded_actions, self._close_position),
                torch.full_like(bounded_actions, self._neutral_position),
            )
        if self._force_open_stage >= 0:
            arm_action = self._env.action_manager.get_term("arm_action")
            joint_error = torch.linalg.vector_norm(arm_action.reference_error, dim=-1)
            axial_error, cross_track_error = self._env.grasp_approach_error()
            cup_velocity = self._env.cup_velocity_w()
            capture_required = self._env.curriculum_stage >= self._force_open_stage
            capture_ready = (
                (arm_action.reference_phase >= self._force_open_phase)
                & (cross_track_error <= self._capture_max_lateral_distance)
                & (torch.abs(axial_error) <= self._capture_max_vertical_distance)
                & (joint_error <= self._capture_max_joint_error)
                & (torch.linalg.vector_norm(cup_velocity[:, :3], dim=-1) <= self._capture_max_linear_velocity)
                & (torch.linalg.vector_norm(cup_velocity[:, 3:], dim=-1) <= self._capture_max_angular_velocity)
            )
            dwell_active = capture_required & capture_ready & ~self._capture_unlocked
            self._capture_dwell_count.copy_(
                torch.where(dwell_active, self._capture_dwell_count + 1, torch.zeros_like(self._capture_dwell_count))
            )
            self._capture_unlocked |= self._capture_dwell_count >= self._capture_dwell_steps
            force_open = capture_required & ~self._capture_unlocked
            target = torch.where(force_open.unsqueeze(-1), torch.full_like(target, self._open_position), target)
        self._processed_actions.lerp_(target.expand(-1, self._num_joints), self._alpha)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
        if self._force_open_stage < 0:
            self._capture_unlocked[selected] = True
        else:
            self._capture_unlocked[selected] = self._env.curriculum_stage[selected] < self._force_open_stage
        self._capture_dwell_count[selected] = 0


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
    class_type: type[CurriculumGripperPositionAction] = CurriculumGripperPositionAction
