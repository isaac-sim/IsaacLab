# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction

from isaaclab_tasks.contrib.deploy.mdp.delayed_joint_actions_cfg import (
    DelayedRelativeJointPositionActionCfg,
    FlexivDynamicsAwareRelativeJointPositionActionCfg,
    ShapedDelayedRelativeJointPositionActionCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint-position action that applies a delayed absolute joint target."""

    cfg: DelayedRelativeJointPositionActionCfg

    def __init__(self, cfg: DelayedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if self.cfg.latency_s < 0.0:
            raise ValueError("latency_s must be non-negative")
        if self.cfg.latency_steps is not None and self.cfg.latency_steps < 0:
            raise ValueError("latency_steps must be non-negative")

        # PhysX calls apply_actions() once per physics step. Newton graph mode may
        # fold decimation internally, in which case apply_actions() runs once per env step.
        apply_dt = env.step_dt if getattr(env, "_physics_handles_decimation", False) else env.physics_dt
        if self.cfg.latency_steps is None:
            self._delay_steps = 0 if self.cfg.latency_s == 0.0 else max(1, round(self.cfg.latency_s / apply_dt))
        else:
            self._delay_steps = int(self.cfg.latency_steps)
        self._effective_latency_s = self._delay_steps * apply_dt

        current_target = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids].clone()
        self._latest_target = current_target.clone()
        self._delayed_target = current_target.clone()
        self._target_buffer = None
        if self._delay_steps > 0:
            self._target_buffer = current_target.unsqueeze(0).repeat(self._delay_steps, 1, 1)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        current_joint_pos = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        self._latest_target = current_joint_pos + self.processed_actions

    def apply_actions(self):
        if self._delay_steps > 0:
            self._delayed_target = self._target_buffer[0].clone()
            if self._delay_steps > 1:
                self._target_buffer[:-1] = self._target_buffer[1:].clone()
            self._target_buffer[-1] = self._latest_target
        else:
            self._delayed_target = self._latest_target

        self._asset.set_joint_position_target_index(target=self._delayed_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        reset_ids = slice(None) if env_ids is None else env_ids
        current_target = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        self._latest_target[reset_ids] = current_target[reset_ids]
        self._delayed_target[reset_ids] = current_target[reset_ids]
        if self._target_buffer is not None:
            selected_target = current_target[reset_ids]
            if selected_target.ndim == 1:
                selected_target = selected_target.unsqueeze(0)
            reset_target = selected_target.unsqueeze(0).repeat(self._delay_steps, 1, 1)
            self._target_buffer[:, reset_ids, :] = reset_target


class ShapedDelayedRelativeJointPositionAction(DelayedRelativeJointPositionAction):
    """Delayed relative joint-position action with command-side velocity and acceleration limits."""

    cfg: ShapedDelayedRelativeJointPositionActionCfg

    def __init__(self, cfg: ShapedDelayedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if self.cfg.command_velocity_limit < 0.0:
            raise ValueError("command_velocity_limit must be non-negative")
        if self.cfg.command_acceleration_limit < 0.0:
            raise ValueError("command_acceleration_limit must be non-negative")

        # Command shaping models the external robot command loop, so it advances
        # once per env/control step. PhysX still receives the latest shaped target
        # on every physics substep through apply_actions().
        control_dt = env.step_dt
        if self.cfg.latency_steps is None:
            self._delay_steps = 0 if self.cfg.latency_s == 0.0 else max(1, round(self.cfg.latency_s / control_dt))
        else:
            self._delay_steps = int(self.cfg.latency_steps)
        self._effective_latency_s = self._delay_steps * control_dt

        current_target = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids].clone()
        self._latest_target = current_target.clone()
        self._delayed_target = current_target.clone()
        self._target_buffer = None
        if self._delay_steps > 0:
            self._target_buffer = current_target.unsqueeze(0).repeat(self._delay_steps, 1, 1)

        self._shape_dt = control_dt
        self._shaped_target = self._delayed_target.clone()
        self._shaped_velocity = torch.zeros_like(self._shaped_target)

    def process_actions(self, actions: torch.Tensor):
        RelativeJointPositionAction.process_actions(self, actions)
        current_joint_pos = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        self._latest_target = current_joint_pos + self.processed_actions

        if self._delay_steps > 0:
            delayed_target = self._target_buffer[0].clone()
            if self._delay_steps > 1:
                self._target_buffer[:-1] = self._target_buffer[1:].clone()
            self._target_buffer[-1] = self._latest_target
        else:
            delayed_target = self._latest_target

        self._delayed_target = delayed_target
        self._shaped_target, self._shaped_velocity = self._shape_position_target(
            delayed_target,
            self._shaped_target,
            self._shaped_velocity,
        )

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._shaped_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        reset_ids = slice(None) if env_ids is None else env_ids
        current_target = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        self._shaped_target[reset_ids] = current_target[reset_ids]
        self._shaped_velocity[reset_ids] = 0.0

    def _shape_position_target(
        self,
        desired_target: torch.Tensor,
        shaped_target: torch.Tensor,
        shaped_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.command_velocity_limit == 0.0 and self.cfg.command_acceleration_limit == 0.0:
            return desired_target, shaped_velocity

        target_velocity = (desired_target - shaped_target) / self._shape_dt
        if self.cfg.command_velocity_limit > 0.0:
            target_velocity = torch.clamp(
                target_velocity,
                min=-self.cfg.command_velocity_limit,
                max=self.cfg.command_velocity_limit,
            )

        if self.cfg.command_acceleration_limit > 0.0:
            max_delta_velocity = self.cfg.command_acceleration_limit * self._shape_dt
            target_velocity = shaped_velocity + torch.clamp(
                target_velocity - shaped_velocity,
                min=-max_delta_velocity,
                max=max_delta_velocity,
            )

        next_target = shaped_target + target_velocity * self._shape_dt
        crossed = ((desired_target - shaped_target) * (desired_target - next_target)) < 0.0
        if crossed.any():
            next_target = torch.where(crossed, desired_target, next_target)
            target_velocity = torch.where(crossed, torch.zeros_like(target_velocity), target_velocity)

        return next_target, target_velocity


class FlexivDynamicsAwareRelativeJointPositionAction(DelayedRelativeJointPositionAction):
    """Flexiv-style relative joint-position controller with latency and dynamics-aware damping.

    Flexiv's robot-side impedance API accepts joint stiffness and damping ratios.
    This action term approximates that behavior by using the simulator's implicit
    joint drives, writing the configured stiffness directly, and updating damping
    at every action application as:

    .. math::
        D_i(q) = s \\cdot 2 \\zeta_i \\sqrt{K_i M_{ii}(q)}

    where ``M_ii(q)`` is the current diagonal entry of the articulated mass
    matrix for joint ``i``. This is intentionally a controller/action term, not
    a plotting transform, so replay and policy rollout see the same command path.
    """

    cfg: FlexivDynamicsAwareRelativeJointPositionActionCfg

    def __init__(self, cfg: FlexivDynamicsAwareRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if self.cfg.mass_matrix_mode != "diagonal":
            raise ValueError("Only mass_matrix_mode='diagonal' is currently supported.")
        if self.cfg.damping_scale <= 0.0:
            raise ValueError("damping_scale must be positive.")
        if self.cfg.min_effective_inertia <= 0.0:
            raise ValueError("min_effective_inertia must be positive.")
        if self.cfg.max_damping is not None and self.cfg.max_damping <= 0.0:
            raise ValueError("max_damping must be positive when provided.")

        if isinstance(self._joint_ids, slice):
            self._joint_ids_tensor = torch.arange(self._asset.num_joints, device=self.device, dtype=torch.long)
        else:
            self._joint_ids_tensor = torch.tensor(self._joint_ids, device=self.device, dtype=torch.long)

        self._kp = self._resolve_joint_parameter(self.cfg.stiffness, "stiffness")
        self._zeta = self._resolve_joint_parameter(self.cfg.damping_ratio, "damping_ratio")
        self._min_effective_inertia = torch.full_like(self._kp, float(self.cfg.min_effective_inertia))
        self._have_written_stiffness = False
        self._latest_damping = torch.zeros(self.num_envs, self.action_dim, device=self.device, dtype=self._kp.dtype)

    def apply_actions(self):
        self._write_stiffness_if_needed()
        self._update_dynamics_aware_damping()
        super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self.cfg.rewrite_stiffness_on_reset:
            self._have_written_stiffness = False

    def _resolve_joint_parameter(self, value: float | dict[str, float], field_name: str) -> torch.Tensor:
        if isinstance(value, (float, int)):
            values = [float(value)] * self.action_dim
        elif isinstance(value, dict):
            values = []
            for joint_name in self._joint_names:
                matches = []
                if joint_name in value:
                    matches.append(float(value[joint_name]))
                for pattern, pattern_value in value.items():
                    if pattern == joint_name:
                        continue
                    if re.fullmatch(pattern, joint_name):
                        matches.append(float(pattern_value))
                if not matches:
                    raise ValueError(f"Missing {field_name} value for joint '{joint_name}'.")
                values.append(matches[-1])
        else:
            raise TypeError(f"{field_name} must be a float or dict[str, float].")

        return torch.tensor(values, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)

    def _write_stiffness_if_needed(self) -> None:
        if self._have_written_stiffness and not self.cfg.rewrite_stiffness_each_step:
            return

        self._asset.write_joint_stiffness_to_sim_index(
            stiffness=self._kp,
            joint_ids=self._joint_ids_tensor,
        )
        self._have_written_stiffness = True

    def _mass_matrix_tensor(self) -> torch.Tensor:
        data = self._asset.data
        if not hasattr(data, "mass_matrix"):
            raise RuntimeError(
                "FlexivDynamicsAwareRelativeJointPositionAction requires asset.data.mass_matrix. "
                "Use the Newton backend or another backend that exposes articulated mass matrices."
            )

        mass_matrix = data.mass_matrix
        if isinstance(mass_matrix, wp.array):
            return wp.to_torch(mass_matrix)
        if hasattr(mass_matrix, "torch"):
            return mass_matrix.torch
        if isinstance(mass_matrix, torch.Tensor):
            return mass_matrix
        return torch.as_tensor(mass_matrix, device=self.device)

    def _update_dynamics_aware_damping(self) -> None:
        mass_matrix = self._mass_matrix_tensor()
        effective_inertia = mass_matrix[:, self._joint_ids_tensor, self._joint_ids_tensor]
        effective_inertia = torch.maximum(effective_inertia, self._min_effective_inertia)
        damping = self.cfg.damping_scale * 2.0 * self._zeta * torch.sqrt(self._kp * effective_inertia)
        if self.cfg.max_damping is not None:
            damping = torch.clamp(damping, max=float(self.cfg.max_damping))

        self._latest_damping = damping
        self._asset.write_joint_damping_to_sim_index(
            damping=damping,
            joint_ids=self._joint_ids_tensor,
        )
