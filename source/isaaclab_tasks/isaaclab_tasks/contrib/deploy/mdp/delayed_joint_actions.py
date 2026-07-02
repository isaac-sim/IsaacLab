# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import DelayBuffer
from isaaclab_tasks.contrib.deploy.mdp.delayed_joint_actions_cfg import (
    DelayedRelativeJointPositionActionCfg,
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

        current_target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._latest_target = current_target.clone()
        self._delayed_target = current_target.clone()
        self._target_delay_buffer = self._make_target_delay_buffer(current_target)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        current_joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._latest_target = current_joint_pos + self.processed_actions

    def apply_actions(self):
        self._delayed_target = self._compute_delayed_target(self._latest_target)
        self._asset.set_joint_position_target_index(target=self._delayed_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        reset_ids = slice(None) if env_ids is None else env_ids
        current_target = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._latest_target[reset_ids] = current_target[reset_ids]
        self._delayed_target[reset_ids] = current_target[reset_ids]
        if self._target_delay_buffer is not None:
            selected_target = current_target[reset_ids]
            self._reset_target_delay_buffer(reset_ids, selected_target)

    def _make_target_delay_buffer(self, initial_target: torch.Tensor) -> DelayBuffer | None:
        if self._delay_steps <= 0:
            return None

        target_delay_buffer = DelayBuffer(self._delay_steps, self.num_envs, device=self.device)
        target_delay_buffer.set_time_lag(self._delay_steps)
        target_delay_buffer.compute(initial_target)
        return target_delay_buffer

    def _compute_delayed_target(self, target: torch.Tensor) -> torch.Tensor:
        if self._target_delay_buffer is None:
            return target
        return self._target_delay_buffer.compute(target)

    def _reset_target_delay_buffer(self, reset_ids: Sequence[int] | slice, selected_target: torch.Tensor) -> None:
        if selected_target.ndim == 1:
            selected_target = selected_target.unsqueeze(0)

        self._target_delay_buffer.reset(reset_ids)
        circular_buffer = self._target_delay_buffer._circular_buffer
        if circular_buffer._buffer is not None:
            # DelayBuffer.reset() clears reset rows; restore the reset joint pose so
            # newly-reset envs hold that target for the configured command latency.
            circular_buffer._buffer[:, reset_ids] = selected_target
            circular_buffer._num_pushes[reset_ids] = 1
            circular_buffer._need_reset = True


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

        current_target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._latest_target = current_target.clone()
        self._delayed_target = current_target.clone()
        self._target_delay_buffer = self._make_target_delay_buffer(current_target)

        self._shape_dt = control_dt
        self._shaped_target = self._delayed_target.clone()
        self._shaped_velocity = torch.zeros_like(self._shaped_target)

    def process_actions(self, actions: torch.Tensor):
        RelativeJointPositionAction.process_actions(self, actions)
        current_joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._latest_target = current_joint_pos + self.processed_actions

        delayed_target = self._compute_delayed_target(self._latest_target)
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
        current_target = self._asset.data.joint_pos.torch[:, self._joint_ids]
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
