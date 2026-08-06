# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic reward terms shared across terrain and factory tasks.

- :func:`command_task_reward` — passes the command term's terminal multiplicative
  reward through to the reward manager. Works with any command term that
  exposes a ``task_reward`` attribute (notably :class:`MultiTaskCommand`).
- :func:`action_l2_clamped` / :func:`action_rate_l2_clamped` — generic L2
  action penalties with a saturation clamp.
- :func:`mechanical_power` — Σ |τⱼ · q̇ⱼ| across an articulation's joints,
  NaN-safe. Useful as a soft-safety reward signal for any actuated robot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


def command_task_reward(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Expose the command term's terminal multiplicative reward as a reward term.

    For :class:`MultiTaskCommand`, ``task_reward`` is non-zero only on terminal
    steps — bind with ``weight=1.0`` to use it as the sole task reward.
    """
    return env.command_manager.get_term(command_name).task_reward


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-5000, 5000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-5000, 5000)


def mechanical_power(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Σ |τⱼ · q̇ⱼ| across the articulation's joints. NaN-safe.

    Total instantaneous absolute mechanical power [W]. NaN/Inf outputs (rare —
    seen briefly during reset on some backends) are clamped to 0.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    work = torch.sum((wp.to_torch(robot.data.applied_torque) * wp.to_torch(robot.data.joint_vel)).abs(), dim=1)
    return torch.where(torch.isfinite(work), work, torch.zeros_like(work))


class contact_penalty(ManagerTermBase):
    """Penalize contacts on selected sensor bodies."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        contact_sensor_cfg: SceneEntityCfg | None = cfg.params.get("contact_sensor_cfg")
        exclude_contact_sensor_cfg: SceneEntityCfg | None = cfg.params.get("exclude_contact_sensor_cfg")
        if (contact_sensor_cfg is None) == (exclude_contact_sensor_cfg is None):
            raise ValueError("contact_penalty expects exactly one of contact_sensor_cfg or exclude_contact_sensor_cfg.")

        if contact_sensor_cfg is not None:
            self.contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
            self.body_ids = contact_sensor_cfg.body_ids
        else:
            self.contact_sensor: ContactSensor = env.scene.sensors[exclude_contact_sensor_cfg.name]
            if exclude_contact_sensor_cfg.body_ids == slice(None):
                self.body_ids = []
            else:
                exclude_body_ids = set(exclude_contact_sensor_cfg.body_ids)
                self.body_ids = [
                    body_id for body_id in range(self.contact_sensor.num_sensors) if body_id not in exclude_body_ids
                ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        contact_sensor_cfg: SceneEntityCfg | None = None,
        exclude_contact_sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        net_contact_forces = wp.to_torch(self.contact_sensor.data.net_forces_w_history)
        is_contact = torch.max(torch.linalg.norm(net_contact_forces[:, :, self.body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1)


def progress_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance: torch.Tensor = getattr(context_term, "z_distance")
    return torch.where(orientation_aligned & position_centered, 1 - torch.tanh(z_distance / std), 0.0)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")
    return torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)
