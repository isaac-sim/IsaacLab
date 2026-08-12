# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the manager-based Pendulum MARL task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .shared import compute_success, links_upright, update_upright_steps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def time_out(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the final direct-task timeout signal."""
    return env.episode_length_buf >= env.max_episode_length - 1


def out_of_bounds(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, max_cart_pos: float
) -> torch.Tensor:
    """Return whether the cart or upper link crossed its final direct-task limit."""
    asset = env.scene[cart_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    cart_out_of_bounds = torch.any(torch.abs(joint_pos[:, cart_cfg.joint_ids]) > max_cart_pos, dim=1)
    pole_out_of_bounds = torch.any(torch.abs(joint_pos[:, pole_cfg.joint_ids]) > math.pi / 2, dim=1)
    return cart_out_of_bounds | pole_out_of_bounds


class ConsecutiveUprightSuccess(ManagerTermBase):
    """Track final direct-task success without contributing a done signal.

    The term belongs only to the cart termination manager, so it updates exactly
    once per control step even though both agents have identical termination
    managers. It deliberately always returns false.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._upright_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._success_required_steps = round(cfg.params["success_duration_s"] / env.step_dt)

    def update(self, upright: torch.Tensor) -> None:
        """Record one control-step upright sample."""
        self._upright_steps = update_upright_steps(self._upright_steps, upright)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Log completed episode success and clear only the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        termination_manager = self._env.termination_manager
        success = compute_success(
            termination_manager.time_outs[env_ids],
            termination_manager.terminated[env_ids],
            self._upright_steps[env_ids],
            self._success_required_steps,
        )
        self._env.parent.extras.setdefault("log", {})["Metrics/success_rate"] = success.float().mean().item()
        self._upright_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        pole_cfg: SceneEntityCfg,
        pendulum_cfg: SceneEntityCfg,
        success_upright_angle: float,
        success_duration_s: float,
    ) -> torch.Tensor:
        """Update success state and return no termination signal."""
        asset = env.scene[pole_cfg.name]
        joint_pos = asset.data.joint_pos.torch
        self.update(
            links_upright(
                joint_pos[:, pole_cfg.joint_ids].squeeze(-1),
                joint_pos[:, pendulum_cfg.joint_ids].squeeze(-1),
                success_upright_angle,
            )
        )
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
