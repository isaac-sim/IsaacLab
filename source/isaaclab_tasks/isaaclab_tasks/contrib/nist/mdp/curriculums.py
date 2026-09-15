# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume initial value element and final value element have the same structure as data
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    # Otherwise it's a leaf scalar: do the interpolation
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val.item())
    return new_val.item()


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """Interpolate scalars in an arbitrarily nested list or tuple."""
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        return mdp.modify_env_param.NO_CHANGE

    initial = torch.tensor(initial_value, device=env.device).tolist()
    final = torch.tensor(final_value, device=env.device).tolist()
    return _recurse(initial, final, data, frac)


class DifficultyScheduler(ManagerTermBase):
    """Adjust difficulty from the reset bank's mean success rate."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_rate_callback: str,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
    ):
        success_rates = eval(success_rate_callback, {}, {"env": env})  # noqa: S307
        if success_rates is None:
            return self.difficulty_frac
        success_rate = success_rates.mean()

        move_up = success_rate > 0.8
        demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up,
            self.current_adr_difficulties[env_ids] + 1,
            demot,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac
