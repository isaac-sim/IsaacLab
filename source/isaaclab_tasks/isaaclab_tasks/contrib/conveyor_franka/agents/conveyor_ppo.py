# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL checkpoint integration for the conveyor's adaptive reset evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rsl_rl.algorithms import PPO

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv
    from tensordict import TensorDict

    from ..mdp.curriculums import ConveyorResetCurriculum


class ConveyorPPO(PPO):
    """Use standard PPO while saving and restoring the task's reset curriculum."""

    _curriculum: ConveyorResetCurriculum | None

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ConveyorPPO:
        """Bind the active curriculum through RSL-RL's algorithm construction hook."""
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        curriculum = env.unwrapped.curriculum_manager
        algorithm._curriculum = (
            curriculum.cfg.reset_sampling.func if "reset_sampling" in curriculum.active_terms else None
        )
        return algorithm

    def save(self) -> dict:
        """Include reset evidence in the standard model and optimizer checkpoint."""
        state = super().save()
        if self._curriculum is not None:
            state["conveyor_reset_curriculum"] = self._curriculum.get_state()
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore reset evidence on resume; older policy-only checkpoints remain supported."""
        resume = super().load(loaded_dict, load_cfg, strict)
        if resume and self._curriculum is not None and "conveyor_reset_curriculum" in loaded_dict:
            self._curriculum.set_state(loaded_dict["conveyor_reset_curriculum"])
        return resume
