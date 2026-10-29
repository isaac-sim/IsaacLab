# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-guided reset sampling for conveyor transfer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from isaaclab_tasks.utils.reset_states import ResetStateSampler, ResetStateSamplerCfg
from isaaclab_tasks.utils.sampling import SamplerCfg
from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from .reset_events import ConveyorResetRecipe

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ConveyorResetCurriculum(ManagerTermBase):
    """Sample reset rows using the shared NIST sampler and completed-transfer outcomes.

    Reset geometry and command metadata belong to the reset term. Only recent success
    evidence lives here; standard PPO checkpoints start a fresh sampling history on resume.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._reset_term = env.event_manager.get_term_cfg("reset_from_state_table").func
        self.reset_sampler = ResetStateSampler(
            ResetStateSamplerCfg(success_monitor=cfg.params["success_monitor"], sampling=cfg.params["sampling"]),
            env.num_envs,
            self._reset_term.row_count,
            env.device,
        )
        self.reset_sampler.row_ids.copy_(self._reset_term.row_ids)
        self._reset_term.row_ids = self.reset_sampler.row_ids
        self.success_monitor = self.reset_sampler.success_monitor
        self.sampler = self.reset_sampler.sampler
        self._env_ids = torch.arange(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        success_monitor: SuccessMonitorCfg,
        sampling: SamplerCfg,
        command_name: str = "transfer",
        fixed_source_side_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Record finished episodes and draw rows for only the resetting environments."""
        del success_monitor, sampling
        ids = (
            self._env_ids[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        )
        command = env.command_manager.get_term(command_name)
        completed = (command.command_counter[ids] > 0) & (env.episode_length_buf[ids] > 0)
        completed_ids = ids[completed]
        succeeded = command.ever_success[completed_ids]
        row_mask = None
        if fixed_source_side_id is not None:
            if fixed_source_side_id not in (0, 1):
                raise ValueError("fixed_source_side_id must be 0 (left) or 1 (right).")
            row_mask = self._reset_term.source_side_ids == fixed_source_side_id
        self.reset_sampler.reset(ids, command.ever_success[ids], valid=completed, row_mask=row_mask)
        probabilities = self.reset_sampler.probabilities

        history_size = self.success_monitor.success_size
        history_successes = self.success_monitor.success_buf.sum(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        metrics: dict[str, torch.Tensor] = {
            "batch_success_rate": succeeded.float().sum() / max(completed_ids.numel(), 1),
            "batch_transfer_count": command.transfer_counts[completed_ids].float().sum()
            / max(completed_ids.numel(), 1),
            "batch_left_to_right_transfers": command.direction_transfer_counts[completed_ids, 0].float().sum()
            / max(completed_ids.numel(), 1),
            "batch_right_to_left_transfers": command.direction_transfer_counts[completed_ids, 1].float().sum()
            / max(completed_ids.numel(), 1),
            "row_coverage": (history_size > 0).float().mean(),
            "overall_success_rate": history_successes.sum() / history_size.sum().clamp_min(1),
            "sampling_entropy": entropy / math.log(probabilities.numel()),
        }
        for recipe in ConveyorResetRecipe:
            mask = self._reset_term.recipe_ids == int(recipe)
            metrics[f"recipe_{recipe.name.lower()}_probability"] = probabilities[mask].sum()
            metrics[f"recipe_{recipe.name.lower()}_success_rate"] = history_successes[mask].sum() / history_size[
                mask
            ].sum().clamp_min(1)
        for side_id, side_name in ((0, "left_to_right"), (1, "right_to_left")):
            mask = self._reset_term.source_side_ids == side_id
            metrics[f"direction_{side_name}_probability"] = probabilities[mask].sum()
            metrics[f"direction_{side_name}_success_rate"] = history_successes[mask].sum() / history_size[
                mask
            ].sum().clamp_min(1)
        return metrics
