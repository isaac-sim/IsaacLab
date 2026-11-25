# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive sampling over the Franka Pour reset-state dataset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.reset import BetaSamplingScoreCfg, ResetStateSampler, ResetStateSamplerCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils import index_fill_

from ..reset_sampler import ResetDatasetSamplerCfg

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


class PourResetDatasetCurriculum(ManagerTermBase):
    """Sample reset rows and learn from rank-local asynchronous episode outcomes."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        states = env._reset_dataset_states
        category = states["category"]
        objective = states["objective"]
        if category.ndim != 1 or category.numel() == 0:
            raise ValueError("Reset dataset category must be a non-empty one-dimensional tensor.")
        if objective.shape != category.shape:
            raise ValueError("Reset dataset category and objective tensors must be row-aligned.")
        self._row_count = int(category.numel())
        self._device = category.device

        sampling_mode = env.cfg.reset_dataset_sampling_mode
        if sampling_mode not in ("adaptive", "uniform"):
            raise ValueError("reset_dataset_sampling_mode must be 'adaptive' or 'uniform'.")
        sampler_cfg = env.cfg.reset_dataset_sampler
        if not isinstance(sampler_cfg, ResetDatasetSamplerCfg):
            raise TypeError("reset_dataset_sampler must be ResetDatasetSamplerCfg.")
        sampler_cfg.validate_values()

        self._replay_generator = torch.Generator(device=self._device)
        seed = env.cfg.seed if env.cfg.seed is not None else self._replay_generator.seed()
        self._replay_fraction = float(sampler_cfg.uniform_fraction) if sampling_mode == "adaptive" else 0.0
        self._restart_replay(seed)
        self._requested_rows = torch.empty_like(env.reset_dataset_row_id)

        # Core's Beta score has unit peak. Restore the raw kernel's scale so its
        # weight floor has the same effect as in the original Pour sampler.
        target = float(sampler_cfg.target_success_rate)
        kappa = float(sampler_cfg.kappa)
        peak = target ** (kappa * target) * (1.0 - target) ** (kappa * (1.0 - target))
        scores = [BetaSamplingScoreCfg(target=target, kappa=kappa, weight=peak)] if sampling_mode == "adaptive" else []
        with self._sampler_stream():
            self._sampler = ResetStateSampler(
                ResetStateSamplerCfg(
                    history_length=sampler_cfg.monitored_history_len,
                    scores=scores,
                    uniform_weight=float(sampler_cfg.epsilon),
                    seed=seed,
                ),
                env.num_envs,
                self._row_count,
                self._device,
            )

        self._frozen_rows = torch.arange(self._row_count, device=self._device, dtype=torch.long)
        top_grasp_count = env.cfg.reset_dataset_top_grasp_count
        if top_grasp_count is not None:
            grasp_rows = torch.nonzero(category == 1, as_tuple=False).flatten()
            if top_grasp_count > grasp_rows.numel():
                raise ValueError(
                    "reset_dataset_top_grasp_count exceeds the dataset grasp count: "
                    f"{top_grasp_count} > {grasp_rows.numel()}."
                )
            grasp_order = torch.argsort(objective[grasp_rows], descending=True, stable=True)
            self._frozen_rows = grasp_rows[grasp_order[:top_grasp_count]]
        if self._frozen_rows.numel() == 0:
            raise RuntimeError("The reset dataset must expose at least one playback row.")

        self._metrics_cache: dict[str, float] = {}
        self._local_reset_assignments_since_metrics = 0
        self._local_metrics_refresh_interval = max(1, env.num_envs)
        self._refresh_metrics_cache()

    def __call__(
        self,
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        """Record completed episodes and assign the next reset rows."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=self._device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        with self._sampler_stream():
            if env._reset_dataset_reseed_pending:
                seed = torch.initial_seed()
                self._sampler.reseed(seed)
                self._restart_replay(seed)

            num_envs = env_ids.numel()
            if num_envs == 0:
                return self._metrics(env)

            if env.cfg.curriculum_freeze:
                slots = torch.randint(
                    self._frozen_rows.numel(), (num_envs,), device=self._device, generator=self._replay_generator
                )
                self._reset_requested_rows(env_ids, self._frozen_rows[slots])
            else:
                completed = env.reset_buf & (env.episode_length_buf > 0)
                progress = env.termination_manager.get_term_cfg("learning_progress_context").func
                self._sampler.reset(env_ids=env_ids, success=progress.ever_success, valid=completed)
                self._apply_cyclic_replay(env_ids)

            env.reset_dataset_row_id[env_ids] = self._sampler.state_ids.torch[env_ids]
            index_fill_(env.pour_target_frac, env_ids, float(env.cfg.pour_target_frac))

        if not env.cfg.curriculum_freeze:
            self._local_reset_assignments_since_metrics += num_envs
            if self._local_reset_assignments_since_metrics >= self._local_metrics_refresh_interval:
                self._refresh_metrics_cache()
        return self._metrics(env)

    def _sampler_stream(self) -> wp.ScopedStream:
        """Order resets with Torch request writes and the sampler's persistent Warp state."""
        stream = None
        if self._device.type == "cuda":
            torch_stream = torch.cuda.current_stream(self._device)
            stream = wp.get_stream(wp.device_from_torch(self._device))
            if stream.cuda_stream != torch_stream.cuda_stream:
                stream = wp.stream_from_torch(torch_stream)
        return wp.ScopedStream(stream, sync_enter=True, sync_exit=True)

    def _metrics(self, env: FrankaPourEnv) -> dict[str, float]:
        """Return playback or adaptive-sampler metrics."""
        if env.cfg.curriculum_freeze:
            return {
                "frozen_pool_fraction": float(self._frozen_rows.numel() / self._row_count),
                "frozen_pool_size": float(self._frozen_rows.numel()),
            }
        return self._metrics_cache

    def _reset_requested_rows(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        self._requested_rows[env_ids] = rows
        self._sampler.reset(env_ids=env_ids, state_ids=self._requested_rows)

    def _restart_replay(self, seed: int) -> None:
        self._replay_generator.manual_seed(seed)
        self._replay_credit = 0.0
        self._replay_order = torch.randperm(self._row_count, device=self._device, generator=self._replay_generator)
        self._replay_cursor = 0

    def _apply_cyclic_replay(self, env_ids: torch.Tensor) -> None:
        """Override the reserved assignments with complete shuffled passes over the dataset."""
        count = env_ids.numel()
        credit = self._replay_credit + self._replay_fraction * count
        replay_count = min(count, math.floor(credit + 1.0e-12))
        self._replay_credit = credit - replay_count
        if replay_count == 0:
            return

        slots = torch.randperm(count, device=self._device, generator=self._replay_generator)[:replay_count]
        remaining = replay_count
        chunks = []
        while remaining:
            take = min(remaining, self._row_count - self._replay_cursor)
            chunks.append(self._replay_order[self._replay_cursor : self._replay_cursor + take])
            self._replay_cursor += take
            remaining -= take
            if self._replay_cursor == self._row_count:
                self._replay_order = torch.randperm(
                    self._row_count, device=self._device, generator=self._replay_generator
                )
                self._replay_cursor = 0
        self._reset_requested_rows(env_ids[slots], torch.cat(chunks))

    def _refresh_metrics_cache(self) -> None:
        """Refresh host logging metrics after a bounded number of reset assignments."""
        probabilities = self._sampler.probabilities.torch
        replay_fraction = self._replay_fraction
        probabilities = probabilities * (1.0 - replay_fraction) + replay_fraction / self._row_count
        top_count = max(1, math.ceil(0.01 * self._row_count))
        effective_fraction, top_mass = torch.stack(
            (
                probabilities.square().sum().reciprocal() / self._row_count,
                torch.topk(probabilities, top_count, sorted=False).values.sum(),
            )
        ).tolist()
        self._metrics_cache = {
            "sampler/effective_pool_fraction": effective_fraction,
            "sampler/top_1_percent_mass": top_mass,
            "sampler/uniform_replay_fraction": replay_fraction,
        }
        self._local_reset_assignments_since_metrics = 0
