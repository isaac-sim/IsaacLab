# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state collection for homogeneous assembly variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils import instantiate

from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from . import reset_state
from .sampling import Sampler, SamplerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class variant_reset_accumulator(ManagerTermBase):
    """Accumulate validated reset states into a shared state table and sample from it.

    During the pre-collection phase, reset states are generated and validated
    against acceptance conditions until the table is full. After that, every
    call samples from the finalized table.

    All envs share a single table of validated states stored in
    env-origin-relative coordinates, so any env can be reset to any stored state.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.acceptance_conditions = cfg.params["acceptance_conditions"]

        reset_assets = list(cfg.params["reset_assets"])
        self._requested_reset_assets = reset_assets
        self.reset_assets = sorted((set(env.scene._articulations) | set(env.scene._rigid_objects)) & set(reset_assets))
        state_dim = sum(
            13 + 2 * asset.num_joints for name, asset in env.scene._articulations.items() if name in self.reset_assets
        ) + sum(
            13 + int(getattr(asset, "num_mesh_variants", 0) > 0)
            for name, asset in env.scene._rigid_objects.items()
            if name in self.reset_assets
        )
        self._state_target_size = int(cfg.params["state_table_size"])
        self._tag_term_name: str = cfg.params.get("state_tag_term", "reset_strategies")
        self._variant_context_name: str = cfg.params.get("variant_context", "assembly_variants")

        self.sampled_slots = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.sampled_cells = torch.full_like(self.sampled_slots, -1)
        self.precollecting_phase = True
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]

        self.state_tag_names: list[str] = []
        self.variant_names: tuple[str, ...] = ()
        self.state_data = torch.zeros((self._state_target_size, state_dim), device=env.device)
        self.state_cell_indices = torch.full((self._state_target_size,), -1, device=env.device, dtype=torch.long)
        self._num_cells = 1
        self._num_variants = 1
        self._variant_ids = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.cell_success_rate = torch.full((1, 1), torch.nan, device=env.device)
        self.cell_probabilities = torch.full((1, 1), torch.nan, device=env.device)

        self._success_monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        self.monitor_success_rate: torch.Tensor | None = None
        self.success_monitor = None
        self._sampler: Sampler | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        reset_assets: list[str],
        acceptance_conditions: dict,
        state_table_size: int,
        success_monitor_cfg: SuccessMonitorCfg,
        sampling: SamplerCfg,
        reset_term: EventTermCfg | None = None,
        state_tag_term: str = "reset_strategies",
        variant_context: str = "assembly_variants",
        report: bool = False,
        monitor_exclude_terms: list[str] | tuple[str, ...] = (),
    ):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if reset_assets and list(reset_assets) != self._requested_reset_assets:
            raise ValueError(
                "variant_reset_accumulator reset_assets changed after initialization. "
                f"Expected {self._requested_reset_assets}, got {list(reset_assets)}."
            )
        if self.precollecting_phase:
            if reset_term is None:
                raise RuntimeError("variant_reset_accumulator requires reset_term during precollection.")
            self._precollect(env, reset_term)

        if self.success_monitor is None:
            n_slots = self.state_data.shape[0]
            monitor_cfg = self._success_monitor_cfg
            self.success_monitor = instantiate(monitor_cfg, 1, n_slots, env.device)
            self.monitor_success_rate = self.success_monitor.success_rate

        progress = env.termination_manager.get_term_cfg("progress_context").func
        monitor_ids = env_ids
        if env_ids.numel() > 0 and monitor_exclude_terms:
            exclude_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for term_name in monitor_exclude_terms:
                if term_name in env.termination_manager._term_names:
                    term_idx = env.termination_manager._term_name_to_term_idx[term_name]
                    exclude_mask |= env.termination_manager._last_episode_dones[:, term_idx]
            monitor_ids = env_ids[~exclude_mask[env_ids]]
        if monitor_ids.numel() > 0:
            monitor_ids = monitor_ids[self.sampled_slots[monitor_ids] >= 0]
        if monitor_ids.numel() > 0:
            self.success_monitor.success_update(self.sampled_slots[monitor_ids], progress.is_success[monitor_ids])

        log: dict[str, float] = {}
        if report:
            self._update_cell_success_rate()
            log["Metrics/success_rate"] = self.success_monitor.get_mean_success_rate()

        if env_ids.numel() > 0:
            if self._sampler is None:
                self._sampler = instantiate(self._sampling_cfg, self.monitor_success_rate)

            probs, slot_idx = self._sample_marginally_balanced(len(env_ids))
            self.sampled_slots[env_ids] = slot_idx
            self.sampled_cells[env_ids] = self.state_cell_indices[slot_idx]
            reset_state.set_reset_state(env, self.state_data[slot_idx], env_ids, self.reset_assets, is_relative=True)
            env.extras["diagnostics"] = {
                "factory_reset_slot": self.sampled_slots,
                "factory_reset_cell": self.sampled_cells,
                "factory_reset_labels": tuple(self.state_tag_names),
                "factory_asset_variants": self.variant_names,
            }

        if report:
            env.extras.setdefault("log", {}).update(log)
            env.extras["heatmap"] = {
                "Metrics/ResetSuccessRate": {
                    "values": self.cell_success_rate,
                    "x_labels": self.variant_names,
                    "y_labels": tuple(self.state_tag_names),
                    "color_label": "Success rate",
                    "vmax": 1.0,
                },
                "Metrics/ResetProbs": {
                    "values": self.cell_probabilities,
                    "x_labels": self.variant_names,
                    "y_labels": tuple(self.state_tag_names),
                    "color_label": "Reset probability",
                    "value_format": ".2%",
                    "vmax": 1.0 / max(self.cell_probabilities.shape),
                },
            }

    def _precollect(self, env: ManagerBasedRLEnv, reset_term: EventTermCfg) -> None:
        variants = env.event_manager.get_term_cfg(self._variant_context_name).func
        tag_choice = reset_term.func.terms[self._tag_term_name].func
        self.state_tag_names = list(tag_choice.term_partitions)
        self.variant_names = tuple(variants.variant_names)
        self._variant_ids = variants.variant_ids
        self._num_variants = len(self.variant_names)
        self._num_cells = len(self.state_tag_names) * self._num_variants
        if self._state_target_size < self._num_cells:
            raise ValueError("state_table_size must cover every reset-label and assembly-variant cell.")
        self.cell_success_rate = torch.full(
            (len(self.state_tag_names), self._num_variants), torch.nan, device=env.device
        )
        self.cell_probabilities = torch.full_like(self.cell_success_rate, torch.nan)

        capacity = self.state_data.shape[0]
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        all_tag_indices = tag_choice.term_samples
        state_size = 0
        pbar = tqdm(total=capacity, desc="reset_accumulator")
        while state_size < capacity:
            counts = torch.bincount(self.state_cell_indices[:state_size], minlength=self._num_cells)
            cell_weights = counts.float().add_(1.0).reciprocal_()
            planned_cells = torch.multinomial(cell_weights, env.num_envs, replacement=True)
            tag_choice.prepare(torch.div(planned_cells, self._num_variants, rounding_mode="floor"))
            variants.prepare(planned_cells % self._num_variants)

            reset_term.func(env, all_env_ids, **reset_term.params)
            valid_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            reported = getattr(reset_term.func, "is_valid", None)
            if reported is not None:
                valid_mask &= reported
            for condition in self.acceptance_conditions.values():
                condition_func = condition if callable(condition) else condition.func
                valid_mask &= condition_func(env, all_env_ids)

            valid_env_ids = all_env_ids[valid_mask]
            if valid_env_ids.numel() == 0:
                continue
            tags = all_tag_indices[valid_env_ids].long()
            variant_ids = self._variant_ids[valid_env_ids].long()
            cells = tags * self._num_variants + variant_ids

            count = min(valid_env_ids.numel(), capacity - state_size)
            selected_env_ids = valid_env_ids[:count]
            selected_cells = cells[:count]
            states = reset_state.get_reset_state(env, selected_env_ids, self.reset_assets, is_relative=True)
            end = state_size + count
            self.state_data[state_size:end] = states
            self.state_cell_indices[state_size:end] = selected_cells
            pbar.update(end - state_size)
            state_size = end
        pbar.close()

        order = torch.argsort(self.state_cell_indices)
        self.state_data = self.state_data[order].contiguous()
        self.state_cell_indices = self.state_cell_indices[order].contiguous()
        counts = torch.bincount(self.state_cell_indices, minlength=self._num_cells)
        populated_counts = counts[counts > 0]
        print(
            f"[reset_accumulator] joint cells: {int((counts > 0).sum())}/{self._num_cells}, "
            f"states per populated cell: {int(populated_counts.min())}-{int(populated_counts.max())}"
        )
        del self.cfg.params["reset_term"]
        self.precollecting_phase = False

    def _sample_marginally_balanced(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Balance label and asset marginals without flattening their joint curriculum."""
        probabilities = self._sampler.probabilities()
        raw_cell_probabilities = probabilities.new_zeros(self._num_cells)
        raw_cell_probabilities.scatter_add_(0, self.state_cell_indices, probabilities)
        cell_probabilities = raw_cell_probabilities.view_as(self.cell_probabilities).clone()
        for _ in range(4):
            cell_probabilities.div_(cell_probabilities.sum(dim=1, keepdim=True) * cell_probabilities.shape[0])
            cell_probabilities.div_(cell_probabilities.sum(dim=0, keepdim=True) * cell_probabilities.shape[1])

        self.cell_probabilities.copy_(cell_probabilities)
        cell_scale = cell_probabilities.flatten().div_(raw_cell_probabilities)
        probabilities.mul_(cell_scale[self.state_cell_indices]).div_(probabilities.sum())
        return probabilities, self._sampler.sample(probabilities, num_samples)

    def _sample_variant(self, num_samples: int, variant_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample reset states for one scheduled assembly variant."""
        probabilities = self._sampler.probabilities()
        probabilities.mul_(self.state_cell_indices.remainder(self._num_variants) == variant_id)
        probabilities.div_(probabilities.sum())
        self.cell_probabilities.zero_()
        self.cell_probabilities.flatten().scatter_add_(0, self.state_cell_indices, probabilities)
        return probabilities, self._sampler.sample(probabilities, num_samples)

    def _update_cell_success_rate(self) -> None:
        successes = self.success_monitor.success_buf.sum(dim=1)
        episodes = self.success_monitor.success_size.to(successes.dtype)
        cell_successes = successes.new_zeros(self._num_cells).scatter_add_(0, self.state_cell_indices, successes)
        cell_episodes = episodes.new_zeros(self._num_cells).scatter_add_(0, self.state_cell_indices, episodes)
        rates = cell_successes / cell_episodes.clamp_min(1.0)
        rates[cell_episodes == 0] = torch.nan
        self.cell_success_rate.copy_(rates.view_as(self.cell_success_rate))

    def apply_sampled_slots(self, env_ids: torch.Tensor) -> None:
        """Restore the assigned table slots for the requested environments."""
        reset_state.set_reset_state(
            self._env, self.state_data[self.sampled_slots[env_ids]], env_ids, self.reset_assets, is_relative=True
        )


class PreparedTermChoice(ManagerTermBase):
    """Dispatch one prepared reset partition per environment."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_partitions: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        if not self.term_partitions:
            raise ValueError("PreparedTermChoice requires at least one reset partition.")
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        self.is_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        """Whether the strategy each env drew reported success, per env, for the last call."""
        self._next_samples = torch.zeros_like(self.term_samples)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, ManagerTermBase],
    ) -> None:
        self.term_samples[env_ids] = self._next_samples[env_ids]

        self.is_valid[env_ids] = True
        for i, (_, term_cfg) in enumerate(self.term_partitions.items()):
            term_ids = env_ids[self.term_samples[env_ids] == i]
            if term_ids.numel() > 0:
                term_cfg.func(env, term_ids, **term_cfg.params)
                reported = getattr(term_cfg.func, "is_valid", None)
                if reported is not None:
                    self.is_valid[term_ids] = reported[term_ids]

    def prepare(self, samples: torch.Tensor) -> None:
        """Set the partition indices used by the next call."""
        self._next_samples.copy_(samples)
