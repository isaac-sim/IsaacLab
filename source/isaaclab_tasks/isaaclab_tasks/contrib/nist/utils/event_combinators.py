# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic reset-strategy combinators.

Three :class:`~isaaclab.managers.ManagerTermBase` subclasses that compose
reset strategies without baking in robot or task semantics:

- :class:`reset_accumulator` — pre-collect validated reset states into a
  shared state table, then sample from the static finalized slots.
- :class:`TermChoice` — partition envs over a set of sub-event-terms;
  uniform or success-rate-weighted sampling.
- :class:`ChainedResetTerms` — sequentially apply a list of reset terms,
  optionally gated by a per-env Bernoulli probability.

The combinators couple to a ``progress_context`` termination term that
exposes ``.is_success``: a per-env bool tensor read at runtime to update
the success monitor. Any domain (locomotion, manipulation, …) can satisfy
this contract by registering a termination term named ``progress_context``
whose function exposes an ``.is_success`` attribute.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_tasks.core.lift.mdp.events_cfg import SuccessMonitorCfg

from . import reset_state
from .grid_downsample import extract_features, grid_bucket_downsample
from .sampling import Sampler, SamplerCfg
from .state_layout import StateLayout

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class reset_accumulator(ManagerTermBase):
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

        # Filter to scene-resident assets: callers may include "optional" assets
        # that only exist for a subset of presets.
        reset_assets = list(cfg.params["reset_assets"])
        self._requested_reset_assets = reset_assets
        self.reset_assets = sorted((set(env.scene._articulations) | set(env.scene._rigid_objects)) & set(reset_assets))
        state_dim = reset_state.get_reset_state(env, torch.tensor([0], device=env.device), self.reset_assets).shape[-1]
        self._state_target_size = int(cfg.params["state_table_size"])
        self._state_fps_features = cfg.params.get("state_table_fps_features")
        self._state_tag_names_bind: str | None = cfg.params.get("state_tag_names_bind")
        self._tag_indices_bind: str | None = cfg.params.get("state_tag_indices_bind")

        oversample_capacity = int(self._state_target_size * float(cfg.params.get("state_table_oversample_ratio", 1.0)))
        self.sampled_slots = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.precollecting_phase = True
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs

        self.state_tag_names: list[str] | None = None
        self.state_data = torch.zeros((oversample_capacity, state_dim), device=env.device)
        self.state_tag_indices = torch.full((oversample_capacity,), -1, device=env.device, dtype=torch.int64)

        self._success_monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        self.monitor_success_rate: torch.Tensor | None = None
        self.success_monitor = None
        self._sampler: Sampler | None = None
        self._wandb_3d_log_state: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reset_term: EventTermCfg,
        reset_assets: list[str],
        acceptance_conditions: dict,
        state_table_size: int,
        success_monitor_cfg: SuccessMonitorCfg,
        sampling: SamplerCfg,
        state_table_oversample_ratio: float = 1.0,
        state_table_fps_features: Callable | None = None,
        state_tag_names_bind: str | None = None,
        state_tag_indices_bind: str | None = None,
        report: bool = False,
        monitor_exclude_terms: list[str] | tuple[str, ...] = (),
        wandb_3d_log: Callable[
            [torch.Tensor, torch.Tensor, Sampler, dict[str, object], ManagerBasedRLEnv, list[str]], None
        ]
        | None = None,
        wandb_3d_log_period: int = 100,
    ):
        """Args (additional 3D-vis params):

        wandb_3d_log: Optional caller-owned logging hook. Receives
            ``(state_data, success_rates, sampler, log_state, env,
            reset_assets)``. ``None`` (default) skips upload.
        wandb_3d_log_period: Number of ``__call__`` invocations between
            hook calls. Ignored when :paramref:`wandb_3d_log` is ``None``.
        """
        if reset_assets and list(reset_assets) != self._requested_reset_assets:
            raise ValueError(
                "reset_accumulator reset_assets changed after initialization. "
                f"Expected {self._requested_reset_assets}, got {list(reset_assets)}."
            )
        if self.precollecting_phase:
            if self._state_tag_names_bind is not None:
                self.state_tag_names = eval(self._state_tag_names_bind)  # noqa: S307
            all_env_ids = torch.arange(env.num_envs, device=env.device)
            precollect_capacity = self.state_data.shape[0]
            target_size = min(self._state_target_size, precollect_capacity)
            state_size = 0
            pbar = tqdm(total=precollect_capacity, desc="reset_accumulator")
            while state_size < precollect_capacity:
                prev_size = state_size
                reset_term.func(env, all_env_ids, **reset_term.params)
                valid_mask = torch.ones(len(all_env_ids), dtype=torch.bool, device=env.device)
                for condition in self.acceptance_conditions.values():
                    condition_func = condition if callable(condition) else condition.func
                    valid_mask &= condition_func(env, all_env_ids)

                valid_env_ids = all_env_ids[valid_mask]
                if valid_env_ids.numel() > 0:
                    states = reset_state.get_reset_state(self._env, valid_env_ids, self.reset_assets, is_relative=True)
                    tag_indices = None
                    if self._tag_indices_bind is not None:
                        all_tag_indices = eval(self._tag_indices_bind)  # noqa: S307
                        tag_indices = all_tag_indices[valid_mask].long()

                    count = min(int(states.shape[0]), precollect_capacity - state_size)
                    end = state_size + count
                    self.state_data[state_size:end] = states[:count]
                    if tag_indices is not None:
                        self.state_tag_indices[state_size:end] = tag_indices[:count]
                    state_size = end
                    if end >= precollect_capacity and target_size < precollect_capacity:
                        features = extract_features(self.state_data[:end], self._state_fps_features)
                        keep_indices = grid_bucket_downsample(features, target_size).sort().values
                        self.state_data = self.state_data[keep_indices].contiguous()
                        self.state_tag_indices = self.state_tag_indices[keep_indices].contiguous()
                        pbar.update(precollect_capacity - prev_size)
                        break

                pbar.update(state_size - prev_size)
            pbar.close()
            self.precollecting_phase = False
            if self.state_tag_names is not None:
                tags = self.state_tag_indices[: self.state_data.shape[0]]
                counts = torch.bincount(tags[tags >= 0], minlength=len(self.state_tag_names))
                print(
                    "[reset_accumulator] state table filled per category: "
                    + ", ".join(f"{name}={int(counts[i])}" for i, name in enumerate(self.state_tag_names))
                    + f" (total={int(counts.sum())})"
                )

        if self.success_monitor is None:
            n_slots = self.state_data.shape[0]
            monitor_cfg = self._success_monitor_cfg
            # one partition: the bank draws across every slot, and the samplers below do the
            # drawing themselves. ``success_rate`` is updated in place, so aliasing it here keeps
            # the strategies' ``success_rates`` binding live.
            self.success_monitor = monitor_cfg.class_type(monitor_cfg, 1, n_slots, env.device)
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
            self.success_monitor.success_update(self.sampled_slots[monitor_ids], progress.is_success[monitor_ids])

        log: dict[str, float] = {}
        if report:
            n_slots = self.state_data.shape[0]
            if self.state_tag_names:
                tags = self.state_tag_indices[:n_slots]
                names = self.state_tag_names
                monitor_rates = self.monitor_success_rate[:n_slots]
                log["Metrics/MonitorSuccessRate"] = monitor_rates.mean().item()
                for i, name in enumerate(names):
                    mask = tags == i
                    log[f"Metrics/MonitorSuccessRate/{name}"] = monitor_rates[mask].mean().item() if mask.any() else 0.0

        if env_ids.numel() > 0:
            if self._sampler is None:
                coords = extract_features(self.state_data, self._state_fps_features)
                n = int(coords.shape[0])
                layout = StateLayout(
                    coords=coords,
                    spawn_index=torch.arange(n, device=coords.device, dtype=torch.long),
                    target_index=None,
                )
                self._sampler = self._sampling_cfg.class_type(
                    self._sampling_cfg, layout, env=env, success_rates=self.monitor_success_rate
                )

            num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
            probs, slot_idx = self._sampler.probabilities_and_sample(int(num_samples))
            slot_idx = slot_idx[: len(env_ids)]
            self.sampled_slots[env_ids] = slot_idx
            reset_state.set_reset_state(env, self.state_data[slot_idx], env_ids, self.reset_assets, is_relative=True)
            if report and self.state_tag_names:
                tags = self.state_tag_indices[: self.state_data.shape[0]]
                for i, name in enumerate(self.state_tag_names):
                    log[f"Metrics/SampleProb/{name}"] = probs[tags == i].sum().item()

        if report:
            env.extras.setdefault("log", {}).update(log)

        if wandb_3d_log is not None and self._sampler is not None and not self.precollecting_phase:
            self._wandb_3d_counter = getattr(self, "_wandb_3d_counter", 0) + 1
            if self._wandb_3d_counter % wandb_3d_log_period == 0:
                wandb_3d_log(
                    self.state_data,
                    self.monitor_success_rate,
                    self._sampler,
                    self._wandb_3d_log_state,
                    env,
                    self.reset_assets,
                )

    def apply_sampled_slots(self, env_ids: torch.Tensor) -> None:
        """Realize each env's currently-assigned table slot state in the sim.

        The caller writes the desired slot index into :attr:`sampled_slots` for
        ``env_ids`` and then calls this so the chosen stored state is written
        back to the assets.
        """
        reset_state.set_reset_state(
            self._env, self.state_data[self.sampled_slots[env_ids]], env_ids, self.reset_assets, is_relative=True
        )


class TermChoice(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_partitions: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        self.num_partitions = len(self.term_partitions)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs

        # Need a monitor when the sampler has any non-uniform strategy, or when the user
        # explicitly requested reporting. Decided from the strategy configs because the monitor
        # owns the rate tensor the sampler binds, so it has to exist first.
        needs_rates = cfg.params.get("report", False) or any(
            "Uniform" not in str(strategy.class_type) for strategy in self._sampling_cfg.strategies
        )
        if needs_rates:
            monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
            self.success_monitor = monitor_cfg.class_type(monitor_cfg, 1, self.num_partitions, env.device)
            self.term_success_rate = self.success_monitor.success_rate
        else:
            self.success_monitor = None
            self.term_success_rate = torch.zeros(self.num_partitions, device=env.device)

        layout = StateLayout(
            coords=torch.zeros(self.num_partitions, 1, device=env.device),
            spawn_index=torch.arange(self.num_partitions, device=env.device, dtype=torch.long),
            target_index=None,
        )
        self._sampler = self._sampling_cfg.class_type(
            self._sampling_cfg, layout, env=env, success_rates=self.term_success_rate
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, ManagerTermBase],
        sampling: SamplerCfg,
        success_monitor_cfg: SuccessMonitorCfg,
        report: bool = False,
    ) -> None:
        if self.num_partitions == 0:
            return
        if report:
            log = {
                f"Metrics/SuccessRate/{name}": self.term_success_rate[i].item()
                for i, name in enumerate(self.term_partitions.keys())
            }
            log["Metrics/SuccessRate"] = self.term_success_rate.mean().item()
        if self.success_monitor:
            success = env.termination_manager.get_term_cfg("progress_context").func.is_success
            self.success_monitor.success_update(self.term_samples[env_ids], success[env_ids])

        num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
        probs, choices = self._sampler.probabilities_and_sample(int(num_samples))
        self.term_samples[env_ids] = choices[: len(env_ids)]
        if report:
            log.update(
                {f"Metrics/SampleProb/{name}": probs[i].item() for i, name in enumerate(self.term_partitions.keys())}
            )

        for i, (_, term_cfg) in enumerate(self.term_partitions.items()):
            term_ids = env_ids[self.term_samples[env_ids] == i]
            if term_ids.numel() > 0:
                term_cfg.func(env, term_ids, **term_cfg.params)

        if report:
            env.extras.setdefault("log", {}).update(log)


class ChainedResetTerms(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terms: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, callable],
        probability: float = 1.0,
    ) -> None:
        keep = torch.rand(env_ids.size(0), device=env_ids.device) < probability
        if not keep.any():
            return
        env_ids_to_reset = env_ids[keep]
        for _, term in terms.items():
            term.func(env, env_ids_to_reset, **term.params)  # type: ignore
