# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
import warp.utils as wpu

from ..sampling_strategies import BetaSamplingStrategy, FrontierSamplingStrategy

if TYPE_CHECKING:
    from ...state_layout import StateLayout
    from ..sampler_cfg import SamplerCfg


_STRATEGY_UNIFORM = 0
_STRATEGY_BETA = 1
_STRATEGY_FRONTIER = 2


def _task_features(layout: StateLayout) -> torch.Tensor:
    """Return task feature rows used for frontier locality ordering."""
    spawn_feat = layout.coords[layout.spawn_index]
    if layout.target_index is None:
        return spawn_feat
    target_feat = layout.coords[layout.target_index]
    return torch.cat([spawn_feat, target_feat], dim=-1)


def _spatial_order(features: torch.Tensor, partition: torch.Tensor | None) -> torch.Tensor:
    """Sort item ids by a coarse spatial key, optionally grouped by partition."""
    n = int(features.shape[0])
    if n == 0:
        return torch.empty(0, device=features.device, dtype=torch.long)

    features = features.detach().to(dtype=torch.float32)
    dim = int(features.shape[1])
    bits = max(1, min(10, 60 // max(dim, 1)))
    bins = 1 << bits
    lo = features.amin(dim=0)
    span = (features.amax(dim=0) - lo).clamp_min(1.0e-12)
    q = (((features - lo) / span) * float(bins - 1)).clamp_(0, bins - 1).to(torch.int64)

    key = torch.zeros(n, device=features.device, dtype=torch.int64)
    for d in range(dim):
        key = key * bins + q[:, d]
    order = torch.argsort(key, stable=True)
    if partition is not None:
        partition = partition.to(device=features.device, dtype=torch.long)
        order = order[torch.argsort(partition[order], stable=True)]
    return order


@wp.kernel
def _frontier_init_kernel(
    success_rates: wp.array(dtype=wp.float32),
    frontier_order: wp.array(dtype=wp.int32),
    frontier_prev: wp.array2d(dtype=wp.float32),
):
    f, i = wp.tid()
    frontier_prev[f, i] = success_rates[int(frontier_order[i])]


@wp.kernel
def _frontier_dilate_kernel(
    frontier_prev: wp.array2d(dtype=wp.float32),
    frontier_next: wp.array2d(dtype=wp.float32),
    frontier_results: wp.array2d(dtype=wp.float32),
    frontier_order: wp.array(dtype=wp.int32),
    frontier_result_for_step: wp.array2d(dtype=wp.int32),
    frontier_knn: wp.array3d(dtype=wp.int32),
    frontier_k: wp.array(dtype=wp.int32),
    frontier_group_max_dilation_steps: wp.array(dtype=wp.int32),
    step: int,
    max_k: int,
):
    g, i = wp.tid()
    if step >= int(frontier_group_max_dilation_steps[g]):
        return

    v = frontier_prev[g, i]
    k = int(frontier_k[g])
    for j in range(max_k):
        if j < k:
            neighbor = int(frontier_knn[g, i, j])
            v = wp.max(v, frontier_prev[g, neighbor])
    frontier_next[g, i] = v

    result_row = int(frontier_result_for_step[g, step])
    if result_row >= 0:
        frontier_results[result_row, int(frontier_order[i])] = v


@wp.kernel
def _sampler_score_kernel(
    success_rates: wp.array(dtype=wp.float32),
    score_rows: wp.array2d(dtype=wp.float32),
    strategy_kind: wp.array(dtype=wp.int32),
    beta_a: wp.array(dtype=wp.float32),
    beta_b: wp.array(dtype=wp.float32),
    frontier_ids: wp.array(dtype=wp.int32),
    frontier_dilated: wp.array2d(dtype=wp.float32),
):
    s, i = wp.tid()
    rate = success_rates[i]
    kind = int(strategy_kind[s])
    if kind == _STRATEGY_UNIFORM:
        score_rows[s, i] = 1.0
    elif kind == _STRATEGY_BETA:
        score_rows[s, i] = wp.pow(rate, beta_a[s] - 1.0) * wp.pow(1.0 - rate, beta_b[s] - 1.0)
    else:
        f = int(frontier_ids[s])
        delta = frontier_dilated[f, i] - rate
        if delta < 0.0:
            delta = 0.0
        score_rows[s, i] = (1.0 - rate) * delta


@wp.kernel
def _sampler_weight_kernel(
    success_rates: wp.array(dtype=wp.float32),
    weighted: wp.array(dtype=wp.float32),
    strategy_kind: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    beta_a: wp.array(dtype=wp.float32),
    beta_b: wp.array(dtype=wp.float32),
    frontier_ids: wp.array(dtype=wp.int32),
    frontier_dilated: wp.array2d(dtype=wp.float32),
    eps: float,
    num_strategies: int,
):
    i = wp.tid()
    rate = success_rates[i]
    w = eps
    for s in range(num_strategies):
        kind = int(strategy_kind[s])
        score = float(1.0)
        if kind == _STRATEGY_BETA:
            score = wp.pow(rate, beta_a[s] - 1.0) * wp.pow(1.0 - rate, beta_b[s] - 1.0)
        elif kind == _STRATEGY_FRONTIER:
            f = int(frontier_ids[s])
            delta = frontier_dilated[f, i] - rate
            if delta < 0.0:
                delta = 0.0
            score = (1.0 - rate) * delta
        weight = weights[s]
        if weight > 0.0:
            w += weight * score
    weighted[i] = w


@wp.kernel
def _sampler_normalize_kernel(
    weighted: wp.array(dtype=wp.float32),
    total: wp.array(dtype=wp.float32),
    probs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    probs[i] = weighted[i] / total[0]


@wp.kernel
def _sample_counter_kernel(counter: wp.array(dtype=wp.int64), base: wp.array(dtype=wp.int64), num_samples: int):
    base[0] = counter[0]
    counter[0] = counter[0] + wp.int64(num_samples)


@wp.kernel
def _sample_cdf_kernel(
    cdf: wp.array(dtype=wp.float32),
    samples: wp.array(dtype=wp.int64),
    base: wp.array(dtype=wp.int64),
    seed: int,
    num_items: int,
):
    i = wp.tid()
    rng = wp.rand_init(seed, int(base[0]) + i)
    u = wp.randf(rng)

    lo = int(0)
    hi = int(num_items - 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if u <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    samples[i] = wp.int64(lo)


class SamplerWarp:
    """Warp backend for weighted sampling strategies.

    Restricted to success-rate-driven strategies (Beta / Frontier / Uniform).
    Strategies with non-success-rate signals are rejected at construction time;
    route those through :class:`SamplerTorch` instead.
    """

    def __init__(self, cfg: SamplerCfg, layout: StateLayout, **bind_ns) -> None:
        from ..sampling_strategies_cfg import (
            BetaSamplingStrategyCfg,
            FrontierSamplingStrategyCfg,
            UniformSamplingStrategyCfg,
        )

        supported_cfg_types = (BetaSamplingStrategyCfg, FrontierSamplingStrategyCfg, UniformSamplingStrategyCfg)
        for strategy_cfg in cfg.strategies:
            if not isinstance(strategy_cfg, supported_cfg_types):
                raise NotImplementedError(
                    f"SamplerWarp does not implement strategy {type(strategy_cfg).__name__};"
                    " use the Torch backend (set ``warp=False`` on SamplerCfg)."
                )

        wp.init()
        self.eps = float(cfg.eps)
        self.seed = int(cfg.seed)
        self.names: list[str] = []
        self._plot_strategy_indices = [i for i, strategy_cfg in enumerate(cfg.strategies) if strategy_cfg.plot]

        # Bind the success-rate tensor once; all Beta / Frontier / Uniform
        # strategies share it. Resolved against the caller's ``bind_ns``.
        success_rate_binds = {
            strategy_cfg.success_rate_bind
            for strategy_cfg in cfg.strategies
            if isinstance(strategy_cfg, (BetaSamplingStrategyCfg, FrontierSamplingStrategyCfg))
        }
        if len(success_rate_binds) > 1:
            raise ValueError(
                f"SamplerWarp requires a single shared success_rate_bind; got {sorted(success_rate_binds)}."
            )
        if success_rate_binds:
            (bind_expr,) = success_rate_binds
            self._success_rates: torch.Tensor = eval(bind_expr, bind_ns)  # noqa: S307
        else:
            # Only Uniform strategies; allocate a zero placeholder of the right shape.
            self._success_rates = torch.zeros(int(layout.spawn_index.shape[0]), device=layout.coords.device)

        kinds: list[int] = []
        weights: list[float] = []
        beta_a: list[float] = []
        beta_b: list[float] = []
        frontier_ids: list[int] = []
        frontier_group_by_k: dict[int, int] = {}
        frontier_group_knn: list[torch.Tensor] = []
        frontier_group_k: list[int] = []
        frontier_group_max_dilation_steps: list[int] = []
        frontier_result_by_group_step: dict[tuple[int, int], int] = {}

        device = layout.coords.device
        self._num_items = int(layout.spawn_index.shape[0])
        frontier_order = _spatial_order(_task_features(layout), layout.task_partition)
        frontier_inverse = torch.empty_like(frontier_order)
        frontier_inverse[frontier_order] = torch.arange(self._num_items, device=device, dtype=torch.long)

        for strategy_cfg in cfg.strategies:
            weights.append(float(strategy_cfg.weight))
            if isinstance(strategy_cfg, BetaSamplingStrategyCfg):
                strategy = BetaSamplingStrategy(strategy_cfg, layout, **bind_ns)
                self.names.append(strategy.name)
                kinds.append(_STRATEGY_BETA)
                beta_a.append(float(strategy._a))
                beta_b.append(float(strategy._b))
                frontier_ids.append(-1)
            elif isinstance(strategy_cfg, FrontierSamplingStrategyCfg):
                strategy = FrontierSamplingStrategy(strategy_cfg, layout, **bind_ns)
                k = int(strategy_cfg.k)
                dilation_steps = int(strategy._dilation_steps)
                group_id = frontier_group_by_k.get(k)
                if group_id is None:
                    group_id = len(frontier_group_knn)
                    frontier_group_by_k[k] = group_id
                    frontier_group_knn.append(strategy._knn.to(dtype=torch.int64).contiguous())
                    frontier_group_k.append(k)
                    frontier_group_max_dilation_steps.append(dilation_steps)
                else:
                    frontier_group_max_dilation_steps[group_id] = max(
                        frontier_group_max_dilation_steps[group_id], dilation_steps
                    )

                self.names.append(strategy.name)
                kinds.append(_STRATEGY_FRONTIER)
                beta_a.append(1.0)
                beta_b.append(1.0)
                result_key = (group_id, dilation_steps)
                result_id = frontier_result_by_group_step.get(result_key)
                if result_id is None:
                    result_id = len(frontier_result_by_group_step)
                    frontier_result_by_group_step[result_key] = result_id
                frontier_ids.append(result_id)
            elif isinstance(strategy_cfg, UniformSamplingStrategyCfg):
                strategy = strategy_cfg.class_type(strategy_cfg, layout, **bind_ns)
                self.names.append(strategy.name)
                kinds.append(_STRATEGY_UNIFORM)
                beta_a.append(1.0)
                beta_b.append(1.0)
                frontier_ids.append(-1)
            else:
                raise TypeError(f"Unsupported Warp sampler strategy cfg: {type(strategy_cfg).__name__}")

        self._num_strategies = len(kinds)
        self._num_frontier_groups = len(frontier_group_knn)
        self._num_frontier_results = len(frontier_result_by_group_step)
        self._max_k = max(frontier_group_k) if frontier_group_k else 1
        self._max_dilation_steps = max(frontier_group_max_dilation_steps) if frontier_group_max_dilation_steps else 0

        self._strategy_kind = torch.tensor(kinds, device=device, dtype=torch.int32)
        self._weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self._beta_a = torch.tensor(beta_a, device=device, dtype=torch.float32)
        self._beta_b = torch.tensor(beta_b, device=device, dtype=torch.float32)
        self._frontier_ids = torch.tensor(frontier_ids, device=device, dtype=torch.int32)
        self._frontier_k = torch.tensor(frontier_group_k or [1], device=device, dtype=torch.int32)
        self._frontier_group_max_dilation_steps = torch.tensor(
            frontier_group_max_dilation_steps or [0], device=device, dtype=torch.int32
        )
        self._frontier_order = frontier_order.to(dtype=torch.int32)
        frontier_result_for_step = torch.full(
            (max(self._num_frontier_groups, 1), max(self._max_dilation_steps, 1)),
            -1,
            device=device,
            dtype=torch.int32,
        )
        for (group_id, dilation_steps), result_id in frontier_result_by_group_step.items():
            frontier_result_for_step[group_id, dilation_steps - 1] = result_id
        self._frontier_result_for_step = frontier_result_for_step

        if frontier_group_knn:
            knn = torch.empty(
                (self._num_frontier_groups, self._num_items, self._max_k),
                device=device,
                dtype=torch.int32,
            )
            for i, indices in enumerate(frontier_group_knn):
                k = int(indices.shape[1])
                internal_knn = frontier_inverse[indices[frontier_order].to(dtype=torch.long)].to(dtype=torch.int32)
                knn[i, :, :k] = internal_knn
                if k < self._max_k:
                    self_idx = torch.arange(self._num_items, device=device, dtype=torch.int32).unsqueeze(-1)
                    knn[i, :, k:] = self_idx.expand(self._num_items, self._max_k - k)
        else:
            knn = torch.zeros((1, self._num_items, 1), device=device, dtype=torch.int32)
        self._frontier_knn = knn

        self._score_rows = torch.empty((self._num_strategies, self._num_items), device=device, dtype=torch.float32)
        self._weighted = torch.empty(self._num_items, device=device, dtype=torch.float32)
        self._probs = torch.empty(self._num_items, device=device, dtype=torch.float32)
        self._cdf = torch.empty(self._num_items, device=device, dtype=torch.float32)
        self._sum = torch.empty(1, device=device, dtype=torch.float32)
        self._frontier_prev = torch.empty(
            (max(self._num_frontier_groups, 1), self._num_items),
            device=device,
            dtype=torch.float32,
        )
        self._frontier_next = torch.empty_like(self._frontier_prev)
        self._frontier_results = torch.empty(
            (max(self._num_frontier_results, 1), self._num_items),
            device=device,
            dtype=torch.float32,
        )
        max_samples = int(cfg.max_samples) if cfg.max_samples is not None else 1
        self._samples = torch.empty(max_samples, device=device, dtype=torch.int64)
        self._sample_counter = torch.zeros(1, device=device, dtype=torch.int64)
        self._sample_base = torch.zeros(1, device=device, dtype=torch.int64)
        self._graph = None
        self._graph_key: tuple[int, int] | None = None

        self._wp_success_rates = wp.from_torch(self._success_rates, dtype=wp.float32)
        self._wp_strategy_kind = wp.from_torch(self._strategy_kind, dtype=wp.int32)
        self._wp_weights = wp.from_torch(self._weights, dtype=wp.float32)
        self._wp_beta_a = wp.from_torch(self._beta_a, dtype=wp.float32)
        self._wp_beta_b = wp.from_torch(self._beta_b, dtype=wp.float32)
        self._wp_frontier_ids = wp.from_torch(self._frontier_ids, dtype=wp.int32)
        self._wp_frontier_k = wp.from_torch(self._frontier_k, dtype=wp.int32)
        self._wp_frontier_group_max_dilation_steps = wp.from_torch(
            self._frontier_group_max_dilation_steps, dtype=wp.int32
        )
        self._wp_frontier_order = wp.from_torch(self._frontier_order, dtype=wp.int32)
        self._wp_frontier_result_for_step = wp.from_torch(self._frontier_result_for_step, dtype=wp.int32)
        self._wp_frontier_knn = wp.from_torch(self._frontier_knn, dtype=wp.int32)
        self._wp_score_rows = wp.from_torch(self._score_rows, dtype=wp.float32)
        self._wp_weighted = wp.from_torch(self._weighted, dtype=wp.float32)
        self._wp_probs = wp.from_torch(self._probs, dtype=wp.float32)
        self._wp_cdf = wp.from_torch(self._cdf, dtype=wp.float32)
        self._wp_sum = wp.from_torch(self._sum, dtype=wp.float32)
        self._wp_frontier_prev = wp.from_torch(self._frontier_prev, dtype=wp.float32)
        self._wp_frontier_next = wp.from_torch(self._frontier_next, dtype=wp.float32)
        self._wp_frontier_results = wp.from_torch(self._frontier_results, dtype=wp.float32)
        self._wp_sample_counter = wp.from_torch(self._sample_counter, dtype=wp.int64)
        self._wp_sample_base = wp.from_torch(self._sample_base, dtype=wp.int64)
        self._wp_samples = wp.from_torch(self._samples, dtype=wp.int64)

    def scores(self) -> torch.Tensor:
        """Return contiguous per-strategy score rows shaped ``[num_strategies, num_items]``."""
        self._update_frontier()
        wp.launch(
            _sampler_score_kernel,
            dim=(self._num_strategies, self._num_items),
            inputs=[
                self._wp_success_rates,
                self._wp_score_rows,
                self._wp_strategy_kind,
                self._wp_beta_a,
                self._wp_beta_b,
                self._wp_frontier_ids,
                self._wp_frontier_results,
            ],
            device=str(self._success_rates.device),
        )
        return self._score_rows

    def _update_frontier(self) -> None:
        """Update frontier result rows for the current success rates."""
        frontier_prev = self._wp_frontier_prev
        frontier_next = self._wp_frontier_next
        if self._num_frontier_groups > 0:
            wp.launch(
                _frontier_init_kernel,
                dim=(self._num_frontier_groups, self._num_items),
                inputs=[self._wp_success_rates, self._wp_frontier_order, frontier_prev],
                device=str(self._success_rates.device),
            )
            for step in range(self._max_dilation_steps):
                wp.launch(
                    _frontier_dilate_kernel,
                    dim=(self._num_frontier_groups, self._num_items),
                    inputs=[
                        frontier_prev,
                        frontier_next,
                        self._wp_frontier_results,
                        self._wp_frontier_order,
                        self._wp_frontier_result_for_step,
                        self._wp_frontier_knn,
                        self._wp_frontier_k,
                        self._wp_frontier_group_max_dilation_steps,
                        step,
                        self._max_k,
                    ],
                    device=str(self._success_rates.device),
                )
                frontier_prev, frontier_next = frontier_next, frontier_prev

    def probabilities(self) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        self._update_frontier()
        wp.launch(
            _sampler_weight_kernel,
            dim=self._num_items,
            inputs=[
                self._wp_success_rates,
                self._wp_weighted,
                self._wp_strategy_kind,
                self._wp_weights,
                self._wp_beta_a,
                self._wp_beta_b,
                self._wp_frontier_ids,
                self._wp_frontier_results,
                self.eps,
                self._num_strategies,
            ],
            device=str(self._success_rates.device),
        )
        wpu.array_sum(self._wp_weighted, out=self._wp_sum, value_count=self._num_items)
        wp.launch(
            _sampler_normalize_kernel,
            dim=self._num_items,
            inputs=[self._wp_weighted, self._wp_sum, self._wp_probs],
            device=str(self._success_rates.device),
        )
        return self._probs

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample item indices from ``probs``."""
        if num_samples > self._samples.shape[0]:
            raise ValueError(
                f"SamplerWarp.sample received {num_samples} samples, but max_samples={self._samples.shape[0]}."
            )
        wp_probs = wp.from_torch(probs, dtype=wp.float32)
        wpu.array_scan(wp_probs, self._wp_cdf, inclusive=True)
        wp.launch(
            _sample_counter_kernel,
            dim=1,
            inputs=[self._wp_sample_counter, self._wp_sample_base, num_samples],
            device=str(probs.device),
        )
        wp.launch(
            _sample_cdf_kernel,
            dim=num_samples,
            inputs=[self._wp_cdf, self._wp_samples, self._wp_sample_base, self.seed, self._num_items],
            device=str(probs.device),
        )
        return self._samples[:num_samples]

    def probabilities_and_sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities and sampled item indices."""
        if num_samples > self._samples.shape[0]:
            raise ValueError(
                f"SamplerWarp.probabilities_and_sample received {num_samples} samples, "
                f"but max_samples={self._samples.shape[0]}."
            )

        key = (self._success_rates.data_ptr(), int(num_samples))
        if self._graph is None or self._graph_key != key:
            with wp.ScopedCapture(device=str(self._success_rates.device)) as capture:
                probs = self.probabilities()
                self.sample(probs, num_samples)
            self._graph = capture.graph
            self._graph_key = key

        wp.capture_launch(self._graph)
        return self._probs, self._samples[:num_samples]
