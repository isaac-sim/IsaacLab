# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime sampling strategies.

Each strategy binds its own input signals via configclass ``*_bind`` fields,
resolved at construction time against the ``bind_ns`` kwargs forwarded from
:class:`Sampler`. Strategies that need no inputs simply discard their args.

:meth:`SamplingStrategy.score` writes a per-item non-negative score tensor into
the pre-allocated ``out`` buffer; the sampler sums weighted scores and
normalizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from ..state_layout import StateLayout

if TYPE_CHECKING:
    from .sampling_strategies_cfg import (
        BetaSamplingStrategyCfg,
        FrontierSamplingStrategyCfg,
        UniformSamplingStrategyCfg,
    )


class SamplingStrategy(Protocol):
    """Per-item scorer for sampler probability construction.

    Strategies declare their runtime inputs as ``*_bind`` cfg fields (string
    expressions); the sampler resolves them against a shared ``bind_ns`` dict
    at strategy ctor time. :meth:`score` then writes non-negative scores into
    ``out`` using the bound references.
    """

    name: str
    """Short identifier used in diagnostic / log keys."""

    def score(self, out: torch.Tensor) -> None:
        """Write ``[num_items]`` non-negative unnormalized scores into ``out``."""
        ...


class BetaSamplingStrategy:
    """Per-item Beta-kernel score peaked at a target success rate."""

    name = "beta"

    def __init__(self, cfg: BetaSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        del layout
        target = max(0.0, min(1.0, float(cfg.target)))
        kappa = max(0.0, float(cfg.kappa))
        self._a = 1.0 + kappa * target
        self._b = 1.0 + kappa * (1.0 - target)
        self._rates: torch.Tensor = eval(cfg.success_rate_bind, bind_ns)  # noqa: S307

    def score(self, out: torch.Tensor) -> None:
        out.copy_(self._rates)
        out.pow_(self._a - 1.0)
        out.mul_((1.0 - self._rates).pow(self._b - 1.0))


class FrontierSamplingStrategy:
    """Per-task frontier score from a kNN graph over the task feature space."""

    name = "frontier"

    def __init__(self, cfg: FrontierSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        self._dilation_steps = max(1, int(cfg.dilation_steps))
        self._rates: torch.Tensor = eval(cfg.success_rate_bind, bind_ns)  # noqa: S307
        spawn_feat = layout.coords[layout.spawn_index]
        if layout.target_index is None:
            task_features = spawn_feat
        else:
            target_feat = layout.coords[layout.target_index]
            task_features = torch.cat([spawn_feat, target_feat], dim=-1)

        from scipy.spatial import cKDTree

        k = int(cfg.k)
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        n = int(task_features.shape[0])
        device = task_features.device
        self_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(-1)
        self._knn = self_idx.expand(n, k).clone()

        partition_keys = [None] if layout.task_partition is None else torch.unique(layout.task_partition).tolist()
        for partition_key in partition_keys:
            if partition_key is None:
                member_idx = torch.arange(n, device=device, dtype=torch.long)
                features = task_features
            else:
                member_idx = (layout.task_partition == partition_key).nonzero(as_tuple=False).squeeze(-1)
                features = task_features[member_idx]

            n_member = int(features.shape[0])
            if n_member <= 1:
                continue

            k_eff = min(k, n_member - 1)
            features_np = features.detach().cpu().numpy()
            _, idx = cKDTree(features_np).query(features_np, k=k_eff + 1)
            local_knn = torch.as_tensor(idx[:, 1 : k_eff + 1], device=device, dtype=torch.long)
            if k_eff < k:
                pad = torch.arange(n_member, device=device, dtype=torch.long).unsqueeze(-1).expand(n_member, k - k_eff)
                local_knn = torch.cat([local_knn, pad], dim=1)
            self._knn[member_idx] = member_idx[local_knn]

    def score(self, out: torch.Tensor) -> None:
        rates = self._rates
        s_dil = rates
        for _ in range(self._dilation_steps):
            neighbor_max = s_dil[self._knn].amax(dim=-1)
            s_dil = torch.maximum(s_dil, neighbor_max)
        out.copy_(s_dil)
        out.sub_(rates).clamp_min_(0.0)
        out.mul_(1.0 - rates)


class UniformSamplingStrategy:
    """Constant 1.0 per item -- the trivial baseline / floor."""

    name = "uniform"

    def __init__(self, cfg: UniformSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        del cfg, layout, bind_ns

    def score(self, out: torch.Tensor) -> None:
        out.fill_(1.0)
