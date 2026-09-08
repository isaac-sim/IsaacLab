# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-rate monitoring shared by task reset strategies."""

from __future__ import annotations

import torch

from isaaclab.utils.configclass import configclass


@configclass
class SuccessMonitorCfg:
    """Configuration for :class:`SuccessMonitor`."""

    class_type: type[SuccessMonitor] | str = "{DIR}.success_monitor:SuccessMonitor"
    """Monitor implementation, resolved when the environment starts."""

    monitored_history_len: int = 10
    """Episodes remembered per slot."""

    target_success_rate: float = 0.5
    """Success rate favored by sampling, in ``[0, 1]``."""

    kappa: float = 1.0
    """Concentration around :attr:`target_success_rate`; zero is uniform."""

    temperature: float = 1.0
    """Sampling-weight temperature, at or above ``1.0``."""


class SuccessMonitor:
    """Track recent outcomes per slot and sample within partitioned slot banks."""

    def __init__(self, cfg: SuccessMonitorCfg, num_partitions: int, partition_size: int, device: str):
        self.cfg = cfg
        self.num_partitions = num_partitions
        self.partition_size = partition_size
        self.device = device

        num_slots = num_partitions * partition_size
        self.success_buf = torch.zeros((num_slots, cfg.monitored_history_len), device=device)
        self.success_rate = torch.zeros(num_slots, device=device)
        self.success_pointer = torch.zeros(num_slots, device=device, dtype=torch.long)
        self.success_size = torch.zeros(num_slots, device=device, dtype=torch.long)

    def get_success_rate(self) -> torch.Tensor:
        """Return a copy of every slot's measured success rate."""
        return self.success_rate.clone()

    def get_mean_success_rate(self) -> float:
        """Average rates across slots that have recorded outcomes."""
        measured = self.success_size > 0
        return float(self.success_rate[measured].mean()) if bool(measured.any()) else 0.0

    def success_update(self, slot_ids: torch.Tensor, success: torch.Tensor):
        """Append outcomes to their slots' ring buffers and update success rates."""
        if len(slot_ids) == 0:
            return
        history = self.cfg.monitored_history_len
        order = torch.argsort(slot_ids, stable=True)
        ordered_slots = slot_ids[order]
        unique_slots, counts = torch.unique_consecutive(ordered_slots, return_counts=True)

        starts = counts.cumsum(0) - counts
        offset = torch.arange(len(ordered_slots), device=self.device) - starts.repeat_interleave(counts)
        offset -= (counts - history).clamp(min=0).repeat_interleave(counts)
        kept = offset >= 0

        slots = ordered_slots[kept]
        positions = (self.success_pointer[slots] + offset[kept]) % history
        self.success_buf[slots, positions] = success[order][kept].to(dtype=self.success_buf.dtype)

        written = counts.clamp(max=history)
        self.success_pointer[unique_slots] = (self.success_pointer[unique_slots] + written) % history
        self.success_size[unique_slots] = (self.success_size[unique_slots] + written).clamp(max=history)
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)

    def target_weights(self) -> torch.Tensor:
        """Return unnormalized slot weights peaking at the target success rate."""
        target = min(max(self.cfg.target_success_rate, 0.0), 1.0)
        kappa = max(self.cfg.kappa, 0.0)
        a = 1.0 + kappa * target
        b = 1.0 + kappa * (1.0 - target)
        eps = 1e-4
        rate = self.success_rate
        weights = ((rate + eps).pow(a - 1.0) * (1.0 - rate + eps).pow(b - 1.0)).clamp_min(eps)
        return weights.pow(1.0 / max(self.cfg.temperature, 1.0))

    def sample_by_target_rate(self, partition_ids: torch.Tensor) -> torch.Tensor:
        """Draw one slot from each requested partition."""
        weights = self.target_weights().view(self.num_partitions, self.partition_size)
        slots = torch.multinomial(weights[partition_ids], 1).view(-1)
        return partition_ids * self.partition_size + slots
