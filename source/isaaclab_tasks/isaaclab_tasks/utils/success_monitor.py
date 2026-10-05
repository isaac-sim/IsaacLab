# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-rate monitoring shared by task reset strategies."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass


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
        # a scratch row absorbs masked and overflowing outcomes, avoiding sync-inducing boolean indexing
        self._outcome_buf = torch.zeros((num_slots + 1, cfg.monitored_history_len), device=device)
        self.success_buf = self._outcome_buf[:num_slots]
        self.success_rate = torch.zeros(num_slots, device=device)
        self.success_pointer = torch.zeros(num_slots, device=device, dtype=torch.long)
        self.success_size = torch.zeros(num_slots, device=device, dtype=torch.long)

    def get_success_rate(self) -> torch.Tensor:
        """Return a copy of every slot's measured success rate."""
        return self.success_rate.clone()

    def get_mean_success_rate(self) -> torch.Tensor:
        """Average rates across slots that have recorded outcomes, as a 0-d device tensor; zero before any outcome."""
        measured = self.success_size > 0
        return (self.success_rate * measured).sum() / measured.sum().clamp(min=1)

    def success_update(self, slot_ids: torch.Tensor, success: torch.Tensor, valid: torch.Tensor | None = None):
        """Append outcomes to their slots' ring buffers and update success rates.

        Outcomes for the same slot are appended in input order; when a slot receives more outcomes than
        :attr:`SuccessMonitorCfg.monitored_history_len`, only its latest ones are kept.

        Args:
            slot_ids: Slot of each outcome, shape (N,).
            success: Whether each outcome succeeded, shape (N,).
            valid: Which outcomes to record, shape (N,). Defaults to None, which records all of them.
        """
        if len(slot_ids) == 0:
            return
        history = self.cfg.monitored_history_len
        scratch = self.success_rate.shape[0]
        if valid is not None:
            slot_ids = torch.where(valid, slot_ids, scratch)
        counts = torch.zeros(scratch + 1, dtype=torch.long, device=self.device)
        counts.index_add_(0, slot_ids, torch.ones(len(slot_ids), dtype=torch.long, device=self.device))

        # rank each outcome within its slot, in input order, then keep the latest ``history`` of them
        order = torch.argsort(slot_ids, stable=True)
        ordered_slots = slot_ids[order]
        starts = counts.cumsum(0) - counts
        offset = torch.arange(len(ordered_slots), device=self.device) - starts[ordered_slots]
        offset -= (counts[ordered_slots] - history).clamp(min=0)
        rows = torch.where((offset >= 0) & (ordered_slots < scratch), ordered_slots, scratch)
        positions = (self.success_pointer[rows.clamp(max=scratch - 1)] + offset) % history
        self._outcome_buf[rows, positions] = success[order].to(dtype=self._outcome_buf.dtype)

        written = counts[:scratch].clamp(max=history)
        self.success_pointer.add_(written).remainder_(history)
        self.success_size.add_(written).clamp_(max=history)
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

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return the rolling outcome history for checkpointing."""
        return {
            "success_history": self.success_buf.clone(),
            "history_pointer": self.success_pointer.clone(),
            "history_size": self.success_size.clone(),
        }

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore rolling outcome history from a checkpoint.

        Args:
            state: State previously returned by :meth:`get_state`.

        Raises:
            KeyError: If a required state tensor is missing.
            ValueError: If a tensor has an incompatible shape or contains an
                invalid ring-buffer pointer or size.
        """
        targets = {
            "success_history": self.success_buf,
            "history_pointer": self.success_pointer,
            "history_size": self.success_size,
        }
        for name, target in targets.items():
            if name not in state:
                raise KeyError(f"Success-monitor checkpoint is missing '{name}'.")
            if state[name].shape != target.shape:
                raise ValueError(
                    f"Success-monitor checkpoint '{name}' has shape {state[name].shape}; expected {target.shape}."
                )

        history_length = self.cfg.monitored_history_len
        if bool(torch.any((state["history_pointer"] < 0) | (state["history_pointer"] >= history_length))):
            raise ValueError("Success-monitor checkpoint contains an invalid history pointer.")
        if bool(torch.any((state["history_size"] < 0) | (state["history_size"] > history_length))):
            raise ValueError("Success-monitor checkpoint contains an invalid history size.")

        for name, target in targets.items():
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)
