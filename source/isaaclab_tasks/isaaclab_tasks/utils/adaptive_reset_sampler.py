# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive sampling over a difficulty-ordered reset-state cache."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from isaaclab.utils.configclass import configclass

__all__ = ["AdaptiveResetSampler", "AdaptiveResetSamplerCfg"]


@configclass
class AdaptiveResetSamplerCfg:
    """Configuration for :class:`AdaptiveResetSampler`.

    The sampler targets a mixture of successful and unsuccessful episodes while retaining uniform
    replay and a small amount of probing immediately beyond its monotonic difficulty frontier.
    """

    target_success_rate: float = 0.5
    """Desired predicted success rate of sampled resets."""

    temperature: float = 0.1
    """Minimum softmax temperature used to bound sampling concentration."""

    history_capacity: int = 32
    """Maximum effective recent outcome count retained for each reset row."""

    prior_strength: float = 4.0
    """Effective observation count assigned to the cold-start success prior."""

    initial_frontier_size: int = 128
    """Number of easiest rows initially included in the active frontier."""

    probe_size: int = 256
    """Number of rows immediately beyond the active frontier eligible for probing."""

    probe_fraction: float = 0.1
    """Sampling probability reserved for rows beyond the active frontier."""

    replay_fraction: float = 0.1
    """Uniform replay floor within the active frontier."""

    frontier_evidence: float = 2.0
    """Excess successful outcomes required to expose one additional reset row."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.validate_values()

    def validate_values(self) -> None:
        """Validate the adaptive sampling parameters after runtime overrides."""
        if not math.isfinite(float(self.target_success_rate)) or not 0.0 < self.target_success_rate < 1.0:
            raise ValueError("target_success_rate must lie strictly between zero and one.")
        for name in ("temperature", "prior_strength", "frontier_evidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("history_capacity", "initial_frontier_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.probe_size, int) or isinstance(self.probe_size, bool) or self.probe_size < 0:
            raise ValueError("probe_size must be a nonnegative integer.")
        if not math.isfinite(float(self.probe_fraction)) or not 0.0 <= self.probe_fraction < 1.0:
            raise ValueError("probe_fraction must lie in [0, 1).")
        if not math.isfinite(float(self.replay_fraction)) or not 0.0 <= self.replay_fraction < 1.0:
            raise ValueError("replay_fraction must lie in [0, 1).")


class AdaptiveResetSampler:
    """Sample reset rows near a requested success rate.

    Reset rows are identified by arbitrary non-negative integer IDs. The caller provides those IDs
    in easiest-to-hardest order and reports Boolean episode outcomes using the same IDs. No task or
    environment semantics are assumed.

    Outcome counts are kept in bounded, exponentially truncated buffers. Sampling combines a
    calibrated softmax over the active frontier, uniform replay, and uniform probes from the next
    rows in the difficulty ordering. The frontier can only advance.

    Args:
        difficulty_order: Unique raw reset-row IDs ordered from easiest to hardest.
        cfg: Sampler configuration.
    """

    _STATE_VERSION = 1
    _BISECTION_STEPS = 24
    _STATE_KEYS = (
        "version",
        "difficulty_order",
        "effective_successes",
        "effective_attempts",
        "total_successes",
        "total_attempts",
        "has_outcome",
        "latest_success",
        "frontier_size",
        "frontier_credit",
    )

    def __init__(self, difficulty_order: torch.Tensor, cfg: AdaptiveResetSamplerCfg | None = None):
        self.cfg = cfg if cfg is not None else AdaptiveResetSamplerCfg()
        self.cfg.validate_values()
        if difficulty_order.ndim != 1 or difficulty_order.numel() == 0:
            raise ValueError("difficulty_order must be a non-empty one-dimensional tensor.")
        if difficulty_order.dtype != torch.long:
            raise TypeError("difficulty_order must have dtype torch.long.")
        if bool(torch.any(difficulty_order < 0)):
            raise ValueError("difficulty_order must contain non-negative raw row IDs.")

        self._difficulty_order = difficulty_order.detach().clone()
        self._sorted_row_ids, self._sorted_to_rank = torch.sort(self._difficulty_order)
        if self._sorted_row_ids.numel() > 1 and bool(torch.any(self._sorted_row_ids[1:] == self._sorted_row_ids[:-1])):
            raise ValueError("difficulty_order must contain unique raw row IDs.")

        row_count = self._difficulty_order.numel()
        device = self._difficulty_order.device
        self._effective_successes = torch.zeros(row_count, device=device, dtype=torch.float32)
        self._effective_attempts = torch.zeros_like(self._effective_successes)
        self._total_successes = torch.zeros(row_count, device=device, dtype=torch.long)
        self._total_attempts = torch.zeros_like(self._total_successes)
        self._has_outcome = torch.zeros(row_count, device=device, dtype=torch.bool)
        self._latest_success = torch.zeros_like(self._has_outcome)

        self._frontier_size = min(self.cfg.initial_frontier_size, row_count)
        self._frontier_credit = torch.zeros((), device=device, dtype=torch.float32)

        rank = torch.arange(row_count, device=device, dtype=torch.float32)
        remaining = max(row_count - self._frontier_size, 1)
        hardness = (rank - self._frontier_size + 1.0).clamp_min(0.0) / remaining
        self._prior_success = self.cfg.target_success_rate * (1.0 - hardness).clamp_min(0.0)
        self._probabilities_dirty = True
        self._probabilities = torch.zeros(row_count, device=device, dtype=torch.float32)

    @property
    def difficulty_order(self) -> torch.Tensor:
        """Raw row IDs in easiest-to-hardest order."""
        return self._difficulty_order

    @property
    def frontier_size(self) -> int:
        """Number of rows currently exposed by the monotonic frontier."""
        return self._frontier_size

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        """Sampling probabilities aligned with :attr:`difficulty_order`."""
        if self._probabilities_dirty:
            self._probabilities = self._compute_probabilities()
            self._probabilities_dirty = False
        return self._probabilities

    @property
    def success_estimates(self) -> torch.Tensor:
        """Bayesian-smoothed success estimates aligned with :attr:`difficulty_order`."""
        return (self._effective_successes + self.cfg.prior_strength * self._prior_success) / (
            self._effective_attempts + self.cfg.prior_strength
        )

    def sample(
        self,
        count: int,
        forced_row_ids: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample raw reset-row IDs.

        Args:
            count: Number of row IDs to return.
            forced_row_ids: Optional row IDs with shape ``(count,)``. Non-negative entries are
                returned exactly, including rows outside the current frontier. ``-1`` entries are
                sampled adaptively.
            generator: Optional random-number generator passed to :func:`torch.multinomial`.

        Returns:
            Raw reset-row IDs with shape ``(count,)``.
        """
        if count < 0:
            raise ValueError("count cannot be negative.")
        if forced_row_ids is not None:
            if forced_row_ids.shape != (count,) or forced_row_ids.dtype != torch.long:
                raise ValueError("forced_row_ids must have shape (count,) and dtype torch.long.")
            if forced_row_ids.device != self._difficulty_order.device:
                raise ValueError("forced_row_ids must be on the sampler device.")
            if bool(torch.any(forced_row_ids < -1)):
                raise ValueError("forced_row_ids may contain only known raw row IDs or -1.")
            forced = forced_row_ids >= 0
            if bool(torch.any(forced)):
                self._row_ids_to_ranks(forced_row_ids[forced])
        else:
            forced = None

        if count == 0:
            return self._difficulty_order.new_empty(0)
        ranks = torch.multinomial(self.sampling_probabilities, count, replacement=True, generator=generator)
        row_ids = self._difficulty_order[ranks]
        if forced_row_ids is not None:
            row_ids = torch.where(forced, forced_row_ids, row_ids)
        return row_ids

    def record(self, row_ids: torch.Tensor, successes: torch.Tensor) -> None:
        """Record completed episode outcomes and advance the frontier.

        Args:
            row_ids: Raw reset-row IDs with shape ``(N,)``.
            successes: Boolean success outcomes with shape ``(N,)``.
        """
        if row_ids.ndim != 1 or row_ids.dtype != torch.long:
            raise ValueError("row_ids must be a one-dimensional torch.long tensor.")
        if successes.shape != row_ids.shape or successes.dtype != torch.bool:
            raise ValueError("successes must be a Boolean tensor aligned with row_ids.")
        if row_ids.device != self._difficulty_order.device or successes.device != row_ids.device:
            raise ValueError("row_ids and successes must be on the sampler device.")
        if row_ids.numel() == 0:
            return

        ranks = self._row_ids_to_ranks(row_ids)
        row_count = self._difficulty_order.numel()
        batch_attempts = torch.bincount(ranks, minlength=row_count)
        batch_successes = torch.bincount(ranks, weights=successes.float(), minlength=row_count)
        touched = batch_attempts > 0

        capacity = float(self.cfg.history_capacity)
        kept_batch_attempts = batch_attempts[touched].float().clamp_max(capacity)
        kept_batch_scale = kept_batch_attempts / batch_attempts[touched].float()
        kept_batch_successes = batch_successes[touched] * kept_batch_scale
        old_attempts = self._effective_attempts[touched]
        old_scale = ((capacity - kept_batch_attempts) / old_attempts.clamp_min(1.0)).clamp_(0.0, 1.0)
        self._effective_successes[touched] = self._effective_successes[touched] * old_scale + kept_batch_successes
        self._effective_attempts[touched] = old_attempts * old_scale + kept_batch_attempts
        self._total_attempts.add_(batch_attempts)
        self._total_successes.add_(torch.bincount(ranks[successes], minlength=row_count))

        occurrence = torch.arange(row_ids.numel(), device=row_ids.device, dtype=torch.long)
        latest_occurrence = torch.full((row_count,), -1, device=row_ids.device, dtype=torch.long)
        latest_occurrence.scatter_reduce_(0, ranks, occurrence, reduce="amax", include_self=True)
        latest_ranks = torch.nonzero(latest_occurrence >= 0, as_tuple=False).flatten()
        self._latest_success[latest_ranks] = successes[latest_occurrence[latest_ranks]]
        self._has_outcome[latest_ranks] = True

        self._advance_frontier(ranks, successes)
        self._probabilities_dirty = True

    def metrics(self) -> dict[str, float]:
        """Return compact sampler metrics using one device-to-host transfer."""
        probabilities = self.sampling_probabilities
        success = self.success_estimates
        effective_attempts = self._effective_attempts.sum()
        values = torch.stack(
            (
                torch.as_tensor(self.cfg.target_success_rate, device=success.device),
                torch.dot(probabilities, success),
                self._effective_successes.sum() / effective_attempts.clamp_min(1.0),
                self._latest_success.sum() / success.numel(),
                (self._total_successes > 0).sum() / success.numel(),
                self._has_outcome.sum() / success.numel(),
                torch.as_tensor(self._frontier_size / success.numel(), device=success.device),
                probabilities.square().sum().reciprocal(),
            )
        )
        names = (
            "target_success_rate",
            "predicted_success_rate",
            "bounded_success_rate",
            "cache_success_rate",
            "ever_solved_fraction",
            "evaluated_fraction",
            "frontier_fraction",
            "effective_pool_size",
        )
        return dict(zip(names, values.tolist(), strict=True))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of the adaptive sampling state."""
        device = self._difficulty_order.device
        return {
            "version": torch.tensor(self._STATE_VERSION, device=device, dtype=torch.long),
            "difficulty_order": self._difficulty_order.detach().clone(),
            "effective_successes": self._effective_successes.detach().clone(),
            "effective_attempts": self._effective_attempts.detach().clone(),
            "total_successes": self._total_successes.detach().clone(),
            "total_attempts": self._total_attempts.detach().clone(),
            "has_outcome": self._has_outcome.detach().clone(),
            "latest_success": self._latest_success.detach().clone(),
            "frontier_size": torch.tensor(self._frontier_size, device=device, dtype=torch.long),
            "frontier_credit": self._frontier_credit.detach().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Restore adaptive sampling state produced by :meth:`state_dict`.

        Args:
            state_dict: Mapping containing all sampler state tensors.
        """
        expected = set(self._STATE_KEYS)
        if set(state_dict) != expected:
            missing = sorted(expected - set(state_dict))
            unexpected = sorted(set(state_dict) - expected)
            raise ValueError(f"Invalid sampler state keys; missing={missing}, unexpected={unexpected}.")
        device = self._difficulty_order.device
        self._validate_state_tensor("version", torch.Size([]), torch.long, state_dict)
        self._validate_state_tensor("difficulty_order", self._difficulty_order.shape, torch.long, state_dict)
        self._validate_state_tensor(
            "effective_successes", self._effective_successes.shape, self._effective_successes.dtype, state_dict
        )
        self._validate_state_tensor(
            "effective_attempts", self._effective_attempts.shape, self._effective_attempts.dtype, state_dict
        )
        self._validate_state_tensor("total_successes", self._total_successes.shape, torch.long, state_dict)
        self._validate_state_tensor("total_attempts", self._total_attempts.shape, torch.long, state_dict)
        self._validate_state_tensor("has_outcome", self._has_outcome.shape, torch.bool, state_dict)
        self._validate_state_tensor("latest_success", self._latest_success.shape, torch.bool, state_dict)
        self._validate_state_tensor("frontier_size", torch.Size([]), torch.long, state_dict)
        self._validate_state_tensor("frontier_credit", torch.Size([]), torch.float32, state_dict)

        version = int(state_dict["version"].to(device=device).item())
        if version != self._STATE_VERSION:
            raise ValueError(f"Unsupported adaptive reset sampler state version {version}.")
        restored_order = state_dict["difficulty_order"].to(device=device, dtype=torch.long)
        if not torch.equal(restored_order, self._difficulty_order):
            raise ValueError("Sampler state difficulty_order does not match this reset cache.")

        effective_successes = state_dict["effective_successes"].to(device=device)
        effective_attempts = state_dict["effective_attempts"].to(device=device)
        total_successes = state_dict["total_successes"].to(device=device)
        total_attempts = state_dict["total_attempts"].to(device=device)
        invalid_effective = (
            ~torch.isfinite(effective_successes)
            | ~torch.isfinite(effective_attempts)
            | (effective_successes < 0.0)
            | (effective_attempts < effective_successes)
            | (effective_attempts > self.cfg.history_capacity)
        )
        if bool(torch.any(invalid_effective)):
            raise ValueError("Sampler state contains invalid bounded outcome counts.")
        if bool(torch.any((total_successes < 0) | (total_attempts < total_successes))):
            raise ValueError("Sampler state contains invalid lifetime outcome counts.")
        frontier_size = int(state_dict["frontier_size"].to(device=device).item())
        if not 1 <= frontier_size <= self._difficulty_order.numel():
            raise ValueError("Sampler state frontier_size is outside the reset cache.")
        frontier_credit = state_dict["frontier_credit"].to(device=device)
        if (
            not bool(torch.isfinite(frontier_credit))
            or not 0.0 <= float(frontier_credit.item()) < self.cfg.frontier_evidence
        ):
            raise ValueError("Sampler state frontier_credit is outside its valid range.")

        self._copy_state_tensor("effective_successes", self._effective_successes, state_dict)
        self._copy_state_tensor("effective_attempts", self._effective_attempts, state_dict)
        self._copy_state_tensor("total_successes", self._total_successes, state_dict)
        self._copy_state_tensor("total_attempts", self._total_attempts, state_dict)
        self._copy_state_tensor("has_outcome", self._has_outcome, state_dict)
        self._copy_state_tensor("latest_success", self._latest_success, state_dict)
        self._frontier_size = frontier_size
        self._frontier_credit.copy_(frontier_credit)
        self._probabilities_dirty = True

    def _compute_probabilities(self) -> torch.Tensor:
        """Build the target-success mixture in difficulty-rank order."""
        success = self.success_estimates
        active_rows = torch.arange(self._frontier_size, device=success.device)
        active_success = success[active_rows]
        probe_size = min(self.cfg.probe_size, success.numel() - self._frontier_size)
        probe_fraction = self.cfg.probe_fraction if probe_size > 0 else 0.0
        probabilities = torch.zeros_like(success)

        if probe_size > 0:
            probe_rows = torch.arange(
                self._frontier_size,
                self._frontier_size + probe_size,
                device=success.device,
            )
            probabilities[probe_rows] = probe_fraction / probe_size
            probe_success = success[probe_rows].mean()
            active_target = (self.cfg.target_success_rate - probe_fraction * probe_success) / (1.0 - probe_fraction)
        else:
            active_target = torch.as_tensor(self.cfg.target_success_rate, device=success.device)

        max_inverse_temperature = 1.0 / self.cfg.temperature
        lower = torch.full((), -max_inverse_temperature, device=success.device)
        upper = torch.full((), max_inverse_temperature, device=success.device)
        uniform = torch.full_like(active_success, 1.0 / self._frontier_size)
        active_target = active_target.clamp(0.0, 1.0)
        for _ in range(self._BISECTION_STEPS):
            inverse_temperature = 0.5 * (lower + upper)
            softmax = torch.softmax(-inverse_temperature * active_success, dim=0)
            candidate = (1.0 - self.cfg.replay_fraction) * softmax + self.cfg.replay_fraction * uniform
            predicted = torch.dot(candidate, active_success)
            lower = torch.where(predicted > active_target, inverse_temperature, lower)
            upper = torch.where(predicted > active_target, upper, inverse_temperature)

        softmax = torch.softmax(-0.5 * (lower + upper) * active_success, dim=0)
        active_probabilities = (1.0 - self.cfg.replay_fraction) * softmax + self.cfg.replay_fraction * uniform
        probabilities[active_rows] = (1.0 - probe_fraction) * active_probabilities
        return probabilities

    def _advance_frontier(self, ranks: torch.Tensor, successes: torch.Tensor) -> None:
        """Advance, but never retract, the active difficulty frontier."""
        row_count = self._difficulty_order.numel()
        if self._frontier_size >= row_count:
            return
        window_size = max(self.cfg.probe_size, 1)
        frontier_start = max(self._frontier_size - window_size, 0)
        frontier_end = min(self._frontier_size + window_size, row_count)
        near_frontier = (ranks >= frontier_start) & (ranks < frontier_end)
        contribution = torch.where(
            near_frontier,
            successes.float() - self.cfg.target_success_rate,
            torch.zeros_like(successes, dtype=torch.float32),
        ).sum()
        self._frontier_credit.add_(contribution).clamp_min_(0.0)
        advance = min(
            int(torch.floor(self._frontier_credit / self.cfg.frontier_evidence).item()),
            row_count - self._frontier_size,
        )
        if advance > 0:
            self._frontier_size += advance
            self._frontier_credit.sub_(advance * self.cfg.frontier_evidence)

    def _row_ids_to_ranks(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Resolve raw row IDs to positions in the difficulty ordering."""
        positions = torch.searchsorted(self._sorted_row_ids, row_ids)
        in_range = positions < self._sorted_row_ids.numel()
        safe_positions = positions.clamp_max(self._sorted_row_ids.numel() - 1)
        valid = in_range & (self._sorted_row_ids[safe_positions] == row_ids)
        if not bool(torch.all(valid)):
            invalid = row_ids[~valid].detach().cpu().tolist()
            raise ValueError(f"Unknown raw reset-row IDs: {invalid}.")
        return self._sorted_to_rank[safe_positions]

    @staticmethod
    def _validate_state_tensor(
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Validate the metadata of one serialized state tensor."""
        source = state_dict[name]
        if not isinstance(source, torch.Tensor) or source.shape != shape or source.dtype != dtype:
            source_shape = source.shape if isinstance(source, torch.Tensor) else None
            source_dtype = source.dtype if isinstance(source, torch.Tensor) else None
            raise ValueError(
                f"Sampler state {name!r} has shape/dtype {source_shape}/{source_dtype}; expected {shape}/{dtype}."
            )

    @staticmethod
    def _copy_state_tensor(
        name: str,
        target: torch.Tensor,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Validate and copy one state tensor."""
        target.copy_(state_dict[name].to(device=target.device))
