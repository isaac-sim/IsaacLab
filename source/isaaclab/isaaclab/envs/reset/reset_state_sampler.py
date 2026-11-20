# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select reset starts after crediting the completed episodes that used them.

The public sampler owns catalog assignments, success history, and random streams.
Private kernels below implement ID lookup, ordered history updates, and draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.utils import instantiate
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.memory import resolve_1d_mask

from .reset_state_sampler_cfg import ResetStateSamplerCfg
from .sampling_scores import SamplingScore


class ResetStateSampler:
    """Credit completed episodes before assigning starts, using a fixed-size environment mask.

    Persistent storage is allocated at construction. Mask selection, history, score composition, and draws
    use Warp; same-device Torch arrays are zero-copy input shims. Warm the configured kernels and native
    sort/scan primitives before CUDA graph capture. Masks and array values can then change without
    dynamic indexing or host data reads; Warp manages the primitives' temporary storage.

    Catalog entries form equal contiguous partitions. Each environment samples only its configured group;
    the one-group default shares the whole catalog. Optional opaque integer IDs identify procedural recipes
    or cached states, while history and metric arrays follow catalog order.

    Args:
        cfg: History, scoring, and random-stream configuration.
        num_envs: Number of environments.
        num_states: Number of catalog entries, divisible by ``num_groups``.
        device: Storage and computation device.
        group_ids: Fixed per-environment partition indices, shape [num_envs]; defaults to zero.
        num_groups: Number of equal catalog partitions.
        state_ids: Unique signed integer catalog labels, shape [num_states]; -1 is reserved for unassigned
            environments. Defaults to dense indices.
        metrics: Additional borrowed per-entry arrays. Update them in place. ``success_rate``
            and ``episode_counts`` are supplied by the sampler.

    Attributes:
        state_ids: Borrowed catalog labels, shape [num_envs], with -1 for environments not yet assigned.
        indices: Borrowed dense catalog positions, shape [num_envs], with -1 before assignment.
        success_rate: Borrowed recent success rates, shape [num_states].
        episode_counts: Borrowed lifetime counts of completed outcomes, shape [num_states].
        probabilities: Borrowed probabilities, shape [num_states]; each partition sums to one.
        invalid_requests: Borrowed flags for rejected explicit requests in the latest reset, shape [num_envs].

    Treat output arrays as read-only. Their storage is reused by subsequent resets.
    """

    def __init__(
        self,
        cfg: ResetStateSamplerCfg,
        num_envs: int,
        num_states: int,
        device: str | torch.device,
        *,
        group_ids: wp.array | ProxyArray | torch.Tensor | None = None,
        num_groups: int = 1,
        state_ids: wp.array | ProxyArray | torch.Tensor | None = None,
        metrics: Mapping[str, wp.array | ProxyArray | torch.Tensor] | None = None,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (num_envs, num_states, num_groups, cfg.history_length)):
            raise ValueError("Environment, state, group, and history counts must be positive integers.")
        if num_states % num_groups:
            raise ValueError("num_states must be divisible by num_groups.")
        if not math.isfinite(cfg.uniform_weight) or cfg.uniform_weight < 0.0:
            raise ValueError("uniform_weight must be finite and non-negative.")
        self.cfg = cfg
        self.num_envs, self.num_states = num_envs, num_states
        self.device = wp.get_device(str(device))

        # Environment partitions and scratch space for normalizing their weights independently.
        self._group_size = num_states // num_groups
        env_group_ids = np.zeros(num_envs, dtype=np.int32)
        if group_ids is not None:
            group_values = self._input(group_ids, num_envs)
            if group_values.dtype not in (wp.int32, wp.int64):
                raise ValueError("group_ids must use int32 or int64.")
            env_group_ids = group_values.numpy()
        if np.any((env_group_ids < 0) | (env_group_ids >= num_groups)):
            raise ValueError("group_ids must contain valid integer partition indices.")
        self._env_group_ids = wp.array(env_group_ids, dtype=wp.int32, device=self.device)
        self._group_max_weights = wp.zeros(num_groups, dtype=wp.float64, device=self.device) if num_groups > 1 else None

        # Catalog lookup maps externally visible labels to dense storage positions.
        self._dense_ids = state_ids is None
        catalog_ids = np.arange(num_states, dtype=np.int64)
        if state_ids is not None:
            catalog_values = self._input(state_ids, num_states)
            if catalog_values.dtype not in (wp.int32, wp.int64):
                raise ValueError("state_ids must use int32 or int64.")
            catalog_ids = catalog_values.numpy()
        if np.unique(catalog_ids).size != num_states or np.any(catalog_ids == -1):
            raise ValueError("state_ids must be unique and cannot contain the unassigned label -1.")
        self._catalog_ids = wp.array(catalog_ids, dtype=wp.int64, device=self.device)
        sorted_catalog_indices = np.argsort(catalog_ids, kind="stable").astype(np.int32)
        self._sorted_catalog_ids = wp.array(catalog_ids[sorted_catalog_indices], dtype=wp.int64, device=self.device)
        self._sorted_catalog_indices = wp.array(sorted_catalog_indices, dtype=wp.int32, device=self.device)

        # Per-environment reset selection, requested rows, current assignments, and random streams.
        self._all_mask = wp.ones(num_envs, dtype=wp.bool, device=self.device)
        self._scratch_mask = wp.empty_like(self._all_mask)
        self._requested_indices = wp.empty(num_envs, dtype=wp.int32, device=self.device)
        self.invalid_requests = ProxyArray(wp.zeros(num_envs, dtype=wp.bool, device=self.device))
        self._assigned_indices = wp.full(num_envs, -1, dtype=wp.int32, device=self.device)
        self.indices = ProxyArray(self._assigned_indices)
        self.state_ids = ProxyArray(wp.full(num_envs, -1, dtype=wp.int64, device=self.device))
        self._rng = wp.empty(num_envs, dtype=wp.uint32, device=self.device)
        self.reseed(cfg.seed)

        # Per-state outcome history and scratch space for grouping completed episodes.
        self._success_history = wp.zeros((num_states, cfg.history_length), dtype=float, device=self.device)
        self.episode_counts = ProxyArray(wp.zeros(num_states, dtype=wp.int64, device=self.device))
        self.success_rate = ProxyArray(wp.zeros(num_states, dtype=float, device=self.device))
        # Stable radix sort uses the second half of each array as workspace.
        self._outcome_state_indices = wp.empty(2 * num_envs, dtype=wp.int32, device=self.device)
        self._outcome_env_indices = wp.empty_like(self._outcome_state_indices)
        self._state_outcome_start = wp.zeros(num_states, dtype=wp.int32, device=self.device)
        self._state_outcome_end = wp.zeros_like(self._state_outcome_start)

        # Metrics, scores, and the weighted distribution used to draw the next start.
        self._metrics = {"success_rate": self.success_rate.warp, "episode_counts": self.episode_counts.warp}
        if metrics is not None:
            if self._metrics.keys() & metrics.keys():
                raise ValueError("success_rate and episode_counts are supplied by the sampler.")
            self._metrics.update({name: self._input(value, num_states) for name, value in metrics.items()})
        self._scores: list[SamplingScore] = [instantiate(score) for score in cfg.scores]
        weights = [float(score.weight) for score in cfg.scores]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Score weights must be finite and non-negative.")
        self._score_weights = wp.array(weights, dtype=wp.float64, device=self.device)
        self._score_values = wp.empty((len(self._scores), num_states), dtype=float, device=self.device)
        self._weights = wp.empty(num_states, dtype=wp.float64, device=self.device)
        self._cdf = wp.empty_like(self._weights)
        self.probabilities = ProxyArray(wp.empty(num_states, dtype=float, device=self.device))
        self._update_probabilities()

    def reseed(self, seed: int) -> None:
        """Restart each environment's random stream in place, including streams retained by captured graphs.

        Assignments, outcome history, metric values, and the configured initial seed remain unchanged.
        """
        seed = (int(seed) + 2**31) % 2**32 - 2**31
        wp.launch(_initialize_rng, self.num_envs, [seed, self._rng], device=self.device)

    def reset(
        self,
        mask: wp.array | ProxyArray | torch.Tensor | None = None,
        *,
        env_ids: Sequence[int] | slice | wp.array | torch.Tensor | None = None,
        success: wp.array | ProxyArray | torch.Tensor | None = None,
        valid: wp.array | ProxyArray | torch.Tensor | None = None,
        state_ids: wp.array | ProxyArray | torch.Tensor | None = None,
    ) -> ProxyArray:
        """Record previous outcomes and assign starts for selected environments.

        Assignments and random streams of untouched environments are preserved. Simultaneous
        outcomes for a state are ordered by environment index.

        Explicit requested IDs must belong to each selected environment's group. Invalid requests set
        ``invalid_requests`` and leave both the assignment and its outcome history unchanged; there is
        no synchronizing device-value validation. Inspect that flag outside the captured region.

        Args:
            mask: Boolean selection with shape [num_envs]. Selects all environments when
                both selection arguments are omitted.
            env_ids: Alternative selection using environment indices or a slice.
            success: Previous episode success flags, shape [num_envs]. Omit at startup.
            valid: Completed-episode mask, shape [num_envs]. Exclude unfinished manual resets.
            state_ids: Explicit catalog labels, shape [num_envs]. Omit to sample from scores.

        Returns:
            Borrowed persistent full-environment ID assignments. Initially unassigned environments contain -1.
        """
        if isinstance(mask, ProxyArray):
            mask = mask.warp
        selection = resolve_1d_mask(ids=env_ids, mask=mask, all_mask=self._all_mask, scratch_mask=self._scratch_mask)
        # Resolve selection and explicit labels before touching the previous episode.
        if state_ids is None:
            self.invalid_requests.warp.zero_()
        else:
            self._resolve_requested_indices(selection, state_ids)
        # Credit the old assignments, then score and select their replacements.
        if success is not None:
            self._record_completed_outcomes(selection, success, valid)
        if self._scores:
            self._update_probabilities()
        wp.launch(
            _assign_reset_states,
            self.num_envs,
            [
                selection,
                self._env_group_ids,
                self._group_size,
                self._requested_indices,
                state_ids is not None,
                self.invalid_requests.warp,
                self._cdf,
                self._catalog_ids,
                self._rng,
                self._assigned_indices,
                self.state_ids.warp,
            ],
            device=self.device,
        )
        return self.state_ids

    def _resolve_requested_indices(self, selection: wp.array, state_ids: wp.array | ProxyArray | torch.Tensor) -> None:
        """Resolve explicit catalog labels before any previous outcomes are credited."""
        requests = self._input(state_ids, self.num_envs)
        if requests.dtype not in (wp.int32, wp.int64):
            raise TypeError("Requested state_ids must use int32 or int64.")
        wp.launch(
            _resolve_requests,
            self.num_envs,
            [
                selection,
                requests,
                self._sorted_catalog_ids,
                self._sorted_catalog_indices,
                self._env_group_ids,
                self._group_size,
                self._dense_ids,
                self._requested_indices,
                self.invalid_requests.warp,
            ],
            device=self.device,
        )

    def _record_completed_outcomes(
        self,
        selection: wp.array,
        success: wp.array | ProxyArray | torch.Tensor,
        valid: wp.array | ProxyArray | torch.Tensor | None,
    ) -> None:
        """Append completed outcomes to their previous entries in stable environment order.

        Several environments may finish the same entry together. Grouping them by entry
        avoids competing ring-buffer writes; stable sorting retains environment order.
        """
        outcomes = self._input(success, self.num_envs)
        validity = self._all_mask if valid is None else self._input(valid, self.num_envs)
        if validity.dtype != wp.bool:
            raise TypeError("valid must use the boolean dtype.")
        self._state_outcome_start.zero_()
        self._state_outcome_end.zero_()
        wp.launch(
            _index_completed_outcomes,
            self.num_envs,
            [
                selection,
                validity,
                self._assigned_indices,
                self.invalid_requests.warp,
                self.num_states,
                self._outcome_state_indices,
                self._outcome_env_indices,
            ],
            device=self.device,
        )
        # Include num_states, the sentinel key for ignored outcomes.
        wp.utils.radix_sort_pairs(
            self._outcome_state_indices, self._outcome_env_indices, self.num_envs, end_bit=self.num_states.bit_length()
        )
        wp.launch(
            _find_state_outcome_ranges,
            self.num_envs,
            [self._outcome_state_indices, self.num_envs, self._state_outcome_start, self._state_outcome_end],
            device=self.device,
        )
        wp.launch(
            _append_success_history,
            self.num_states,
            [
                self._state_outcome_start,
                self._state_outcome_end,
                self._outcome_env_indices,
                outcomes,
                self._success_history,
                self.episode_counts.warp,
                self.success_rate.warp,
            ],
            device=self.device,
        )

    def _update_probabilities(self) -> None:
        for i, score in enumerate(self._scores):
            score.score(self._metrics, self._score_values[i])
        if self._group_max_weights is not None:
            self._group_max_weights.zero_()
        wp.launch(
            _combine_scores,
            self.num_states,
            [
                self._score_values,
                self._score_weights,
                self.cfg.uniform_weight,
                self._group_size,
                self._weights,
                self._group_max_weights,
            ],
            device=self.device,
        )
        if self._group_max_weights is not None:
            # Keep every nonzero group's largest weight at one before the shared prefix sum.
            # Otherwise a large earlier group can erase small later groups through rounding.
            wp.launch(
                _scale_group_weights,
                self.num_states,
                [self._weights, self._group_max_weights, self._group_size],
                device=self.device,
            )
        wp.utils.array_scan(self._weights, self._cdf)
        wp.launch(
            _probabilities,
            self.num_states,
            [self._weights, self._cdf, self._group_size, self.probabilities.warp],
            device=self.device,
        )

    def _input(self, value: wp.array | ProxyArray | torch.Tensor, size: int) -> wp.array:
        result = _warp_array(value)
        if not isinstance(result, wp.array) or result.shape != (size,) or result.device != self.device:
            raise ValueError(f"Expected a same-device one-dimensional array of size {size}.")
        return result


@wp.kernel
def _initialize_rng(seed: int, states: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    states[i] = wp.rand_init(seed, i)


@wp.kernel
def _resolve_requests(
    mask: wp.array(dtype=wp.bool),
    requested: wp.array(dtype=Any),
    catalog_ids: wp.array(dtype=wp.int64),
    sorted_catalog_indices: wp.array(dtype=wp.int32),
    env_group_ids: wp.array(dtype=wp.int32),
    group_size: int,
    dense: bool,
    requested_indices: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.bool),
):
    env = wp.tid()
    invalid[env] = False
    requested_indices[env] = -1
    if mask[env]:
        value = wp.int64(requested[env])
        index = int(-1)
        if dense:
            if value >= wp.int64(0) and value < wp.int64(catalog_ids.shape[0]):
                index = int(value)
        else:
            lo = int(0)
            hi = catalog_ids.shape[0]
            while lo < hi:
                mid = (lo + hi) // 2
                if catalog_ids[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < catalog_ids.shape[0] and catalog_ids[lo] == value:
                index = sorted_catalog_indices[lo]
        if index >= 0 and index // group_size == env_group_ids[env]:
            requested_indices[env] = index
        else:
            invalid[env] = True


@wp.kernel
def _index_completed_outcomes(
    mask: wp.array(dtype=wp.bool),
    valid: wp.array(dtype=wp.bool),
    assigned_indices: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.bool),
    num_states: int,
    outcome_state_indices: wp.array(dtype=wp.int32),
    outcome_env_indices: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    key = num_states
    if mask[env] and valid[env] and not invalid[env] and assigned_indices[env] >= 0:
        key = assigned_indices[env]
    outcome_state_indices[env] = key
    outcome_env_indices[env] = env


@wp.kernel
def _find_state_outcome_ranges(
    outcome_state_indices: wp.array(dtype=wp.int32),
    size: int,
    state_outcome_start: wp.array(dtype=wp.int32),
    state_outcome_end: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    key = outcome_state_indices[i]
    if key < state_outcome_start.shape[0]:
        if i == 0 or outcome_state_indices[i - 1] != key:
            state_outcome_start[key] = i
        if i == size - 1 or outcome_state_indices[i + 1] != key:
            state_outcome_end[key] = i + 1


@wp.kernel
def _append_success_history(
    state_outcome_start: wp.array(dtype=wp.int32),
    state_outcome_end: wp.array(dtype=wp.int32),
    outcome_env_indices: wp.array(dtype=wp.int32),
    success: wp.array(dtype=Any),
    success_history: wp.array2d(dtype=float),
    episode_counts: wp.array(dtype=wp.int64),
    success_rate: wp.array(dtype=float),
):
    state_index = wp.tid()
    start = state_outcome_start[state_index]
    end = state_outcome_end[state_index]
    if end > start:
        length = success_history.shape[1]
        old_episodes = episode_counts[state_index]
        for i in range(wp.max(start, end - length), end):
            position = int((old_episodes + wp.int64(i - start)) % wp.int64(length))
            success_history[state_index, position] = wp.where(float(success[outcome_env_indices[i]]) > 0.5, 1.0, 0.0)
        episode_counts[state_index] = old_episodes + wp.int64(end - start)
        successes = float(0.0)
        for i in range(length):
            successes += success_history[state_index, i]
        success_rate[state_index] = successes / float(wp.min(episode_counts[state_index], wp.int64(length)))


@wp.kernel
def _combine_scores(
    scores: wp.array2d(dtype=float),
    weights: wp.array(dtype=wp.float64),
    uniform_weight: wp.float64,
    group_size: int,
    out: wp.array(dtype=wp.float64),
    group_max_weights: wp.array(dtype=wp.float64),
):
    state_index = wp.tid()
    value = uniform_weight
    for term in range(weights.shape[0]):
        value += weights[term] * wp.float64(scores[term, state_index])
    out[state_index] = value
    if group_max_weights:
        wp.atomic_max(group_max_weights, state_index // group_size, value)


@wp.kernel
def _scale_group_weights(
    weights: wp.array(dtype=wp.float64), group_max_weights: wp.array(dtype=wp.float64), group_size: int
):
    state_index = wp.tid()
    maximum = group_max_weights[state_index // group_size]
    if maximum > wp.float64(0.0):
        weights[state_index] /= maximum


@wp.kernel
def _probabilities(
    weights: wp.array(dtype=wp.float64),
    cumulative: wp.array(dtype=wp.float64),
    group_size: int,
    out: wp.array(dtype=float),
):
    state_index = wp.tid()
    start = (state_index // group_size) * group_size
    base = wp.float64(0.0)
    if start > 0:
        base = cumulative[start - 1]
    total = cumulative[start + group_size - 1] - base
    if total > wp.float64(0.0):
        out[state_index] = float(weights[state_index] / total)
    else:
        out[state_index] = 1.0 / float(group_size)


@wp.func
def sample_cdf(cumulative: wp.array(dtype=Any), start: int, count: int, uniform: float) -> int:
    """Draw from a cumulative-weight segment using a supplied sample in [0, 1).

    ``count`` must be positive. A zero-total segment falls back to uniform selection.
    Both float32 and float64 storage can be used by inlined MDP terms.
    """
    end = start + count
    base = wp.float64(0.0)
    if start > 0:
        base = wp.float64(cumulative[start - 1])
    total = wp.float64(cumulative[end - 1]) - base
    if total <= wp.float64(0.0):
        return start + wp.min(int(uniform * float(count)), count - 1)
    draw = base + wp.float64(uniform) * total
    lo = start
    hi = end - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if wp.float64(cumulative[mid]) <= draw:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def _assign_reset_states(
    mask: wp.array(dtype=wp.bool),
    env_group_ids: wp.array(dtype=wp.int32),
    group_size: int,
    requested: wp.array(dtype=wp.int32),
    use_requested: bool,
    invalid: wp.array(dtype=wp.bool),
    cumulative: wp.array(dtype=wp.float64),
    catalog_ids: wp.array(dtype=wp.int64),
    rng: wp.array(dtype=wp.uint32),
    assigned_indices: wp.array(dtype=wp.int32),
    state_ids: wp.array(dtype=wp.int64),
):
    env = wp.tid()
    if mask[env] and not invalid[env]:
        index = int(-1)
        if use_requested:
            index = requested[env]
        else:
            start = env_group_ids[env] * group_size
            rng_state = rng[env]
            uniform = wp.randf(rng_state)
            rng[env] = rng_state
            index = sample_cdf(cumulative, start, group_size, uniform)
        assigned_indices[env] = index
        state_ids[env] = catalog_ids[index]


def _warp_array(value: wp.array | ProxyArray | torch.Tensor) -> wp.array:
    if isinstance(value, ProxyArray):
        return value.warp
    return wp.from_torch(value) if isinstance(value, torch.Tensor) else value
