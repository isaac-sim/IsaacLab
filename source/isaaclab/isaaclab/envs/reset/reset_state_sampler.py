# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-storage, mask-native reset selection and completed-outcome attribution."""

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
from .sampling import SamplingScore, sample_cdf


@wp.kernel
def _initialize_rng(seed: int, states: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    states[i] = wp.rand_init(seed, i)


@wp.kernel
def _resolve_requests(
    mask: wp.array(dtype=wp.bool),
    requested: wp.array(dtype=Any),
    catalog: wp.array(dtype=wp.int64),
    order: wp.array(dtype=wp.int32),
    groups: wp.array(dtype=wp.int32),
    group_size: int,
    dense: bool,
    indices: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.bool),
):
    env = wp.tid()
    invalid[env] = False
    indices[env] = -1
    if mask[env]:
        value = wp.int64(requested[env])
        index = int(-1)
        if dense:
            if value >= wp.int64(0) and value < wp.int64(catalog.shape[0]):
                index = int(value)
        else:
            lo = int(0)
            hi = catalog.shape[0]
            while lo < hi:
                mid = (lo + hi) // 2
                if catalog[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < catalog.shape[0] and catalog[lo] == value:
                index = order[lo]
        if index >= 0 and index // group_size == groups[env]:
            indices[env] = index
        else:
            invalid[env] = True


@wp.kernel
def _outcome_keys(
    mask: wp.array(dtype=wp.bool),
    valid: wp.array(dtype=wp.bool),
    indices: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.bool),
    num_states: int,
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    key = num_states
    if mask[env] and valid[env] and not invalid[env] and indices[env] >= 0:
        key = indices[env]
    keys[env] = key
    values[env] = env


@wp.kernel
def _outcome_ranges(
    keys: wp.array(dtype=wp.int32),
    size: int,
    starts: wp.array(dtype=wp.int32),
    ends: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    key = keys[i]
    if key < starts.shape[0]:
        if i == 0 or keys[i - 1] != key:
            starts[key] = i
        if i == size - 1 or keys[i + 1] != key:
            ends[key] = i + 1


@wp.kernel
def _record_outcomes(
    starts: wp.array(dtype=wp.int32),
    ends: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
    success: wp.array(dtype=Any),
    history: wp.array2d(dtype=float),
    episodes: wp.array(dtype=wp.int64),
    rates: wp.array(dtype=float),
):
    state = wp.tid()
    start = starts[state]
    end = ends[state]
    if end > start:
        length = history.shape[1]
        old_episodes = episodes[state]
        for i in range(wp.max(start, end - length), end):
            position = int((old_episodes + wp.int64(i - start)) % wp.int64(length))
            history[state, position] = wp.where(float(success[values[i]]) > 0.5, 1.0, 0.0)
        episodes[state] = old_episodes + wp.int64(end - start)
        successes = float(0.0)
        for i in range(length):
            successes += history[state, i]
        rates[state] = successes / float(wp.min(episodes[state], wp.int64(length)))


@wp.kernel
def _combine_scores(
    scores: wp.array2d(dtype=float),
    weights: wp.array(dtype=float),
    eps: float,
    out: wp.array(dtype=wp.float64),
):
    state = wp.tid()
    value = eps
    for term in range(weights.shape[0]):
        value += weights[term] * scores[term, state]
    out[state] = wp.float64(value)


@wp.kernel
def _probabilities(
    weights: wp.array(dtype=wp.float64),
    cumulative: wp.array(dtype=wp.float64),
    group_size: int,
    out: wp.array(dtype=float),
):
    state = wp.tid()
    start = (state // group_size) * group_size
    base = wp.float64(0.0)
    if start > 0:
        base = cumulative[start - 1]
    total = cumulative[start + group_size - 1] - base
    if total > wp.float64(0.0):
        out[state] = float(weights[state] / total)
    else:
        out[state] = 1.0 / float(group_size)


@wp.kernel
def _assign(
    mask: wp.array(dtype=wp.bool),
    groups: wp.array(dtype=wp.int32),
    group_size: int,
    requested: wp.array(dtype=wp.int32),
    use_requested: bool,
    invalid: wp.array(dtype=wp.bool),
    cumulative: wp.array(dtype=wp.float64),
    catalog: wp.array(dtype=wp.int64),
    rng: wp.array(dtype=wp.uint32),
    indices: wp.array(dtype=wp.int32),
    state_ids: wp.array(dtype=wp.int64),
):
    env = wp.tid()
    if mask[env] and not invalid[env]:
        index = int(-1)
        if use_requested:
            index = requested[env]
        else:
            start = groups[env] * group_size
            state = rng[env]
            uniform = wp.randf(state)
            rng[env] = state
            index = sample_cdf(cumulative, start, group_size, uniform)
        indices[env] = index
        state_ids[env] = catalog[index]


def _warp_array(value: wp.array | ProxyArray | torch.Tensor) -> wp.array:
    if isinstance(value, ProxyArray):
        return value.warp
    return wp.from_torch(value) if isinstance(value, torch.Tensor) else value


class ResetStateSampler:
    """Credit completed episodes before assigning starts, using a fixed-size environment mask.

    Persistent storage is allocated at construction. Mask selection, history, score composition, and draws
    use Warp; same-device Torch arrays are zero-copy input shims. Warm the configured kernels and native
    sort/scan primitives before CUDA graph capture. Masks and array values can then change without
    dynamic indexing or host data reads; Warp manages the primitives' temporary storage.

    Catalog entries form equal contiguous partitions. Each environment samples only its configured group;
    the one-group default shares the whole catalog. Optional opaque integer IDs identify procedural recipes
    or cached states, while history and metric arrays follow catalog order.

    ``indices`` and ``state_ids`` are borrowed full-environment assignments, initially -1.
    ``success_rate`` and ``episode_counts`` describe catalog entries; ``probabilities`` sums
    to one within each partition. ``score_values`` has shape [num_scores, num_states], in
    ``score_names`` order. Treat these outputs as read-only and change assignments through
    :meth:`reset`. ``invalid_requests`` flags rejected explicit requests without a host read.

    Args:
        cfg: History, scoring, and random-stream configuration.
        num_envs: Number of environments.
        num_states: Number of catalog entries, divisible by ``num_groups``.
        device: Storage and computation device.
        group_ids: Fixed per-environment partition indices, shape [num_envs]; defaults to zero.
        num_groups: Number of equal catalog partitions.
        state_ids: Unique signed integer catalog labels, shape [num_states]; -1 is reserved for unassigned
            environments. Defaults to dense indices.
        metrics: Additional borrowed per-entry arrays. Update their values in place. ``success_rate`` is reserved.

    Attributes:
        state_ids: Borrowed catalog labels, shape [num_envs], with -1 for environments not yet assigned.
        indices: Borrowed dense catalog positions, shape [num_envs], with -1 before assignment.
        success_rate: Borrowed recent success rates, shape [num_states].
        episode_counts: Borrowed lifetime counts of completed outcomes, shape [num_states].
        probabilities: Borrowed probabilities, shape [num_states]; each partition sums to one.
        invalid_requests: Borrowed flags for rejected explicit requests in the latest reset, shape [num_envs].
        score_values: Borrowed unweighted scores, shape [num_scores, num_states].
        score_names: Names corresponding to the score rows.

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
        if not math.isfinite(cfg.eps) or cfg.eps < 0.0:
            raise ValueError("eps must be finite and non-negative.")
        self.cfg = cfg
        self.num_envs, self.num_states = num_envs, num_states
        self.device = wp.get_device(str(device))
        self._group_size = num_states // num_groups
        self._all_mask = wp.ones(num_envs, dtype=wp.bool, device=self.device)
        self._scratch_mask = wp.empty_like(self._all_mask)
        self._indices = wp.full(num_envs, -1, dtype=wp.int32, device=self.device)
        self.indices = ProxyArray(self._indices)
        self.state_ids = ProxyArray(wp.full(num_envs, -1, dtype=wp.int64, device=self.device))
        self.invalid_requests = ProxyArray(wp.zeros(num_envs, dtype=wp.bool, device=self.device))
        self._requested_indices = wp.empty(num_envs, dtype=wp.int32, device=self.device)
        groups = np.zeros(num_envs, dtype=np.int32)
        if group_ids is not None:
            values = self._input(group_ids, num_envs)
            if values.dtype not in (wp.int32, wp.int64):
                raise ValueError("group_ids must use int32 or int64.")
            groups = values.numpy()
        if np.any((groups < 0) | (groups >= num_groups)):
            raise ValueError("group_ids must contain valid integer partition indices.")
        self._groups = wp.array(groups, dtype=wp.int32, device=self.device)
        self._dense_ids = state_ids is None
        catalog = np.arange(num_states, dtype=np.int64)
        if state_ids is not None:
            values = self._input(state_ids, num_states)
            if values.dtype not in (wp.int32, wp.int64):
                raise ValueError("state_ids must use int32 or int64.")
            catalog = values.numpy()
        if np.unique(catalog).size != num_states or np.any(catalog == -1):
            raise ValueError("state_ids must be unique and cannot contain the unassigned label -1.")
        self._catalog = wp.array(catalog, dtype=wp.int64, device=self.device)
        order = np.argsort(catalog, kind="stable").astype(np.int32)
        self._sorted_catalog = wp.array(catalog[order], dtype=wp.int64, device=self.device)
        self._catalog_order = wp.array(order, dtype=wp.int32, device=self.device)
        self._history = wp.zeros((num_states, cfg.history_length), dtype=float, device=self.device)
        self.episode_counts = ProxyArray(wp.zeros(num_states, dtype=wp.int64, device=self.device))
        self.success_rate = ProxyArray(wp.zeros(num_states, dtype=float, device=self.device))
        self._keys = wp.empty(2 * num_envs, dtype=wp.int32, device=self.device)
        self._values = wp.empty_like(self._keys)
        self._starts = wp.zeros(num_states, dtype=wp.int32, device=self.device)
        self._ends = wp.zeros_like(self._starts)
        self._rng = wp.empty(num_envs, dtype=wp.uint32, device=self.device)
        self.reseed(cfg.seed)
        self._metrics = {"success_rate": self.success_rate.warp}
        if metrics is not None:
            if "success_rate" in metrics:
                raise ValueError("success_rate is supplied by the sampler.")
            self._metrics.update({name: self._input(value, num_states) for name, value in metrics.items()})
        self._scores: list[SamplingScore] = [instantiate(score) for score in cfg.scores]
        self.score_names = tuple(score.name for score in self._scores)
        weights = [float(score.weight) for score in cfg.scores]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Score weights must be finite and non-negative.")
        self._score_weights = wp.array(weights, dtype=float, device=self.device)
        self.score_values = ProxyArray(wp.empty((len(self._scores), num_states), dtype=float, device=self.device))
        # A global prefix sum is shared by partitions; double precision keeps small later weights
        # from disappearing when their partition's large prefix is subtracted.
        self._weights = wp.empty(num_states, dtype=wp.float64, device=self.device)
        self._cdf = wp.empty_like(self._weights)
        self.probabilities = ProxyArray(wp.empty(num_states, dtype=float, device=self.device))
        self._update_probabilities()

    def _input(self, value: wp.array | ProxyArray | torch.Tensor, size: int) -> wp.array:
        result = _warp_array(value)
        if not isinstance(result, wp.array) or result.shape != (size,) or result.device != self.device:
            raise ValueError(f"Expected a same-device one-dimensional array of size {size}.")
        return result

    def _update_probabilities(self) -> None:
        for i, score in enumerate(self._scores):
            score.score(self._metrics, self.score_values.warp[i])
        wp.launch(
            _combine_scores,
            self.num_states,
            [self.score_values.warp, self._score_weights, self.cfg.eps, self._weights],
            device=self.device,
        )
        wp.utils.array_scan(self._weights, self._cdf)
        wp.launch(
            _probabilities,
            self.num_states,
            [self._weights, self._cdf, self._group_size, self.probabilities.warp],
            device=self.device,
        )

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

        All outcome, validity, and requested-ID inputs have shape [num_envs]. Omit outcomes at startup;
        mask unfinished/manual episodes with ``valid``. Assignments and random streams of untouched
        environments are preserved. Simultaneous outcomes for a state are ordered by environment index.

        Explicit requested IDs must belong to each selected environment's group. Invalid requests set
        ``invalid_requests`` and leave both the assignment and its outcome history unchanged; there is
        no synchronizing device-value validation. Inspect that flag outside the captured region.

        Returns:
            Borrowed persistent full-environment ID assignments. Initially unassigned environments contain -1.
        """
        if isinstance(mask, ProxyArray):
            mask = mask.warp
        selection = resolve_1d_mask(ids=env_ids, mask=mask, all_mask=self._all_mask, scratch_mask=self._scratch_mask)
        self.invalid_requests.warp.zero_()
        if state_ids is not None:
            requests = self._input(state_ids, self.num_envs)
            if requests.dtype not in (wp.int32, wp.int64):
                raise TypeError("Requested state_ids must use int32 or int64.")
            wp.launch(
                _resolve_requests,
                self.num_envs,
                [
                    selection,
                    requests,
                    self._sorted_catalog,
                    self._catalog_order,
                    self._groups,
                    self._group_size,
                    self._dense_ids,
                    self._requested_indices,
                    self.invalid_requests.warp,
                ],
                device=self.device,
            )
        if success is not None:
            outcomes = self._input(success, self.num_envs)
            validity = self._all_mask if valid is None else self._input(valid, self.num_envs)
            if validity.dtype != wp.bool:
                raise TypeError("valid must use the boolean dtype.")
            self._starts.zero_()
            self._ends.zero_()
            wp.launch(
                _outcome_keys,
                self.num_envs,
                [
                    selection,
                    validity,
                    self._indices,
                    self.invalid_requests.warp,
                    self.num_states,
                    self._keys,
                    self._values,
                ],
                device=self.device,
            )
            wp.utils.radix_sort_pairs(self._keys, self._values, self.num_envs)
            wp.launch(
                _outcome_ranges,
                self.num_envs,
                [self._keys, self.num_envs, self._starts, self._ends],
                device=self.device,
            )
            wp.launch(
                _record_outcomes,
                self.num_states,
                [
                    self._starts,
                    self._ends,
                    self._values,
                    outcomes,
                    self._history,
                    self.episode_counts.warp,
                    self.success_rate.warp,
                ],
                device=self.device,
            )
        self._update_probabilities()
        wp.launch(
            _assign,
            self.num_envs,
            [
                selection,
                self._groups,
                self._group_size,
                self._requested_indices,
                state_ids is not None,
                self.invalid_requests.warp,
                self._cdf,
                self._catalog,
                self._rng,
                self._indices,
                self.state_ids.warp,
            ],
            device=self.device,
        )
        return self.state_ids
