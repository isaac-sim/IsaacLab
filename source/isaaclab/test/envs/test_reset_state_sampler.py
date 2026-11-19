# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Outcome attribution, native scores, and changing-input CUDA replay contracts."""

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.envs.reset import (
    BetaSamplingScore,
    BetaSamplingScoreCfg,
    ResetStateSampler,
    ResetStateSamplerCfg,
    SamplingScoreCfg,
    sample_cdf,
)
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


class _PriorityScore:
    def __init__(self, cfg):
        del cfg

    def score(self, metrics, out):
        wp.copy(out, metrics["priority"])


@wp.kernel
def _inline_draw(cumulative: wp.array(dtype=float), uniform: wp.array(dtype=float), out: wp.array(dtype=int)):
    i = wp.tid()
    out[i] = sample_cdf(cumulative, 0, cumulative.shape[0], uniform[i])


def test_inline_cdf_draw_skips_zero_weights_and_falls_back_for_an_empty_distribution():
    cumulative = wp.array([1.0, 1.0, 3.0, 4.0], dtype=float, device="cpu")
    uniform = wp.array([0.0, 0.25, 0.5, 0.999], dtype=float, device="cpu")
    out = wp.empty(4, dtype=int, device="cpu")
    wp.launch(_inline_draw, 4, [cumulative, uniform, out], device="cpu")
    np.testing.assert_array_equal(out.numpy(), [0, 2, 2, 3])
    cumulative.zero_()
    wp.launch(_inline_draw, 4, [cumulative, uniform, out], device="cpu")
    np.testing.assert_array_equal(out.numpy(), [0, 1, 2, 3])


def test_requested_ids_respect_groups_and_credit_only_completed_previous_episodes():
    sampler = ResetStateSampler(
        ResetStateSamplerCfg(history_length=2),
        4,
        4,
        "cpu",
        group_ids=torch.tensor([0, 0, 0, 1]),
        num_groups=2,
        state_ids=torch.tensor([42, 7, 101, 300]),
    )
    requested = torch.tensor([42, 42, 42, 300])
    sampler.reset(success=torch.ones(4, dtype=torch.bool), state_ids=requested)
    np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), 0)
    sampler.reset(
        torch.tensor([True, True, True, False]),
        success=torch.tensor([True, False, True, True]),
        valid=torch.ones(4, dtype=torch.bool),
        state_ids=torch.tensor([7, 7, 7, 101]),
    )
    np.testing.assert_array_equal(sampler.state_ids.warp.numpy(), [7, 7, 7, 300])
    np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), [3, 0, 0, 0])
    np.testing.assert_allclose(sampler.success_rate.warp.numpy(), [0.5, 0, 0, 0])
    assert sampler.probabilities.warp.numpy()[0] > sampler.probabilities.warp.numpy()[1]
    sampler.reset(
        env_ids=torch.tensor([2, 0]),
        success=torch.tensor([False, True, True, True]),
        valid=torch.ones(4, dtype=torch.bool),
        state_ids=torch.tensor([42, 7, 42, 300]),
    )
    np.testing.assert_array_equal(sampler.indices.warp.numpy(), [0, 1, 0, 3])
    np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), [3, 2, 0, 0])
    np.testing.assert_allclose(sampler.success_rate.warp.numpy(), [0.5, 0.5, 0, 0])
    sampler.reset(
        env_ids=torch.tensor([1, 3]),
        success=torch.ones(4, dtype=torch.bool),
        state_ids=torch.tensor([42, 999, 42, 42]),
    )
    np.testing.assert_array_equal(sampler.invalid_requests.warp.numpy(), [False, True, False, True])
    np.testing.assert_array_equal(sampler.state_ids.warp.numpy(), [42, 7, 42, 300])
    np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), [3, 2, 0, 0])
    # Manual resets may change starts, but cannot append an unfinished episode.
    sampler.reset(success=torch.ones(4, dtype=torch.bool), valid=torch.zeros(4, dtype=torch.bool))
    np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), [3, 2, 0, 0])
    probabilities = sampler.probabilities.warp.numpy()
    np.testing.assert_allclose([probabilities[:2].sum(), probabilities[2:].sum()], 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    ("target", "kappa", "rates", "expected"),
    [
        (0.5, 2.0, [0.25, 0.75], [0.75, 0.75]),
        (0.0, 2.0, [0.0, 0.5, 1.0], [1.0, 0.25, 0.0]),
        (1.0, 2.0, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0]),
        (0.66, 0.0, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0]),
        (0.66, 1.0e6, [0.0, 0.65, 0.66, 0.67, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_beta_score_keeps_analytic_unit_peak_without_mutating_metrics(target, kappa, rates, expected):
    rates = wp.array(rates, dtype=float, device="cpu")
    original = rates.numpy().copy()
    out = wp.empty_like(rates)
    BetaSamplingScore(BetaSamplingScoreCfg(metric="completion", target=target, kappa=kappa)).score(
        {"completion": rates}, out
    )
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)
    np.testing.assert_array_equal(rates.numpy(), original)


def test_named_metrics_and_concentration_control_the_current_distribution():
    readiness = wp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float, device="cpu")
    cfg = ResetStateSamplerCfg(scores=[BetaSamplingScoreCfg(metric="readiness", weight=0.25)])
    sampler = ResetStateSampler(cfg, 1, 5, "cpu", metrics={"readiness": readiness})
    broad = sampler.probabilities.warp.numpy()
    np.testing.assert_allclose(broad, [0.05722444, 0.27138778, 0.34277556, 0.27138778, 0.05722444], rtol=1e-6)
    cfg.scores[0].kappa = 1000.0
    concentrated = ResetStateSampler(cfg, 1, 5, "cpu", metrics={"readiness": readiness})
    narrow = concentrated.probabilities.warp.numpy()
    np.testing.assert_allclose(narrow[2] / narrow[0], broad[2] / broad[0])
    assert narrow[2] > broad[2]
    readiness.assign(np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    concentrated.reset(wp.zeros(1, dtype=wp.bool, device="cpu"))
    assert concentrated.probabilities.warp.numpy().argmax() == 0
    np.testing.assert_array_equal(concentrated.episode_counts.warp.numpy(), 0)
    # Different asset groups need independent distributions even when a custom metric
    # puts their scores on very different scales.
    priorities = wp.array([1.0e20, 1.0e20, 0.1, 0.3], dtype=float, device="cpu")
    partitioned = ResetStateSampler(
        ResetStateSamplerCfg(scores=[SamplingScoreCfg(class_type=_PriorityScore)], uniform_weight=0.0),
        2,
        4,
        "cpu",
        group_ids=torch.tensor([0, 1]),
        num_groups=2,
        metrics={"priority": priorities},
    )
    np.testing.assert_allclose(partitioned.probabilities.warp.numpy(), [0.5, 0.5, 0.25, 0.75], rtol=1e-6)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_cuda_graph_replays_changed_masks_ids_outcomes_and_random_streams(device):
    cfg = ResetStateSamplerCfg(history_length=2, seed=27, scores=[])
    sampler = ResetStateSampler(cfg, 4, 5, device)
    reference = ResetStateSampler(cfg, 4, 5, device)
    mask = wp.zeros(4, dtype=wp.bool, device=device)
    success = wp.zeros(4, dtype=wp.bool, device=device)
    valid = wp.ones_like(mask)
    requests = wp.zeros(4, dtype=wp.int64, device=device)
    restored = wp.empty_like(requests)
    for instance in (sampler, reference):
        instance.reset(state_ids=requests)

    def reset_cycle(instance):
        instance.reset(mask, success=success, valid=valid, state_ids=requests)
        wp.copy(restored, instance.state_ids.warp)
        instance.reset(mask)

    # Compile every specialization and initialize sort/scan scratch before capture.
    reset_cycle(sampler)
    reset_cycle(reference)
    with wp.ScopedCapture(device=device) as capture:
        reset_cycle(sampler)

    expected_history = [[] for _ in range(5)]
    expected_visits = np.zeros(5, dtype=np.int64)
    draws = []
    batches = [
        ([True, True, True, False], [True, False, True, False], [True] * 4, [1, 1, 1, 4]),
        ([False] * 4, [True] * 4, [True] * 4, [4] * 4),
        ([False, True, False, True], [False, True, True, False], [True] * 4, [0, 2, 0, 3]),
        ([True] * 4, [False, True, True, True], [False, True, False, True], [4, 3, 2, 1]),
    ]
    for selected, outcomes, validity, chosen in batches:
        before = sampler.indices.warp.numpy()
        mask.assign(np.array(selected, dtype=bool))
        success.assign(np.array(outcomes, dtype=bool))
        valid.assign(np.array(validity, dtype=bool))
        requests.assign(np.array(chosen, dtype=np.int64))
        for env in range(4):
            if selected[env] and validity[env]:
                state = before[env]
                expected_history[state].append(outcomes[env])
                expected_visits[state] += 1
        wp.capture_launch(capture.graph)
        snapshot = restored.numpy()
        for env in range(4):
            assert snapshot[env] == (chosen[env] if selected[env] else before[env])
        # An entirely empty captured reset must not consume any random stream.
        if any(selected):
            reset_cycle(reference)
        np.testing.assert_array_equal(sampler.state_ids.warp.numpy(), reference.state_ids.warp.numpy())
        if selected[1]:
            draws.append(int(sampler.indices.warp.numpy()[1]))
        np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), expected_visits)
        expected_rates = [np.mean(values[-2:]) if values else 0.0 for values in expected_history]
        np.testing.assert_allclose(sampler.success_rate.warp.numpy(), expected_rates)
        np.testing.assert_array_equal(sampler.indices.warp.numpy()[~np.array(selected)], before[~np.array(selected)])
    assert len(set(draws)) > 1
    # Seeding must update storage already retained by this graph, rather than replace its RNG.
    mask.fill_(True)
    valid.zero_()
    counts = sampler.episode_counts.warp.numpy()
    probabilities = sampler.probabilities.warp.numpy()
    sequences = []
    for _ in range(2):
        sampler.reseed(93)
        sequence = []
        for _ in range(2):
            wp.capture_launch(capture.graph)
            sequence.append(sampler.state_ids.warp.numpy())
        sequences.append(sequence)
    np.testing.assert_array_equal(sequences[0], sequences[1])
    sampler.reseed(94)
    different_seed = []
    for _ in range(2):
        wp.capture_launch(capture.graph)
        different_seed.append(sampler.state_ids.warp.numpy())
    assert not np.array_equal(sequences[0], different_seed)
    np.testing.assert_array_equal(sampler.episode_counts.warp.numpy(), counts)
    np.testing.assert_array_equal(sampler.probabilities.warp.numpy(), probabilities)
    assert sampler.cfg.seed == 27
