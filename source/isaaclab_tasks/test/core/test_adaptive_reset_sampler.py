# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the task-agnostic adaptive reset sampler."""

import pytest
import torch

from isaaclab_tasks.utils.adaptive_reset_sampler import AdaptiveResetSampler, AdaptiveResetSamplerCfg


def test_config_rejects_invalid_values():
    """Configuration validation rejects values that break the probability model."""
    with pytest.raises(ValueError, match="target_success_rate"):
        AdaptiveResetSamplerCfg(target_success_rate=1.0)
    with pytest.raises(ValueError, match="history_capacity"):
        AdaptiveResetSamplerCfg(history_capacity=0)
    with pytest.raises(ValueError, match="probe_fraction"):
        AdaptiveResetSamplerCfg(probe_fraction=1.0)


def test_sample_preserves_raw_ids_and_exact_overrides():
    """Sampling returns opaque raw IDs and honors overrides outside the frontier."""
    order = torch.tensor([50, 3, 99, 7, 42], dtype=torch.long)
    cfg = AdaptiveResetSamplerCfg(initial_frontier_size=2, probe_size=1, probe_fraction=0.2)
    sampler = AdaptiveResetSampler(order, cfg)

    forced = torch.tensor([-1, 42, 7, -1], dtype=torch.long)
    samples = sampler.sample(4, forced)

    assert samples[1].item() == 42
    assert samples[2].item() == 7
    assert bool(torch.isin(samples[[0, 3]], order[:3]).all())
    with pytest.raises(ValueError, match="Unknown raw reset-row IDs"):
        sampler.sample(1, torch.tensor([1234], dtype=torch.long))
    with pytest.raises(ValueError, match="-1"):
        sampler.sample(1, torch.tensor([-2], dtype=torch.long))
    assert sampler.sample(0).shape == (0,)


def test_record_bounds_effective_outcome_history():
    """Recent evidence remains bounded while retaining the newest batch statistics."""
    sampler = AdaptiveResetSampler(
        torch.tensor([10, 20], dtype=torch.long),
        AdaptiveResetSamplerCfg(history_capacity=4, initial_frontier_size=2, probe_size=0),
    )

    sampler.record(torch.full((4,), 10), torch.ones(4, dtype=torch.bool))
    sampler.record(torch.full((4,), 10), torch.zeros(4, dtype=torch.bool))
    state = sampler.state_dict()
    assert state["effective_attempts"][0].item() == pytest.approx(4.0)
    assert state["effective_successes"][0].item() == pytest.approx(0.0)

    sampler.record(torch.full((2,), 10), torch.ones(2, dtype=torch.bool))
    state = sampler.state_dict()
    assert state["effective_attempts"][0].item() == pytest.approx(4.0)
    assert state["effective_successes"][0].item() == pytest.approx(2.0)
    assert state["total_attempts"][0].item() == 10


def test_probabilities_target_aggregate_success_with_replay():
    """The calibrated softmax approaches the requested aggregate success rate."""
    order = torch.arange(10, dtype=torch.long)
    cfg = AdaptiveResetSamplerCfg(
        target_success_rate=0.3,
        temperature=0.02,
        history_capacity=16,
        prior_strength=0.01,
        initial_frontier_size=10,
        probe_size=0,
        replay_fraction=0.1,
    )
    sampler = AdaptiveResetSampler(order, cfg)
    rows = order.repeat_interleave(16)
    successes = (rows < 5).to(dtype=torch.bool)
    sampler.record(rows, successes)

    predicted = torch.dot(sampler.sampling_probabilities, sampler.success_estimates)
    assert predicted.item() == pytest.approx(0.3, abs=2.0e-3)
    assert sampler.sampling_probabilities.min().item() >= 0.01 - 1.0e-6


def test_probe_and_replay_probability_mass_are_retained():
    """Probe rows and active rows retain their configured probability floors."""
    cfg = AdaptiveResetSamplerCfg(
        initial_frontier_size=4,
        probe_size=2,
        probe_fraction=0.2,
        replay_fraction=0.1,
    )
    sampler = AdaptiveResetSampler(torch.arange(8, dtype=torch.long), cfg)
    probabilities = sampler.sampling_probabilities

    assert probabilities[4:6].sum().item() == pytest.approx(0.2)
    assert probabilities[:4].min().item() >= 0.8 * 0.1 / 4 - 1.0e-6
    assert probabilities[6:].sum().item() == 0.0
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_frontier_advances_monotonically_from_local_evidence():
    """Success near the frontier exposes harder rows and later failures never retract it."""
    cfg = AdaptiveResetSamplerCfg(
        target_success_rate=0.5,
        initial_frontier_size=2,
        probe_size=2,
        frontier_evidence=1.0,
    )
    sampler = AdaptiveResetSampler(torch.arange(8, dtype=torch.long), cfg)

    sampler.record(torch.tensor([1, 2]), torch.tensor([True, True]))
    promoted_size = sampler.frontier_size
    assert promoted_size == 3

    sampler.record(torch.tensor([2, 3, 4]), torch.tensor([False, False, False]))
    assert sampler.frontier_size == promoted_size


def test_state_dict_round_trip_restores_sampling_state():
    """A checkpoint round trip preserves estimates, frontier, and sampling probabilities."""
    order = torch.tensor([7, 19, 2, 31, 4], dtype=torch.long)
    cfg = AdaptiveResetSamplerCfg(initial_frontier_size=2, probe_size=2, frontier_evidence=1.0)
    sampler = AdaptiveResetSampler(order, cfg)
    sampler.record(torch.tensor([7, 19, 2]), torch.tensor([True, True, False]))
    state = sampler.state_dict()

    restored = AdaptiveResetSampler(order, cfg)
    restored.load_state_dict(state)

    assert restored.frontier_size == sampler.frontier_size
    assert torch.equal(restored.success_estimates, sampler.success_estimates)
    assert torch.equal(restored.sampling_probabilities, sampler.sampling_probabilities)
    assert restored.metrics() == pytest.approx(sampler.metrics())

    state["effective_attempts"].zero_()
    assert bool(torch.any(sampler.state_dict()["effective_attempts"] > 0))


def test_state_dict_rejects_a_different_reset_cache():
    """Checkpoint restore cannot silently attach outcome history to different rows."""
    cfg = AdaptiveResetSamplerCfg(initial_frontier_size=2)
    state = AdaptiveResetSampler(torch.tensor([1, 2, 3]), cfg).state_dict()
    sampler = AdaptiveResetSampler(torch.tensor([1, 3, 2]), cfg)

    with pytest.raises(ValueError, match="difficulty_order"):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("field", ("effective_successes", "effective_attempts", "frontier_credit"))
def test_state_dict_rejects_nonfinite_sampling_state(field):
    """Non-finite checkpoint counters cannot poison restored sampling probabilities."""
    sampler = AdaptiveResetSampler(torch.tensor([1, 2, 3]), AdaptiveResetSamplerCfg(initial_frontier_size=2))
    state = sampler.state_dict()
    state[field].reshape(-1)[0] = torch.nan

    with pytest.raises(ValueError, match="invalid|outside"):
        sampler.load_state_dict(state)
