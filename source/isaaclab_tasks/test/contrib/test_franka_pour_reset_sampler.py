# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Pour reset-dataset sampler."""

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_sampler import (
    ResetDatasetSamplerCfg,
    _ResetDatasetSampler,
)


def test_reset_dataset_sampler_uses_exact_rolling_boolean_window():
    """Repeated row outcomes retain exactly the most recent configured Boolean values."""
    sampler = _ResetDatasetSampler(
        3,
        "cpu",
        ResetDatasetSamplerCfg(monitored_history_len=4),
    )

    sampler._record_validated(
        torch.tensor([0, 0, 0, 0, 1, 1]),
        torch.tensor([True, True, False, False, True, True]),
    )
    torch.testing.assert_close(sampler._success_rates, torch.tensor([0.5, 1.0, 0.0]))

    # More than one full window in a single batch must leave only the final four outcomes.
    sampler._record_validated(
        torch.zeros(5, dtype=torch.long),
        torch.tensor([True, False, False, False, False]),
    )
    assert sampler._success_rates[0] == 0.0
    assert sampler._history_sizes[0] == 4
    assert sampler._history_success_counts[0] == 0


def test_validated_record_hot_path_never_branches_on_a_tensor(monkeypatch):
    """Trusted reset outcomes stay device-side even when a batch repeats reset rows."""
    sampler = _ResetDatasetSampler(
        3,
        "cpu",
        ResetDatasetSamplerCfg(monitored_history_len=4),
    )

    def fail_tensor_truth(_tensor):
        pytest.fail("The validated record hot path evaluated a tensor on the host.")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", fail_tensor_truth)
        sampler._record_validated(
            torch.tensor((0, 0, 1), dtype=torch.long),
            torch.tensor((True, False, True)),
        )

    torch.testing.assert_close(sampler._success_rates, torch.tensor((0.5, 1.0, 0.0)))


def test_reset_dataset_sampler_focuses_half_solved_rows_without_starvation():
    """The Beta kernel peaks at the target while epsilon keeps all rows sampleable."""
    sampler = _ResetDatasetSampler(
        3,
        "cpu",
        ResetDatasetSamplerCfg(
            monitored_history_len=50,
            target_success_rate=0.5,
            kappa=1.0,
            epsilon=1.0e-4,
        ),
    )
    sampler._record_validated(
        torch.arange(3).repeat_interleave(50),
        torch.cat(
            (
                torch.zeros(50, dtype=torch.bool),
                torch.arange(50).remainder(2) == 0,
                torch.ones(50, dtype=torch.bool),
            )
        ),
    )

    probabilities = sampler._probabilities()

    assert probabilities[1] > 0.999
    assert probabilities[0] > 0.0
    assert probabilities[2] > 0.0
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_reset_dataset_sampler_cyclic_replay_covers_rows(monkeypatch: pytest.MonkeyPatch):
    """Exact cyclic replay covers every row before switching to the steady fraction."""
    sampler = _ResetDatasetSampler(
        5,
        "cpu",
        ResetDatasetSamplerCfg(
            uniform_fraction_initial=0.5,
            uniform_fraction=0.25,
        ),
    )
    cyclic_rows = []
    take_uniform_rows = sampler._take_uniform_rows

    def record_uniform_rows(count: int, generator: torch.Generator | None) -> torch.Tensor:
        rows = take_uniform_rows(count, generator)
        cyclic_rows.extend(rows.tolist())
        return rows

    monkeypatch.setattr(sampler, "_take_uniform_rows", record_uniform_rows)
    expected_uniform_rows = sampler._uniform_order[:4].clone()
    expected_generator = torch.Generator().manual_seed(17)
    uniform_positions = torch.randperm(8, generator=expected_generator)[:4]
    sampled_rows = sampler._sample_with_uniform_replay(
        8,
        generator=torch.Generator().manual_seed(17),
    )

    torch.testing.assert_close(sampled_rows[uniform_positions], expected_uniform_rows)
    assert sampler.metrics()["sampler/uniform_replay_fraction"] == pytest.approx(0.5)
    assert sampler.metrics()["sampler/uniform_first_sweep_progress"] == pytest.approx(0.8)

    while not sampler.metrics()["sampler/uniform_first_sweep_complete"]:
        sampler._sample_with_uniform_replay(4, generator=torch.Generator().manual_seed(23))

    assert set(cyclic_rows[: sampler.row_count]) == set(range(sampler.row_count))
    assert sampler.metrics()["sampler/uniform_replay_fraction"] == pytest.approx(0.25)
    assert sampler.metrics()["sampler/uniform_first_sweep_progress"] == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("monitored_history_len", 0),
        ("monitored_history_len", True),
        ("target_success_rate", 0.0),
        ("target_success_rate", 1.0),
        ("kappa", 0.0),
        ("epsilon", 0.0),
        ("uniform_fraction", 0.0),
        ("uniform_fraction_initial", 1.0),
    ),
)
def test_reset_dataset_sampler_config_rejects_invalid_values(field, value):
    """Sampler parameters reject values that invalidate its probability model."""
    with pytest.raises(ValueError, match=field):
        ResetDatasetSamplerCfg(**{field: value})
