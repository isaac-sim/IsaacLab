# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NIST monitoring owns outcomes independently of native reset-state selection."""

import pytest
import torch

from isaaclab.envs.reset import ResetStateSampler, ResetStateSamplerCfg

from isaaclab_tasks.contrib.nist.utils.variant_sampling import BalancedVariantScoreCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.mark.parametrize("beta_weight,floor", [(1.0, 1e-4), (0.0, 1.0)])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_variant_score_preserves_joint_cell_balancing(beta_weight: float, floor: float, device: str) -> None:
    rates = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0, 0.4, 0.6, 0.9], device=device)
    cells = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5], dtype=torch.long, device=device)
    sampler = ResetStateSampler(
        ResetStateSamplerCfg(
            history_length=1,
            scores=[BalancedVariantScoreCfg(num_labels=2, num_variants=3, beta_weight=beta_weight, floor=floor)],
            uniform_weight=0.0,
        ),
        4,
        len(cells),
        device,
        metrics={"monitor_success_rate": rates, "cell_indices": cells},
    )

    # Compare against the original Torch balancing rule, including uneven cell occupancy.
    expected = floor + beta_weight * rates.pow(0.66) * (1.0 - rates).pow(0.34)
    raw_cells = torch.zeros(6, device=device).scatter_add_(0, cells, expected)
    balanced = raw_cells.reshape(2, 3).clone()
    for _ in range(4):
        balanced /= balanced.sum(dim=1, keepdim=True) * 2
        balanced /= balanced.sum(dim=0, keepdim=True) * 3
    expected *= balanced.flatten()[cells] / raw_cells[cells]
    expected /= expected.sum()
    torch.testing.assert_close(sampler.probabilities.torch, expected, rtol=1e-5, atol=1e-6)

    previous = sampler.reset(env_ids=torch.arange(4, device=device)).torch.clone()
    rates[0] = 0.66
    current = sampler.reset(env_ids=torch.tensor([0], device=device)).torch
    torch.testing.assert_close(current[1:], previous[1:])
    if beta_weight:
        assert not torch.allclose(sampler.probabilities.torch, expected)
