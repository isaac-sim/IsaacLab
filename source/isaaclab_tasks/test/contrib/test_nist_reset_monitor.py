# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NIST monitoring owns outcomes independently of native reset-state selection."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.reset import BetaSamplingScoreCfg, ResetStateSampler, ResetStateSamplerCfg
from isaaclab.managers import EventTermCfg

from isaaclab_tasks.contrib.nist.utils.event_combinators import TermChoice
from isaaclab_tasks.contrib.nist.utils.variant_sampling import BalancedVariantScoreCfg
from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_monitor_supplies_live_scores_and_credits_distinct_completed_choices() -> None:
    progress = SimpleNamespace(is_success=torch.zeros(3, dtype=torch.bool))
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        cfg=SimpleNamespace(seed=17),
        scene=SimpleNamespace(_ALL_INDICES=torch.arange(3)),
        episode_length_buf=torch.zeros(3, dtype=torch.long),
        reset_buf=torch.zeros(3, dtype=torch.bool),
        termination_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=progress)),
        extras={"log": {}},
        applied=torch.full((3,), -1),
    )

    def record_choice(env, env_ids, choice):
        env.applied[env_ids] = choice

    cfg = EventTermCfg(
        func=TermChoice,
        mode="reset",
        params={
            "terms": {str(i): EventTermCfg(func=record_choice, mode="reset", params={"choice": i}) for i in range(3)},
            "sampling": ResetStateSamplerCfg(
                history_length=1,
                scores=[BetaSamplingScoreCfg(metric="monitor_success_rate", target=0.5)],
                uniform_weight=0.0,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=2),
            "report": True,
        },
    )
    term = TermChoice(cfg, env)
    # Monitor updates are usable independently; native sampling borrows their live metric.
    term.success_monitor.success_update(torch.tensor([0, 1, 1, 2]), torch.tensor([False, False, True, True]))
    term(env, torch.arange(3), **cfg.params)
    torch.testing.assert_close(term.term_success_rate, torch.tensor([0.0, 0.5, 1.0]))
    probabilities = torch.tensor([env.extras["log"][f"Metrics/SampleProb/{i}"] for i in range(3)])
    torch.testing.assert_close(probabilities, torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(env.applied, torch.ones(3, dtype=torch.long))
    assert env.extras["log"]["Metrics/SuccessRate"] == 0.5

    # Credit each completed environment once and in environment order, before resampling.
    env.episode_length_buf.fill_(1)
    env.reset_buf.fill_(True)
    progress.is_success[:] = torch.tensor([False, True, True])
    term(env, torch.tensor([2, 2, 0]), **cfg.params)
    torch.testing.assert_close(term.term_success_rate, torch.tensor([0.0, 0.5, 1.0]))
    term(env, torch.tensor([0]), **cfg.params)
    torch.testing.assert_close(term.term_success_rate, torch.tensor([0.0, 0.5, 1.0]))

    env.reset_buf[0] = False
    progress.is_success[0] = False
    term(env, torch.tensor([0]), **cfg.params)
    torch.testing.assert_close(term.term_success_rate, torch.tensor([0.0, 0.5, 1.0]))


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
