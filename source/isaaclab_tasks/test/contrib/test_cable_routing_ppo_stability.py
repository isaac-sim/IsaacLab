# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the cable-routing PPO stability configuration."""

import torch

from isaaclab.utils.string import string_to_callable

from isaaclab_tasks.contrib.cable_routing.agents.models import BoundedGaussianDistribution
from isaaclab_tasks.contrib.cable_routing.agents.rsl_rl_ppo_cfg import CableRoutingPPORunnerCfg


def test_cable_routing_stable_ppo_config_serializes_for_rsl_rl() -> None:
    """The serialized task config must preserve the bounded fixed-schedule settings."""
    runner_cfg = CableRoutingPPORunnerCfg().to_dict()

    assert runner_cfg["actor"]["distribution_cfg"] == {
        "class_name": "isaaclab_tasks.contrib.cable_routing.agents.models:BoundedGaussianDistribution",
        "init_std": 0.25,
        "std_type": "log",
        "std_range": (0.02, 0.5),
    }
    assert runner_cfg["algorithm"]["entropy_coef"] == 0.0
    assert runner_cfg["algorithm"]["schedule"] == "fixed"
    assert runner_cfg["algorithm"]["learning_rate"] == 1.0e-4


def test_serialized_cable_routing_distribution_clamps_large_log_std() -> None:
    """The serialized class must clamp its parameter, reported scale, and sampled scale."""
    distribution_cfg = CableRoutingPPORunnerCfg().to_dict()["actor"]["distribution_cfg"]
    distribution_class = string_to_callable(distribution_cfg.pop("class_name"))
    distribution = distribution_class(output_dim=4, **distribution_cfg)
    assert isinstance(distribution, BoundedGaussianDistribution)

    with torch.no_grad():
        distribution.log_std_param.fill_(10.0)
    distribution.update(torch.zeros(32_768, 4))

    torch.testing.assert_close(
        distribution.log_std_param,
        torch.full_like(distribution.log_std_param, torch.log(torch.tensor(0.5))),
        rtol=0.0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(distribution.std, torch.full_like(distribution.std, 0.5), rtol=0.0, atol=1.0e-7)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        sampled_std = distribution.sample().std(dim=0)
    torch.testing.assert_close(sampled_std, torch.full_like(sampled_std, 0.5), rtol=0.0, atol=0.015)

    with torch.no_grad():
        distribution.log_std_param.fill_(-10.0)
    distribution.update(torch.zeros(32_768, 4))
    torch.testing.assert_close(distribution.std, torch.full_like(distribution.std, 0.02), rtol=0.0, atol=1.0e-7)


def test_bounded_distribution_rejects_invalid_ranges() -> None:
    """Invalid bounds must fail at policy construction instead of destabilizing training."""
    for std_range in ((0.0, 0.5), (0.3, 0.2), (0.3, 0.5), (0.02, float("inf"))):
        try:
            BoundedGaussianDistribution(output_dim=2, init_std=0.25, std_range=std_range, std_type="log")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid std_range={std_range} to be rejected")
