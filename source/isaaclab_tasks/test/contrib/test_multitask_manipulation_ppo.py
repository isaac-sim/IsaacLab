# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for task-balanced PPO's optimizer and logging contract."""

import math

import pytest
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.kitless


def test_task_balanced_ppo_returns_losses_and_task_statistics() -> None:
    """Real optimizer updates retain base statistics and report task scales separately from losses."""
    pytest.importorskip("rsl_rl")
    from rsl_rl.storage import RolloutStorage

    from isaaclab_tasks.contrib.multitask_manipulation.agents.models import TaskHeadedMLPModel, TaskHeadedValueModel
    from isaaclab_tasks.contrib.multitask_manipulation.agents.ppo import TaskBalancedPPO

    torch.manual_seed(42)
    task_names = ("lift", "cabinet", "reach")
    task_ids = torch.arange(3).repeat_interleave(2)
    task_encoding = torch.nn.functional.one_hot(task_ids, num_classes=3).float()
    obs = TensorDict({"policy": torch.cat((task_encoding, torch.randn(6, 5)), dim=-1)}, batch_size=[6])
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = TaskHeadedMLPModel(
        obs,
        obs_groups,
        "actor",
        output_dim=22,
        hidden_dims=(16,),
        distribution_cfg={
            "class_name": (
                "isaaclab_tasks.contrib.multitask_manipulation.agents.models:TaskHeadedGaussianDistribution"
            ),
            "init_std": 0.3,
        },
    )
    critic = TaskHeadedValueModel(obs, obs_groups, "critic", output_dim=1, hidden_dims=(16,))
    storage = RolloutStorage("rl", 6, 4, obs, [22])
    algorithm = TaskBalancedPPO(
        actor,
        critic,
        storage,
        task_names=task_names,
        task_encoding_obs_group="policy",
        task_encoding_slice=(0, 3),
        num_learning_epochs=1,
        num_mini_batches=2,
    )
    diagnostics = {f"{task}_{quantity}_std" for task in task_names for quantity in ("advantage", "return")}

    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    with torch.no_grad():
        for _ in range(4):
            algorithm.act(obs)
            obs = TensorDict({"policy": torch.cat((task_encoding, torch.randn(6, 5)), dim=-1)}, batch_size=[6])
            rewards = task_ids.float() + torch.rand(6)
            algorithm.process_env_step(obs, rewards, torch.zeros(6, dtype=torch.bool), {})
        algorithm.compute_returns(obs)

    losses, statistics = algorithm.update()

    assert {"value", "surrogate", "entropy"} <= losses.keys()
    assert {"Optimizer/learning_rate", "Model/mean_policy_std"} | diagnostics <= statistics.keys()
    assert diagnostics.isdisjoint(losses)
    assert all(math.isfinite(value) for value in (*losses.values(), *statistics.values()))
    assert all(statistics[key] >= 0.0 for key in diagnostics)
    for model, before in ((actor, actor_before), (critic, critic_before)):
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
