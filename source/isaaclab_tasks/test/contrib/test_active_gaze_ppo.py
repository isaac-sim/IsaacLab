# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaze-supervised PPO and the fixed-noise gaze head of the active-gaze tasks, without a simulator."""

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.contrib.active_gaze.gaze_distribution import GazeGaussianDistribution
from isaaclab_tasks.contrib.active_gaze.gaze_distribution_cfg import GazeGaussianDistributionCfg
from isaaclab_tasks.contrib.active_gaze.gaze_supervised_ppo import GazeSupervisedPPO, gaze_loss

pytestmark = pytest.mark.kitless

NUM_ENVS = 64
STEPS = 8


def test_gaze_loss_ignores_invalid_labels():
    mean = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([[0.3, 0.4, 1.0], [float("nan"), float("nan"), 0.0]])
    loss = gaze_loss(mean, labels)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.5)
    assert torch.isfinite(mean.grad).all() and not mean.grad[1].any()


def _algorithm(
    oracle_loss_coef: float = 1.0, with_labels: bool = True, stock: bool = False, gaze_head: bool = False
) -> tuple[PPO | GazeSupervisedPPO, TensorDict]:
    torch.manual_seed(0)
    labels = torch.rand(NUM_ENVS, 3) * 2 - 1
    labels[:, 2] = (torch.arange(NUM_ENVS) % 4 != 0).float()
    labels[labels[:, 2] == 0, :2] = float("nan")
    groups = {"policy": torch.randn(NUM_ENVS, 6)} | ({"oracle": labels} if with_labels else {})
    obs = TensorDict(groups, batch_size=[NUM_ENVS])
    model_groups = {"actor": ["policy"], "critic": ["policy"]}
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    if gaze_head:
        head = GazeGaussianDistributionCfg(init_std=1.0, gaze_std=0.1)
        distribution = {"class_name": head.class_name, "init_std": 1.0, "gaze_std": head.gaze_std}
    actor = MLPModel(obs, model_groups, "actor", 3, hidden_dims=[32], distribution_cfg=distribution)
    critic = MLPModel(obs, model_groups, "critic", 1, hidden_dims=[32])
    storage = RolloutStorage("rl", NUM_ENVS, STEPS, obs, [3], "cpu")
    settings = dict(num_learning_epochs=4, num_mini_batches=4, learning_rate=3e-3, schedule="fixed", device="cpu")
    if stock:
        return PPO(actor, critic, storage, **settings), obs
    return GazeSupervisedPPO(actor, critic, storage, oracle_loss_coef=oracle_loss_coef, **settings), obs


def _train(algorithm: PPO, obs: TensorDict, iterations: int) -> dict[str, float]:
    for _ in range(iterations):
        for _ in range(STEPS):
            with torch.inference_mode():
                algorithm.act(obs)
                algorithm.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        algorithm.compute_returns(obs)
        losses = algorithm.update()
    return losses


def test_update_pulls_the_gaze_toward_the_teacher():
    algorithm, obs = _algorithm()
    valid = obs["oracle"][:, 2] > 0.5

    def distance() -> float:
        with torch.no_grad():
            mean = algorithm.actor(obs)[:, -2:]
        return float((mean - obs["oracle"][:, :2])[valid].norm(dim=-1).mean())

    before = distance()
    losses = _train(algorithm, obs, 40)
    assert losses["oracle_valid"] == pytest.approx(0.75)
    assert all(torch.isfinite(p).all() for p in algorithm.actor.parameters())
    assert distance() < 0.7 * before


def test_zero_coefficient_update_matches_stock_ppo_without_labels():
    def updated(stock: bool) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
        algorithm, obs = _algorithm(oracle_loss_coef=0.0, with_labels=False, stock=stock)
        modules = (algorithm.actor, algorithm.critic)
        initial = [p.detach().clone() for module in modules for p in module.parameters()]
        torch.manual_seed(1)
        losses = _train(algorithm, obs, 1)
        return initial, [p.detach().clone() for module in modules for p in module.parameters()], losses

    initial, gaze_params, losses = updated(stock=False)
    stock_initial, stock_params, stock_losses = updated(stock=True)
    assert losses["oracle"] == 0.0 and losses["oracle_valid"] == 0.0
    for name in ("value", "surrogate", "entropy"):
        assert losses[name] == pytest.approx(stock_losses[name], rel=1e-6, abs=1e-7)
    for start, stock_start in zip(initial, stock_initial, strict=True):
        torch.testing.assert_close(start, stock_start, atol=0, rtol=0)
    for param, stock_param in zip(gaze_params, stock_params, strict=True):
        torch.testing.assert_close(param, stock_param, atol=0, rtol=0)
    assert all(not torch.equal(param, start) for param, start in zip(gaze_params, initial, strict=True))


def test_gaze_head_keeps_the_gaze_noise_fixed_while_ppo_learns_the_rest():
    """The entropy bonus and the surrogate move the other actions' noise, never the gaze's."""
    algorithm, obs = _algorithm(gaze_head=True)
    algorithm.entropy_coef = 0.5
    head = algorithm.actor.distribution
    assert isinstance(head, GazeGaussianDistribution)
    assert head.std_param.shape == (1,)
    _train(algorithm, obs, iterations=3)
    with torch.no_grad():
        head.update(torch.zeros(1, 3))
    std = head.std[0]
    torch.testing.assert_close(std[1:], torch.full((2,), 0.1))
    assert abs(float(std[0]) - 1.0) > 1e-3
    # The gaze noise is part of the saved policy, so a checkpoint restores it.
    assert torch.equal(head.state_dict()["gaze_std"], torch.full((2,), 0.1))


def test_gaze_head_fixes_the_last_two_actions_noise_and_rejects_invalid_settings():
    head = GazeGaussianDistribution(5, init_std=2.0, std_type="log", gaze_std=0.05)
    assert head.input_dim == 5
    head.update(torch.zeros(3, 5))
    torch.testing.assert_close(head.std, torch.tensor([[2.0, 2.0, 2.0, 0.05, 0.05]]).expand(3, -1))
    for output_dim, gaze_std in ((5, 0.0), (5, float("inf")), (2, 0.1)):
        with pytest.raises(ValueError, match="gaze"):
            GazeGaussianDistribution(output_dim, gaze_std=gaze_std)
