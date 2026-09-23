# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the task-local Asimov-1 AMP components."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


def _load_agent_modules():
    package_name = "_asimov_1_agents_under_test"
    agents_path = (
        Path(__file__).parents[2] / "isaaclab_tasks" / "contrib" / "velocity" / "config" / "asimov_1" / "agents"
    )
    spec = importlib.util.spec_from_file_location(
        package_name,
        agents_path / "__init__.py",
        submodule_search_locations=[str(agents_path)],
    )
    package = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return (
        importlib.import_module(f"{package_name}.amp_ppo"),
        importlib.import_module(f"{package_name}.discriminator"),
        importlib.import_module(f"{package_name}.replay_buffer"),
    )


amp_ppo, discriminator, replay_buffer = _load_agent_modules()
AMPPPO = amp_ppo.AMPPPO
AMPDiscriminator = discriminator.AMPDiscriminator
AMPFeatureNormalizer = discriminator.AMPFeatureNormalizer
AMPReplayBuffer = replay_buffer.AMPReplayBuffer


def test_feature_normalizer_matches_baseline_numpy_equations():
    rng = np.random.default_rng(7)
    batches = [
        rng.normal(size=(17, 4)).astype(np.float32),
        rng.normal(1.0, 2.0, size=(9, 4)).astype(np.float32),
    ]
    expected_mean = np.zeros(4, np.float64)
    expected_var = np.ones(4, np.float64)
    expected_count = 1.0e-4
    normalizer = AMPFeatureNormalizer(4)

    for batch in batches:
        expected_normalized = np.clip((batch - expected_mean) / np.sqrt(expected_var + 1.0e-4), -10.0, 10.0)
        actual_normalized = normalizer(torch.from_numpy(batch)).numpy()
        np.testing.assert_allclose(actual_normalized, expected_normalized, rtol=1.0e-6, atol=1.0e-6)

        batch_mean = np.mean(expected_normalized, axis=0)
        batch_var = np.var(expected_normalized, axis=0)
        batch_count = len(batch)
        delta = batch_mean - expected_mean
        total_count = expected_count + batch_count
        new_mean = expected_mean + delta * batch_count / total_count
        moment_2 = (
            expected_var * expected_count
            + batch_var * batch_count
            + np.square(delta) * expected_count * batch_count / total_count
        )
        expected_mean = new_mean
        expected_var = moment_2 / total_count
        expected_count = total_count

        normalizer.update(torch.from_numpy(expected_normalized))
        np.testing.assert_allclose(normalizer._mean.numpy().squeeze(0), expected_mean, rtol=1.0e-12)
        np.testing.assert_allclose(normalizer._var.numpy().squeeze(0), expected_var, rtol=1.0e-12)
        assert normalizer.count.item() == expected_count


def test_discriminator_reward_and_raw_gradient_penalty_match_formulas():
    torch.manual_seed(3)
    discriminator = AMPDiscriminator(observation_dim=3, hidden_dims=[8, 4])
    state = torch.randn(6, 3)
    next_state = torch.randn(6, 3)
    discriminator.feature_norm._mean.fill_(2.0)
    discriminator.feature_norm._var.fill_(4.0)

    reward, prediction = discriminator.predict_amp_reward(state, next_state)
    normalized_transition = torch.cat((discriminator.normalize(state), discriminator.normalize(next_state)), dim=-1)
    expected_prediction = discriminator(normalized_transition).squeeze(-1)
    expected_reward = torch.clamp(1.0 - 0.25 * torch.square(expected_prediction - 1.0), min=0.0)
    torch.testing.assert_close(prediction, expected_prediction)
    torch.testing.assert_close(reward, expected_reward)

    transition = torch.cat((state, next_state), dim=-1).detach().requires_grad_(True)
    raw_prediction = discriminator(transition)
    raw_gradient = torch.autograd.grad(
        raw_prediction,
        transition,
        grad_outputs=torch.ones_like(raw_prediction),
        create_graph=True,
    )[0]
    expected_penalty = 10.0 * raw_gradient.norm(2, dim=1).pow(2).mean()
    actual_penalty = discriminator.compute_grad_pen(state, next_state, lambda_=10.0)
    torch.testing.assert_close(actual_penalty, expected_penalty)


def test_discriminator_updates_statistics_from_normalized_samples():
    discriminator = AMPDiscriminator(observation_dim=2, hidden_dims=[8])
    policy_state = torch.tensor([[2.0, 4.0], [4.0, 6.0]])
    expert_state = torch.tensor([[6.0, 8.0], [8.0, 10.0]])
    expected = AMPFeatureNormalizer(2)
    normalized_policy = expected(policy_state)
    normalized_expert = expected(expert_state)
    expected.update(normalized_policy)
    expected.update(normalized_expert)

    discriminator.update_normalization(policy_state, expert_state)
    torch.testing.assert_close(discriminator.feature_norm._mean, expected._mean)
    torch.testing.assert_close(discriminator.feature_norm._var, expected._var)


def test_replay_buffer_keeps_state_pairs_aligned_across_wraparound():
    replay = AMPReplayBuffer(observation_dim=2, capacity=5, device="cpu")
    first = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    second = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 20.0
    replay.insert(first, first + 100.0)
    replay.insert(second, second + 100.0)

    assert replay.num_samples == 5
    torch.testing.assert_close(replay.next_states, replay.states + 100.0)
    sampled_states, sampled_next_states = next(replay.generator(num_batches=1, batch_size=20))
    torch.testing.assert_close(sampled_next_states, sampled_states + 100.0)


def _make_algorithm() -> tuple[AMPPPO, RolloutStorage, TensorDict]:
    class Commands:
        def get_command(self, _name):
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    num_envs = 2
    obs = TensorDict(
        {
            "policy": torch.randn(num_envs, 4),
            "critic": torch.randn(num_envs, 5),
            "amp": torch.randn(num_envs, 3),
        },
        batch_size=[num_envs],
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=[8],
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[8], obs_normalization=False)
    storage = RolloutStorage("rl", num_envs, 1, obs, [2], "cpu")
    algorithm = AMPPPO(
        actor,
        critic,
        storage,
        amp_data=object(),
        amp_observation_dim=3,
        command_manager=Commands(),
        amp_reward_command_gate=True,
        amp_reward_command_threshold=0.1,
        device="cpu",
    )
    return algorithm, storage, obs


def test_amp_ppo_pairs_frames_and_blends_rewards_without_custom_runner():
    algorithm, storage, obs = _make_algorithm()
    num_envs = len(obs)

    initial_amp_state = obs["amp"].clone()
    algorithm.act(obs)
    next_obs = TensorDict(
        {
            "policy": torch.full((num_envs, 4), 1000.0),
            "critic": torch.full((num_envs, 5), -1000.0),
            "amp": torch.randn(num_envs, 3),
        },
        batch_size=[num_envs],
    )
    raw_amp_reward, _ = algorithm.discriminator.predict_amp_reward(initial_amp_state, next_obs["amp"])
    task_reward = torch.tensor([2.0, 3.0])
    gate = torch.tensor([1.0, 0.0])
    expected_reward = 0.3 * (0.3 * raw_amp_reward * gate) + 0.7 * task_reward

    algorithm.process_env_step(next_obs, task_reward, torch.zeros(num_envs), {"log": {}})

    torch.testing.assert_close(algorithm.amp_storage.states[:num_envs], initial_amp_state)
    torch.testing.assert_close(algorithm.amp_storage.next_states[:num_envs], next_obs["amp"])
    torch.testing.assert_close(storage.rewards[0, :, 0], expected_reward)
    assert next_obs["policy"].max() == 500.0
    assert next_obs["critic"].min() == -500.0


def test_amp_ppo_uses_one_joint_optimizer():
    algorithm, _, _ = _make_algorithm()

    assert len(algorithm.optimizer.param_groups) == 3
    optimizer_parameters = {
        id(parameter) for group in algorithm.optimizer.param_groups for parameter in group["params"]
    }
    discriminator_parameters = {id(parameter) for parameter in algorithm.discriminator.parameters()}

    assert discriminator_parameters <= optimizer_parameters
    assert "amp_optimizer_state_dict" not in algorithm.save()


def test_amp_ppo_updates_policy_and_discriminator_in_one_step():
    torch.manual_seed(11)
    algorithm, storage, obs = _make_algorithm()
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 1

    class ExpertData:
        def feed_forward_generator(self, num_batches, batch_size):
            for _ in range(num_batches):
                yield torch.randn(batch_size, 3), torch.randn(batch_size, 3)

    algorithm.amp_data = ExpertData()
    algorithm.act(obs)
    next_obs = obs.clone()
    next_obs["amp"] = torch.randn_like(obs["amp"])
    algorithm.process_env_step(next_obs, torch.tensor([1.0, 2.0]), torch.zeros(2), {"log": {}})
    algorithm.compute_returns(next_obs)

    actor_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    discriminator_before = [parameter.detach().clone() for parameter in algorithm.discriminator.parameters()]
    losses = algorithm.update()

    assert "amp" in losses
    assert storage.step == 0
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, algorithm.actor.parameters()))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(discriminator_before, algorithm.discriminator.parameters())
    )
    assert {group["lr"] for group in algorithm.optimizer.param_groups} == {algorithm.learning_rate}
