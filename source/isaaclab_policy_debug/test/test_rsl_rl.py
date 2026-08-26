# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from isaaclab_policy_debug.catalog import LoadedCheckpoint
from isaaclab_policy_debug.rsl_rl import RslRlPolicyFactory


def test_legacy_combined_actor_critic_state_is_converted_strictly():
    payload = {
        "model_state_dict": {
            "log_std": torch.zeros(2),
            "actor.0.weight": torch.zeros(4, 3),
            "actor.0.bias": torch.zeros(4),
            "actor_obs_normalizer._mean": torch.zeros(1, 3),
            "critic.0.weight": torch.zeros(1, 5),
            "critic.0.bias": torch.zeros(1),
            "critic_obs_normalizer._mean": torch.zeros(1, 5),
        }
    }
    checkpoint = LoadedCheckpoint(Path("model.pt"), payload, payload["model_state_dict"], 1, {}, {})

    converted = RslRlPolicyFactory._legacy_state_dicts(checkpoint)

    assert converted is not None
    assert set(converted["actor"]) == {
        "distribution.log_std_param",
        "mlp.0.weight",
        "mlp.0.bias",
        "obs_normalizer._mean",
    }
    assert set(converted["critic"]) == {
        "mlp.0.weight",
        "mlp.0.bias",
        "obs_normalizer._mean",
    }


def test_non_legacy_checkpoint_is_not_converted():
    payload = {"actor_state_dict": {"mlp.0.weight": torch.zeros(2, 2)}}
    checkpoint = LoadedCheckpoint(Path("model.pt"), payload, payload["actor_state_dict"], 1, {}, {})
    assert RslRlPolicyFactory._legacy_state_dicts(checkpoint) is None


def test_legacy_checkpoint_uses_original_deterministic_actor_math():
    actor_0_weight = torch.tensor(((1.0, -1.0), (0.5, 0.25)))
    actor_0_bias = torch.tensor((0.1, -0.2))
    actor_2_weight = torch.tensor(((2.0, -0.5),))
    actor_2_bias = torch.tensor((0.3,))
    state = {
        "log_std": torch.zeros(1),
        "actor.0.weight": actor_0_weight,
        "actor.0.bias": actor_0_bias,
        "actor.2.weight": actor_2_weight,
        "actor.2.bias": actor_2_bias,
        "actor_obs_normalizer._mean": torch.tensor(((1.0, 2.0),)),
        "actor_obs_normalizer._var": torch.ones((1, 2)),
        "actor_obs_normalizer._std": torch.tensor(((2.0, 4.0),)),
        "actor_obs_normalizer.count": torch.tensor(10),
        "critic.0.weight": torch.zeros((1, 1)),
        "critic.0.bias": torch.zeros(1),
    }
    checkpoint = LoadedCheckpoint(Path("model.pt"), {"model_state_dict": state}, state, 1, {}, {})
    agent_cfg = {
        "class_name": "OnPolicyRunner",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {"activation": "elu"},
    }
    observation = {"policy": torch.tensor(((5.0, -2.0),))}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(device="cpu"),
        num_actions=1,
        get_observations=lambda: observation,
    )
    factory = RslRlPolicyFactory(env, agent_cfg, "cpu")

    policy = factory.create(checkpoint)
    action = policy(observation)

    normalized = (observation["policy"] - state["actor_obs_normalizer._mean"]) / (
        state["actor_obs_normalizer._std"] + 1.0e-2
    )
    hidden = torch.nn.functional.elu(torch.nn.functional.linear(normalized, actor_0_weight, actor_0_bias))
    expected = torch.nn.functional.linear(hidden, actor_2_weight, actor_2_bias)
    torch.testing.assert_close(action, expected)


def test_legacy_checkpoint_rejects_incompatible_policy_observation_before_activation():
    state = {
        "actor.0.weight": torch.zeros((1, 2)),
        "actor.0.bias": torch.zeros(1),
        "critic.0.weight": torch.zeros((1, 1)),
        "critic.0.bias": torch.zeros(1),
    }
    checkpoint = LoadedCheckpoint(Path("model.pt"), {"model_state_dict": state}, state, 1, {}, {})
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(device="cpu"),
        num_actions=1,
        get_observations=lambda: {"policy": torch.zeros((1, 3))},
    )
    factory = RslRlPolicyFactory(
        env,
        {
            "class_name": "OnPolicyRunner",
            "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
            "policy": {"activation": "elu"},
        },
        "cpu",
    )

    with pytest.raises(ValueError, match="policy observation"):
        factory.create(checkpoint)
