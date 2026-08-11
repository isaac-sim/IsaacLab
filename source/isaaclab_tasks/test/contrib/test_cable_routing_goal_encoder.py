# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cable-routing shared goal encoder."""

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.contrib.cable_routing.agents.models import SharedEncoderMLPModel, SharedEncoderPPO
from isaaclab_tasks.contrib.cable_routing.agents.rsl_rl_ppo_cfg import (
    CableRoutingGaussianDistributionCfg,
    CableRoutingPPORunnerCfg,
)

_OBS_GROUPS = {
    "actor": ["policy", "proprio", "cable_state", "goal"],
    "critic": ["policy", "proprio", "cable_state", "goal"],
}
_ENCODER_CFG = {
    "goal": {
        "hidden_dims": [64, 64],
        "latent_dim": 32,
        "activation": "elu",
        "last_activation": "elu",
    }
}


def _make_observations(batch_size: int = 4) -> TensorDict:
    """Create bounded cable-routing observation groups."""
    generator = torch.Generator().manual_seed(7)
    return TensorDict(
        {
            "policy": torch.rand(batch_size, 24, generator=generator) * 2.0 - 1.0,
            "proprio": torch.rand(batch_size, 36, generator=generator) * 2.0 - 1.0,
            "cable_state": torch.rand(batch_size, 96, generator=generator) * 2.0 - 1.0,
            "goal": torch.rand(batch_size, 18, generator=generator) * 2.0 - 1.0,
        },
        batch_size=[batch_size],
    )


def _make_model(obs: TensorDict, obs_set: str, output_dim: int) -> SharedEncoderMLPModel:
    """Create a compact shared-encoder model for unit tests."""
    return SharedEncoderMLPModel(
        obs,
        _OBS_GROUPS,
        obs_set,
        output_dim,
        hidden_dims=[64, 32],
        obs_normalization=True,
        encoder_cfg=_ENCODER_CFG,
    )


def test_goal_encoder_eager_forward_uses_expected_layout():
    """Test eager inference with the 18-D route program and 14-D action output."""
    obs = _make_observations()
    actor = _make_model(obs, "actor", output_dim=14)

    actions = actor(obs)

    assert actions.shape == (4, 14)
    assert actor.encoder_obs_groups == ["goal"]
    assert actor.encoder_input_dims == [18]
    assert actor.obs_groups == ["policy", "proprio", "cable_state"]
    assert actor.encoder_latent_dim == 32
    assert actor._get_latent_dim() == 24 + 36 + 96 + 32


def test_goal_encoder_is_shared_once_and_receives_both_model_gradients():
    """Test actor/critic encoder identity, optimizer ownership, and gradient flow."""
    obs = _make_observations()
    actor = _make_model(obs, "actor", output_dim=14)
    critic = _make_model(obs, "critic", output_dim=1)
    storage = RolloutStorage("rl", 4, 2, obs, [14], "cpu")

    algorithm = SharedEncoderPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
    )

    encoder_parameter = next(algorithm.actor.encoders["goal"].parameters())
    actor_parameter_ids = [id(parameter) for parameter in algorithm.actor.parameters()]
    critic_parameter_ids = [id(parameter) for parameter in algorithm.critic.parameters()]
    optimizer_parameter_ids = [
        id(parameter) for group in algorithm.optimizer.param_groups for parameter in group["params"]
    ]

    assert algorithm.actor.encoders is algorithm.critic.encoders
    assert id(encoder_parameter) in actor_parameter_ids
    assert id(encoder_parameter) not in critic_parameter_ids
    assert optimizer_parameter_ids.count(id(encoder_parameter)) == 1
    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))

    algorithm.optimizer.zero_grad()
    actor_loss = algorithm.actor(obs).square().mean()
    critic_loss = algorithm.critic(obs).square().mean()
    (actor_loss + critic_loss).backward()

    assert encoder_parameter.grad is not None
    assert torch.isfinite(encoder_parameter.grad).all()
    assert torch.count_nonzero(encoder_parameter.grad) > 0


def test_goal_encoder_torchscript_matches_eager_inference():
    """Test the split-input TorchScript wrapper against eager inference."""
    obs = _make_observations()
    actor = _make_model(obs, "actor", output_dim=14).eval()
    raw_obs = torch.cat([obs[group] for group in actor.obs_groups], dim=-1)

    with torch.inference_mode():
        eager_actions = actor(obs)
        scripted_actor = torch.jit.script(actor.as_jit().eval())
        scripted_actions = scripted_actor(raw_obs, [obs["goal"]])

    torch.testing.assert_close(scripted_actions, eager_actions)


def test_cable_routing_runner_encodes_only_the_goal_group():
    """Test task-level observation ordering and goal-encoder dimensions."""
    cfg = CableRoutingPPORunnerCfg()

    assert cfg.obs_groups == _OBS_GROUPS
    assert cfg.actor.encoder_cfg["goal"].hidden_dims == [64, 64]
    assert cfg.actor.encoder_cfg["goal"].latent_dim == 32
    assert cfg.critic.encoder_cfg["goal"].hidden_dims == [64, 64]
    assert cfg.critic.encoder_cfg["goal"].latent_dim == 32
    assert cfg.algorithm.class_name == "isaaclab_tasks.contrib.cable_routing.agents.models:SharedEncoderPPO"


def test_cable_routing_runner_uses_bounded_fixed_exploration():
    """Test task exploration remains bounded without entropy-driven standard-deviation growth."""
    cfg = CableRoutingPPORunnerCfg()
    distribution_cfg = cfg.actor.distribution_cfg

    assert isinstance(distribution_cfg, CableRoutingGaussianDistributionCfg)
    assert distribution_cfg.init_std == 0.25
    assert distribution_cfg.std_type == "log"
    assert distribution_cfg.std_range == (0.02, 0.5)
    assert distribution_cfg.to_dict()["std_range"] == (0.02, 0.5)
    assert cfg.algorithm.entropy_coef == 0.0
    assert cfg.algorithm.learning_rate == 1.0e-4
    assert cfg.algorithm.schedule == "fixed"
