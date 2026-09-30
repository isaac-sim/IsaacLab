# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for create_rsl_rl_runner device selection."""

import logging

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg


class _CpuEnv(VecEnv):
    """Minimal CPU environment exposing the attributes the runner reads at construction."""

    def __init__(self):
        self.num_envs = 2
        self.num_actions = 1
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": torch.zeros(self.num_envs, 3)}, batch_size=[self.num_envs])

    def step(self, actions):
        raise NotImplementedError


def _runner_cfg() -> RslRlOnPolicyRunnerCfg:
    model = RslRlMLPModelCfg(hidden_dims=[8], activation="elu")
    cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=4,
        max_iterations=1,
        save_interval=1,
        experiment_name="test",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=model.replace(stochastic=True, init_noise_std=1.0),
        critic=model,
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=1,
            num_mini_batches=1,
            learning_rate=1e-3,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.0,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
    # the entrypoints adapt the configuration to the installed rsl-rl before building the runner
    return handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version())


def test_runner_falls_back_to_env_device_without_cuda(monkeypatch, caplog):
    """The ``cuda:0`` default must not be used on hosts without CUDA, such as macOS, and the switch is reported."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    agent_cfg = _runner_cfg()
    assert agent_cfg.device == "cuda:0"

    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.rsl_rl.utils"):
        runner = create_rsl_rl_runner(_CpuEnv(), agent_cfg)

    assert runner.device == "cpu"
    assert all(p.device.type == "cpu" for p in runner.alg.actor.parameters())
    assert any("'cuda:0' is unavailable" in record.getMessage() for record in caplog.records)
