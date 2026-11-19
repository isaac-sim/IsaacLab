# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the tensor-only LEAPP export task and its agent configurations."""

import gymnasium as gym

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpolePPORunnerCfg
from isaaclab_tasks.utils import load_cfg_from_registry

MANAGER_TASK = "Isaac-Leapp-Synthetic"
_AGENTS = "isaaclab_tasks.core.cartpole.agents"


def manager_rsl_rl_cfg() -> CartpolePPORunnerCfg:
    """Exercise multi-group actor selection and observation normalization."""
    cfg = CartpolePPORunnerCfg()
    cfg.obs_groups = {"actor": ["policy", "proprio", "vision"], "critic": ["critic"]}
    cfg.actor = RslRlCNNModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=cfg.actor.distribution_cfg,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(output_channels=[4], kernel_size=3, activation="elu"),
    )
    return cfg


def manager_rsl_rl_distillation_cfg() -> RslRlDistillationRunnerCfg:
    """The student consumes groups that differ from an actor-only export selection."""
    policy_cfg = manager_rsl_rl_cfg()
    return RslRlDistillationRunnerCfg(
        num_steps_per_env=2,
        max_iterations=1,
        save_interval=1,
        experiment_name="leapp_distillation",
        obs_groups={"student": ["policy", "proprio", "vision"], "teacher": ["critic"], "actor": ["critic"]},
        student=policy_cfg.actor,
        teacher=RslRlMLPModelCfg(hidden_dims=[16], activation="elu", obs_normalization=True),
        algorithm=RslRlDistillationAlgorithmCfg(num_learning_epochs=1, gradient_length=2, learning_rate=1.0e-3),
    )


def manager_rl_games_cfg() -> dict:
    """Exercise RL-Games LSTM state with a native checkpoint."""
    cfg = load_cfg_from_registry("Isaac-Cartpole", "rl_games_cfg_entry_point")
    cfg["params"]["network"]["rnn"] = {"name": "lstm", "units": 16, "layers": 1, "before_mlp": True, "layer_norm": True}
    return cfg


def manager_sb3_cfg() -> dict:
    """Exercise dictionary observations and saved per-key normalization."""
    cfg = load_cfg_from_registry("Isaac-Cartpole", "sb3_cfg_entry_point")
    cfg.update(policy="MultiInputPolicy", normalize_input=True)
    return cfg


gym.register(
    id=MANAGER_TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg:CartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": "leapp_test_tasks:manager_rsl_rl_cfg",
        "rsl_rl_distillation_cfg_entry_point": "leapp_test_tasks:manager_rsl_rl_distillation_cfg",
        "rl_games_cfg_entry_point": "leapp_test_tasks:manager_rl_games_cfg",
        "skrl_cfg_entry_point": f"{_AGENTS}:skrl_manager_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "leapp_test_tasks:manager_sb3_cfg",
    },
)
