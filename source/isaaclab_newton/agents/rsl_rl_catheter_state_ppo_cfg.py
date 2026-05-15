# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl PPO configuration for the state-based catheter navigation task.

Returns a plain dict compatible with ``rsl_rl.runners.OnPolicyRunner``,
avoiding the ``isaaclab_rl`` import chain which pulls in Isaac Sim.
"""


def get_runner_cfg() -> dict:
    """Return rsl_rl OnPolicyRunner configuration as a dict."""
    return {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": 1500,
        "save_interval": 100,
        "experiment_name": "catheter_state",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "empirical_normalization": None,
        "obs_groups": {
            "policy": ["policy"],
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
