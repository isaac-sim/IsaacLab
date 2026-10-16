# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register test-only tasks without importing simulation before the export launcher."""

import importlib
import sys

import gymnasium as gym

MANAGER_TASK = "Isaac-Leapp-Synthetic"
_AGENTS = "isaaclab_tasks.core.cartpole.agents"
gym.register(
    id=MANAGER_TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "leapp_synthetic_env_cfg:manager_env_cfg",
        "rsl_rl_cfg_entry_point": "leapp_synthetic_env_cfg:manager_rsl_rl_cfg",
        "rl_games_cfg_entry_point": "leapp_synthetic_env_cfg:manager_rl_games_cfg",
        "skrl_cfg_entry_point": f"{_AGENTS}:skrl_manager_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "leapp_synthetic_env_cfg:manager_sb3_cfg",
    },
)


if __name__ == "__main__":
    from isaaclab.test.utils import resolve_test_sim_device

    backend = importlib.import_module(f"isaaclab_rl.entrypoints.backends.export_{sys.argv[1]}")
    device = resolve_test_sim_device()
    args = [*sys.argv[2:], "--device", device]
    raise SystemExit(backend.run(args))
