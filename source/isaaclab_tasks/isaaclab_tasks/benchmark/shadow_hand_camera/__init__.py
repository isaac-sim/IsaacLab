# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-throughput benchmark variant of the Shadow Hand camera task."""

import gymnasium as gym

from isaaclab_tasks.core.reorient.config.shadow_hand import agents as core_agents

gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-Camera-Benchmark-Direct",
    entry_point="isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_camera_benchmark_env_cfg:ShadowHandCameraBenchmarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{core_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{core_agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)
