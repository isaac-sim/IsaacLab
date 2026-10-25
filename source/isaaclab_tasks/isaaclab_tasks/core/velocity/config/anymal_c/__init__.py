# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native ANYmal C NeRD collection and velocity-tracking tasks."""

import gymnasium as gym

for suffix, config in (
    ("", "AnymalCNeRDEnvCfg"),
    ("-Explore", "AnymalCNeRDExplorationEnvCfg"),
    ("-Learned", "AnymalCNeRDLearnedEnvCfg"),
):
    gym.register(
        id=f"Isaac-Velocity-Flat-Anymal-C-NeRD{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.nerd_env_cfg:{config}",
            "nerd_cfg_entry_point": f"{__name__}:nerd_cfg.yaml",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
