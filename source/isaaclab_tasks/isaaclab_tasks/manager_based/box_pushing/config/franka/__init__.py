# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, joint_effort_env_cfg

_VARIANTS = {
    "": {
        "Dense": joint_effort_env_cfg.FrankaBoxPushingEnvCfg_Dense,
        "TemporalSparse": joint_effort_env_cfg.FrankaBoxPushingEnvCfg_TemporalSparse,
    },
    "NoIK-": {
        "Dense": joint_effort_env_cfg.FrankaBoxPushingNoIKEnvCfg_Dense,
        "TemporalSparse": joint_effort_env_cfg.FrankaBoxPushingNoIKEnvCfg_TemporalSparse,
    },
}

for variant_prefix, reward_cfgs in _VARIANTS.items():
    for reward_name, env_cfg_class in reward_cfgs.items():
        for rl_type in ["step", "bbrl"]:
            gym.register(
                id=f"Isaac-Box-Pushing-{variant_prefix}{reward_name}-{rl_type}-Franka-v0",
                entry_point="isaaclab_tasks.manager_based.box_pushing.box_pushing_env:BoxPushingEnv",
                kwargs={
                    "env_cfg_entry_point": env_cfg_class,
                    "rsl_rl_cfg_entry_point": getattr(agents.rsl_rl_cfg, f"BoxPushingPPORunnerCfg_{rl_type}"),
                    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
                    "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg_{rl_type}.yaml",
                },
                disable_env_checker=True,
            )
