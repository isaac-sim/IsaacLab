# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa
# fmt: off

# [start-init-import]
from .h1_env import H1Env, H1EnvCfg
# [end-init-import]

# [start-init-register]
gym.register(
    id="Isaac-H1-Direct-v0",
    entry_point="isaaclab_tasks.core.locomotion.humanoid:H1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
# [end-init-register]

# [start-h1_env-import]
from isaaclab_assets import H1_CFG
# [end-h1_env-import]

# [start-h1_env-spaces]
action_space = 19
observation_space = 69
# [end-h1_env-spaces]

# [start-h1_env-robot]
robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
# the gears are keyed by joint name expression, so they are independent of the joint ordering
joint_gears: dict[str, float] = {".*": 50.0}
# [end-h1_env-robot]

# fmt: on
