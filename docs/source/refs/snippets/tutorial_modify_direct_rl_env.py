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
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_CFG

from isaaclab_tasks.core.locomotion.humanoid.humanoid_direct_env import HumanoidEnv
from isaaclab_tasks.core.locomotion.humanoid.humanoid_direct_env_cfg import HumanoidDirectSceneCfg, HumanoidEnvCfg
# [end-h1_env-import]

# [start-h1_env-robot]
@configclass
class H1SceneCfg(HumanoidDirectSceneCfg):
    robot: ArticulationCfg = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# [end-h1_env-robot]


# [start-h1_env-spaces]
@configclass
class H1EnvCfg(HumanoidEnvCfg):
    action_space = 19
    observation_space = 69
    scene: H1SceneCfg = H1SceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=True)
    # the gears are keyed by joint name expression, so they are independent of the joint ordering
    joint_gears: dict[str, float] = {".*": 50.0}


class H1Env(HumanoidEnv):
    cfg: H1EnvCfg
# [end-h1_env-spaces]

# fmt: on
