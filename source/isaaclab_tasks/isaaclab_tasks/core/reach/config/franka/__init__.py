# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec, registry

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Reach-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# Newton Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reach-Franka-Newton-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_newton_env_cfg:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachNewtonIKPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Gymnasium rejects versioned IDs when their versionless replacements are registered,
# so retain the released IDs by inserting their deprecation specs directly.
registry.update(
    {
        "Isaac-Reach-Franka-Newton-IK-Rel-v0": EnvSpec(
            id="Isaac-Reach-Franka-Newton-IK-Rel-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ik_newton_env_cfg:FrankaReachEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachNewtonIKPPORunnerCfg",
                "deprecated": {"alias": "--task Isaac-Reach-Franka-Newton-IK-Rel"},
            },
            disable_env_checker=True,
        ),
    }
)

##
# Operational Space Control
##

gym.register(
    id="Isaac-Reach-Franka-OSC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)
