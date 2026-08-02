# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DR Legs closed-loop biped locomotion tasks (Kamino solver)."""

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec, registry

from . import agents

gym.register(
    id="IsaacContrib-DrLegs-HoldPose",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hold_pose_env_cfg:DrLegsHoldPoseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DrLegsHoldPosePPORunnerCfg",
    },
)

# Retain the released ID as a deprecated alias by inserting its spec directly.
registry.update(
    {
        "Isaac-DrLegs-HoldPose-v0": EnvSpec(
            id="Isaac-DrLegs-HoldPose-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.hold_pose_env_cfg:DrLegsHoldPoseEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DrLegsHoldPosePPORunnerCfg",
                "deprecated": {"alias": "--task IsaacContrib-DrLegs-HoldPose"},
            },
        ),
    }
)

gym.register(
    id="IsaacContrib-DrLegs-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:DrLegsWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DrLegsWalkPPORunnerCfg",
    },
)

registry.update(
    {
        "Isaac-DrLegs-Walk-v0": EnvSpec(
            id="Isaac-DrLegs-Walk-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.walk_env_cfg:DrLegsWalkEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DrLegsWalkPPORunnerCfg",
                "deprecated": {"alias": "--task IsaacContrib-DrLegs-Walk"},
            },
        ),
    }
)
