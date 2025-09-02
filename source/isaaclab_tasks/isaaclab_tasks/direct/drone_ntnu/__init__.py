# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
from .state_based_control import agents as sbc_agents

##
# Register Gym environments.

gym.register(
    id="Isaac-Drone-NTNU-Direct-v0",
    entry_point=f"{__name__}.state_based_control.drone_env:DroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.state_based_control.drone_env_cfg:DroneEnvCfg",
        "rl_games_cfg_entry_point": f"{sbc_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{sbc_agents.__name__}.rsl_rl_ppo_cfg:DronePPORunnerCfg",
    },
)
