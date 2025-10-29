# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .anymal_c_multi_agent_bar_env import AnymalCMultiAgentBar, AnymalCMultiAgentBarEnvCfg
from .h1_anymal_push_env import HeterogeneousPushMultiAgentEnv, HeterogeneousPushMultiAgentEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Multi-Agent-Anymal-C-Bar-Direct-v0",
    entry_point=AnymalCMultiAgentBar,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCMultiAgentBarEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Multi-Agent-H1-Anymal-C-Push-Direct-v0",
    entry_point=HeterogeneousPushMultiAgentEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousPushMultiAgentEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)
