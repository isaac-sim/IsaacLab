# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

###########################
# Register Gym environments
###########################

# The same preset selects the matching environment space and SKRL model config.
gym.register(
    id="IsaacContrib-Cartpole-Showcase-Direct",
    entry_point=f"{__name__}.cartpole_env:CartpoleShowcaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleShowcasePresetsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg",
    },
)
