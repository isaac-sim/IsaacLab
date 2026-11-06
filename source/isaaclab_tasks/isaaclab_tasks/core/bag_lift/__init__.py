# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers the Isaac-BagLift-Direct gym task (Isaac Lab imports this folder automatically).

Train with:  isaaclab train --rl_library rl_games --task=Isaac-BagLift-Direct
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BagLift-Direct",
    entry_point=f"{__name__}.bag_lift_env:BagLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bag_lift_env_cfg:BagLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_direct_ppo_cfg.yaml",
    },
)
