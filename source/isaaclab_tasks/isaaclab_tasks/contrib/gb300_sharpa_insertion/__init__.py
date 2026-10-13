# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager-based GB300 Ethernet cable insertion with a dexterous hand."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Rizon-Sharpa-GB300-Cable-Install",
    entry_point=f"{__name__}.env:GB300SharpaInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GB300SharpaInsertionPresetsCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:GB300SharpaInsertionPPORunnerPresetsCfg",
        "default_agent": "rsl_rl",
    },
)
