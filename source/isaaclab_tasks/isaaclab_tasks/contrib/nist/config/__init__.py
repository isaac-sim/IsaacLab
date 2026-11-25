# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NIST task registration.

The assembly variant is selected at reset, for example::

    uv run isaaclab train --task IsaacContrib-NIST-Franka

"""

import gymnasium as gym

from isaaclab_tasks.contrib.nist.config import agents

gym.register(
    id="IsaacContrib-NIST-Franka",
    entry_point="isaaclab_tasks.contrib.nist.nist_env:NISTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nist.nist_env_cfg:NISTEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NISTPPORunnerCfg",
    },
)
