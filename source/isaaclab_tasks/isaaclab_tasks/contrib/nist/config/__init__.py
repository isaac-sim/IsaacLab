# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory task registration.

The assembly and physics backend are selected through presets, for example::

    uv run isaaclab train --task IsaacContrib-Factory-Franka \
        presets=peg_insert_4mm physics=newton_mjwarp

"""

import gymnasium as gym

from isaaclab_tasks.contrib.nist.config import agents

gym.register(
    id="IsaacContrib-Factory-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nist.factory_env_cfg:FactoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)
