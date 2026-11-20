# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apple pick-and-place with H2 + Sharpa."""

import gymnasium as gym

from . import agents

for task_id in ("IsaacContrib-Pick-And-Place-Apple-H2-Sharpa", "IsaacContrib-Pick-And-Place-Apple-H2-Sharpa-Eval"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pick_and_place_apple_env_cfg:PickAndPlaceAppleEnvCfg",
            "rlinf_cfg_entry_point": f"{agents.__name__}:rlinf_gr00t_pick_and_place_apple_ppo_cfg.yaml",
        },
    )
