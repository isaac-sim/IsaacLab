# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AGX Orin packing with H2 + Sharpa."""

import gymnasium as gym

from . import agents

for task_id in ("IsaacContrib-Pack-AGX-Orin-H2-Sharpa", "IsaacContrib-Pack-AGX-Orin-H2-Sharpa-Eval"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pack_agx_env_cfg:PackAgxOrinEnvCfg",
            "rlinf_cfg_entry_point": f"{agents.__name__}:rlinf_gr00t_pack_agx_ppo_cfg.yaml",
        },
    )
