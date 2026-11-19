# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 medical-pad unfolding with Cosmos3 visual dynamics."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Cosmos3-Medical-Pad-Unfold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.medical_pad_unfold_env_cfg:MedicalPadUnfoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MedicalPadPPORunnerCfg",
    },
)
