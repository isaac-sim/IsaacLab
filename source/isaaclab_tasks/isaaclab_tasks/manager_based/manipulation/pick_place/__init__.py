# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_env_cfg:PickPlaceGR1T2EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nutpour_gr1t2_pink_ik_env_cfg:NutPourGR1T2PinkIKEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_nut_pouring.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exhaustpipe_gr1t2_pink_ik_env_cfg:ExhaustPipeGR1T2PinkIKEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_exhaust_pipe.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_waist_enabled_env_cfg:PickPlaceGR1T2WaistEnabledEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-G1-InspireFTP-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_unitree_g1_inspire_hand_env_cfg:PickPlaceG1InspireFTPEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)
