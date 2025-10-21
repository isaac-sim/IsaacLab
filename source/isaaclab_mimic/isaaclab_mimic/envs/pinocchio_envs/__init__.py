# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .exhaustpipe_gr1t2_mimic_env_cfg import ExhaustPipeGR1T2MimicEnvCfg
from .nutpour_gr1t2_mimic_env_cfg import NutPourGR1T2MimicEnvCfg
from .pickplace_gr1t2_mimic_env import PickPlaceGR1T2MimicEnv
from .pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
from .pickplace_gr1t2_waist_enabled_mimic_env_cfg import PickPlaceGR1T2WaistEnabledMimicEnvCfg

gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_gr1t2_mimic_env_cfg.PickPlaceGR1T2MimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_gr1t2_waist_enabled_mimic_env_cfg.PickPlaceGR1T2WaistEnabledMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-NutPour-GR1T2-Pink-IK-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs:PickPlaceGR1T2MimicEnv",
    kwargs={"env_cfg_entry_point": nutpour_gr1t2_mimic_env_cfg.NutPourGR1T2MimicEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs:PickPlaceGR1T2MimicEnv",
    kwargs={"env_cfg_entry_point": exhaustpipe_gr1t2_mimic_env_cfg.ExhaustPipeGR1T2MimicEnvCfg},
    disable_env_checker=True,
)
