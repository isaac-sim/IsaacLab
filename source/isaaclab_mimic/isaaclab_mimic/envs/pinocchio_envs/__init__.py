# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .pickplace_gr1t2_mimic_env import PickPlaceGR1T2MimicEnv
from .pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
from .stack_so100_mimic_env import StackSO100MimicEnv
from .stack_so100_mimic_env_cfg import StackSO100MimicEnvCfg

gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_gr1t2_mimic_env_cfg.PickPlaceGR1T2MimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-SO100-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs:StackSO100MimicEnv",
    kwargs={
        "env_cfg_entry_point": stack_so100_mimic_env_cfg.StackSO100MimicEnvCfg,
    },
    disable_env_checker=True,
)
