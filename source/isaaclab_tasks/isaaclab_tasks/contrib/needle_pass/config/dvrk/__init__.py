# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the dVRK absolute-IK needle-pass environment."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-NeedlePass-dVRK-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:DVRKNeedlePassEnvCfg",
    },
    disable_env_checker=True,
)
