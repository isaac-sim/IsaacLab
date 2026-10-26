# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 reference-motion tracking with native environment managers."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Motion-Tracking-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_tracking_env_cfg:G1TrackingEnvCfg",
        "default_agent": "supermimic",
        "supermimic_cfg_entry_point": "isaaclab_rl.supermimic.supermimic_cfg:SuperMimicCfg",
    },
)
