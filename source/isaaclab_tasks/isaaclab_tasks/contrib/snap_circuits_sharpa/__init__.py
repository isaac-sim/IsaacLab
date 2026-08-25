# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apple Vision Pro Snap Circuits demo using the bimanual Sharpa Wave hand."""

import gymnasium as gym


gym.register(
    id="IsaacContrib-SnapCircuits-SharpaWave-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snap_circuits_sharpa_env_cfg:SnapCircuitsSharpaEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-SnapCircuits-ProHand-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snap_circuits_prohand_env_cfg:SnapCircuitsProHandEnvCfg",
    },
    disable_env_checker=True,
)
