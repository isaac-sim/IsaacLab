# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Capture input configurations for the existing Franka Reach task."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401

# Reuse the existing environment and RL agents without changing their registration.
_franka_spec = gym.spec("Isaac-Reach-Franka")
gym.register(
    id="Isaac-Reach-Franka-Teleop",
    entry_point=_franka_spec.entry_point,
    disable_env_checker=_franka_spec.disable_env_checker,
    kwargs={
        **_franka_spec.kwargs,
        "env_cfg_entry_point": f"{__name__}.franka_env_cfg:FrankaTeleopEnvCfg",
    },
)
