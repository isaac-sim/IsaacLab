# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-throughput benchmark variant of the MicroDuck flat walking task with a head camera."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Velocity-Flat-MicroDuck-Camera-Benchmark",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.microduck_camera_benchmark_env_cfg:MicroDuckCameraBenchmarkEnvCfg"},
)
