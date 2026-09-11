# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-only benchmark scene for OVRTX / Newton-Warp comparison."""

import gymnasium as gym

gym.register(
    id="Isaac-RenderBenchmark-Franka-Cabinet",
    entry_point=f"{__name__}.render_benchmark_env:RenderBenchmarkEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_benchmark_env_cfg:RenderBenchmarkFrankaCabinetEnvCfg"},
)
