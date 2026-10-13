# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 terrain traversals with pretrained locomotion policies."""

import gymnasium as gym

for terrain, config in (
    ("Platforms", "G1PlatformsEnvCfg"),
    ("Ramps", "G1RampsEnvCfg"),
    ("Granular", "G1GranularEnvCfg"),
):
    gym.register(
        id=f"IsaacContrib-G1-Terrain-{terrain}",
        entry_point=f"{__name__}.terrain_env:G1TerrainEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.terrain_env_cfg:{config}"},
    )
