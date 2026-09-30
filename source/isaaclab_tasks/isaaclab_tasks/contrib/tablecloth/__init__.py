# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 tablecloth task registration."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Tablecloth-H1",
    entry_point=f"{__name__}.h1_env:H1TableclothEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.h1_env_cfg:H1TableclothEnvCfg"},
)
