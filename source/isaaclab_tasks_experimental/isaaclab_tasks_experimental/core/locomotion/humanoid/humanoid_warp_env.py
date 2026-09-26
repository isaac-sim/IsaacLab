# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.core.locomotion.humanoid.humanoid_direct_env_cfg import HumanoidEnvCfg

from isaaclab_tasks_experimental.core.locomotion.locomotion_env_warp import LocomotionWarpEnv


class HumanoidWarpEnv(LocomotionWarpEnv):
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
