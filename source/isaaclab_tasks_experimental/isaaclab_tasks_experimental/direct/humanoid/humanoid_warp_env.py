# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks_experimental.direct.locomotion.locomotion_env_warp import LocomotionWarpEnv

from .humanoid_warp_env_cfg import HumanoidWarpEnvCfg


class HumanoidWarpEnv(LocomotionWarpEnv):
    cfg: HumanoidWarpEnvCfg

    def __init__(self, cfg: HumanoidWarpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
