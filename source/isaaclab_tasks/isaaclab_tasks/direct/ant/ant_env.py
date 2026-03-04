# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from .ant_env_cfg import AntEnvCfg


class AntEnv(LocomotionEnv):
    cfg: AntEnvCfg

    def __init__(self, cfg: AntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
