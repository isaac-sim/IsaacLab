# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment that installs the task-local conveyor force driver."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn

from .conveyor_force_driver import ConveyorForceDriver
from .conveyor_franka_env_cfg import ConveyorFrankaEnvCfg


class ConveyorFrankaEnv(ManagerBasedRLEnv):
    """Manager-based environment with force-driven Newton conveyor surfaces."""

    cfg: ConveyorFrankaEnvCfg

    def __init__(self, cfg: ConveyorFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._conveyor_driver = ConveyorForceDriver(
            num_envs=self.num_envs,
            speed=cfg.conveyor_force.speed,
            friction=cfg.conveyor_force.friction,
            normal_threshold=cfg.conveyor_force.normal_threshold,
        )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the manager-based environment and prepare traction for the next step."""
        result = super().step(action)
        # The contact sensor has now asked Newton to publish per-contact forces.
        self._conveyor_driver.update()
        return result

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset selected environments and discard stale conveyor forces."""
        super()._reset_idx(env_ids)

        conveyor_driver = getattr(self, "_conveyor_driver", None)
        if conveyor_driver is not None:
            conveyor_driver.clear()
