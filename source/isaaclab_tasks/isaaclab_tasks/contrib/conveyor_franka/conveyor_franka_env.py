# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based conveyor environment with Newton contact forces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import instantiate

from .conveyor_franka_env_cfg import ConveyorFrankaEnvCfg

if TYPE_CHECKING:
    from isaaclab.controllers import Conveyor


class ConveyorFrankaEnv(ManagerBasedRLEnv):
    """Manager-based environment with Newton conveyor surfaces."""

    cfg: ConveyorFrankaEnvCfg

    def __init__(self, cfg: ConveyorFrankaEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self.conveyor: Conveyor | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def load_managers(self) -> None:
        """Bind the fixed Panda tool frame before manager terms inspect it."""
        hand_ids, _ = self.scene["robot"].find_bodies("panda_hand")
        if len(hand_ids) != 1:
            raise ValueError(f"Expected one Panda hand body, found {len(hand_ids)}.")
        self._conveyor_hand_body_id = hand_ids[0]
        self._conveyor_tool_offset = torch.tensor((0.0, 0.0, 0.1034), dtype=torch.float32, device=self.device).expand(
            self.num_envs, -1
        )
        super().load_managers()

    def close(self) -> None:
        """Release conveyor callbacks before the physics scene is destroyed."""
        conveyor_driver = self.conveyor
        if conveyor_driver is not None:
            conveyor_driver.close()
            self.conveyor = None
        super().close()

    def _init_sim(self) -> None:
        """Register solved-contact hooks before the first reset captures the solver."""
        controller_cfg = self.cfg.conveyor.replace(
            surface_specs=self.cfg.scene.build_conveyor_belt_specs(
                velocity=self.cfg.belt_speed,
                friction_coefficient=self.cfg.belt_friction,
                contact_threshold=self.cfg.belt_contact_threshold,
            ),
        )
        driver = instantiate(controller_cfg, self.cfg.scene.num_envs, sim=self.sim)
        self.conveyor = driver
        try:
            super()._init_sim()
        except Exception:
            driver.close()
            self.conveyor = None
            raise

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Reset selected environments and discard stale conveyor forces."""
        if isinstance(env_ids, slice):
            env_ids = self.scene._ALL_INDICES[env_ids]
        super()._reset_idx(env_ids)

        conveyor_driver = self.conveyor
        if conveyor_driver is not None:
            conveyor_driver.reset(env_ids)
