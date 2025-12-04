# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class LocomotionFixedRootCmdRetargeter(RetargeterBase):
    """Provides fixed root commands (velocity zero, fixed hip height).

    Useful for standing still or when motion controllers are not available but the pipeline expects
    locomotion commands.
    """

    def __init__(self, cfg: LocomotionFixedRootCmdRetargeterCfg):
        """Initialize the retargeter."""
        super().__init__(cfg)
        self.cfg = cfg

    def retarget(self, data: dict) -> torch.Tensor:
        # Returns [vel_x, vel_y, rot_vel_z, hip_height]
        return torch.tensor([0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        # This retargeter does not consume any device data
        return []


@dataclass
class LocomotionFixedRootCmdRetargeterCfg(RetargeterCfg):
    """Configuration for the fixed locomotion root command retargeter."""

    hip_height: float = 0.72
    """Height of the robot hip in meters."""

    retargeter_type: type[RetargeterBase] = LocomotionFixedRootCmdRetargeter
