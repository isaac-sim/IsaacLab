# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class G1LowerBodyStandingRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body standing retargeter."""

    hip_height: float = 0.72
    """Height of the G1 robot hip in meters. The value is a fixed height suitable for G1 to do tabletop manipulation."""


class G1LowerBodyStandingRetargeter(RetargeterBase):
    """Provides lower body standing commands for the G1 robot."""

    def __init__(self, cfg: G1LowerBodyStandingRetargeterCfg):
        """Initialize the retargeter."""
        self.cfg = cfg

    def retarget(self, data: dict) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device)
