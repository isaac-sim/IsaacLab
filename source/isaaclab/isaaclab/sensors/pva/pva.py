# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_pva import BasePva
from .base_pva_data import BasePvaData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.pva import Pva as PhysXPva
    from isaaclab_physx.sensors.pva import PvaData as PhysXPvaData


class Pva(FactoryBase, BasePva):
    """Factory for creating PVA sensor instances."""

    data: BasePvaData | PhysXPvaData

    def __new__(cls, *args, **kwargs) -> BasePva | PhysXPva:
        """Create a new instance of a PVA sensor based on the backend."""
        return super().__new__(cls, *args, **kwargs)
