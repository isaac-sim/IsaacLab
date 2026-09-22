# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the base PVA data class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_pva_data import BasePvaData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.pva import PvaData as NewtonPvaData
    from isaaclab_physx.sensors.pva import PvaData as PhysXPvaData


class PvaData(FactoryBase, BasePvaData):
    """Factory for creating PVA data instances."""

    def __new__(cls, *args, **kwargs) -> BasePvaData | NewtonPvaData | PhysXPvaData:
        """Create a new instance of PVA data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
