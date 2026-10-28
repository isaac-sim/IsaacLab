# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils.backend_utils import FactoryBase
from .base_pva import BasePva
from .pva_data import PvaData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.pva import Pva as NewtonPva
    from isaaclab_physx.sensors.pva import Pva as PhysXPva


class Pva(FactoryBase, BasePva):
    """Factory for creating PVA sensor instances."""

    data: PvaData

    def __new__(cls, *args, **kwargs) -> BasePva | NewtonPva | PhysXPva:
        """Create a new instance of a PVA sensor based on the backend."""
        return super().__new__(cls, *args, **kwargs)
