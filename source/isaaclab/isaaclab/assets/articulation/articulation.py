# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils.backend_utils import FactoryBase
from .base_articulation import BaseArticulation
from .base_articulation_data import BaseArticulationData

if TYPE_CHECKING:
    from isaaclab_physx.assets.articulation import Articulation as PhysXArticulation
    from isaaclab_physx.assets.articulation import ArticulationData as PhysXArticulationData


class Articulation(FactoryBase, BaseArticulation):
    """Factory for creating articulation instances."""

    data: BaseArticulationData | PhysXArticulationData

    def __new__(cls, *args, **kwargs) -> BaseArticulation | PhysXArticulation:
        """Create a new instance of an articulation based on the backend."""
        return super().__new__(cls, *args, **kwargs)
