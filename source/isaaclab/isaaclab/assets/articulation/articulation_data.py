# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_articulation_data import BaseArticulationData

if TYPE_CHECKING:
    from isaaclab_physx.assets.articulation.articulation_data import ArticulationData as PhysXArticulationData


class ArticulationData(FactoryBase):
    """Factory for creating articulation data instances."""

    def __new__(cls, *args, **kwargs) -> BaseArticulationData | PhysXArticulationData:
        """Create a new instance of an articulation data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
