# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_transformer import BaseFrameTransformer
from .base_frame_transformer_data import BaseFrameTransformerData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.frame_transformer import FrameTransformer as PhysXFrameTransformer
    from isaaclab_physx.sensors.frame_transformer import FrameTransformerData as PhysXFrameTransformerData


class FrameTransformer(FactoryBase, BaseFrameTransformer):
    """Factory for creating frame transformer instances."""

    data: BaseFrameTransformerData | PhysXFrameTransformerData

    def __new__(cls, *args, **kwargs) -> BaseFrameTransformer | PhysXFrameTransformer:
        """Create a new instance of a frame transformer based on the backend."""
        return super().__new__(cls, *args, **kwargs)
