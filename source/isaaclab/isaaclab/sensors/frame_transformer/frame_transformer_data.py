# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the base frame transformer data class for backwards compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_transformer_data import BaseFrameTransformerData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.frame_transformer import FrameTransformerData as PhysXFrameTransformerData


class FrameTransformerData(FactoryBase, BaseFrameTransformerData):
    """Factory for creating frame transformer data instances."""

    def __new__(cls, *args, **kwargs) -> BaseFrameTransformerData | PhysXFrameTransformerData:
        """Create a new instance of a frame transformer data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
