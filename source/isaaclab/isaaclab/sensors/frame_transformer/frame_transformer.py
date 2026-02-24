# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_frame_transformer import BaseFrameTransformer

if TYPE_CHECKING:
    from isaaclab_newton.sensors.frame_transformer import FrameTransformer as NewtonFrameTransformer


class FrameTransformer(FactoryBase):
    """Factory for creating frame transformer sensor instances."""

    def __new__(cls, *args, **kwargs) -> BaseFrameTransformer | NewtonFrameTransformer:
        """Create a new instance of a frame transformer based on the backend."""
        return super().__new__(cls, *args, **kwargs)
