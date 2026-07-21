# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_cable_object_data import BaseCableObjectData

if TYPE_CHECKING:
    from isaaclab_newton.assets.cable_object import CableObjectData as NewtonCableObjectData


class CableObjectData(FactoryBase):
    """Factory for creating cable object data instances."""

    def __new__(cls, *args, **kwargs) -> BaseCableObjectData | NewtonCableObjectData:
        """Create cable object data for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
