# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_cable_object import BaseCableObject
from .base_cable_object_data import BaseCableObjectData

if TYPE_CHECKING:
    from isaaclab_newton.assets.cable_object import CableObject as NewtonCableObject
    from isaaclab_newton.assets.cable_object import CableObjectData as NewtonCableObjectData


class CableObject(FactoryBase, BaseCableObject):
    """Factory for creating cable object instances."""

    data: BaseCableObjectData | NewtonCableObjectData

    def __new__(cls, *args, **kwargs) -> BaseCableObject | NewtonCableObject:
        """Create a cable object for the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
