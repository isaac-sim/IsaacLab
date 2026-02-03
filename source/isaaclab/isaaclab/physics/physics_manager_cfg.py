# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .physics_manager import PhysicsManager


@configclass
class PhysicsManagerCfg:
    """Abstract base configuration for physics managers.

    Subclasses should override :meth:`create_manager` to return the appropriate
    physics manager class.
    """

    def create_manager(self) -> type["PhysicsManager"]:
        """Create and return the physics manager class for this configuration.

        Returns:
            The physics manager class (not an instance).

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement create_manager()")
