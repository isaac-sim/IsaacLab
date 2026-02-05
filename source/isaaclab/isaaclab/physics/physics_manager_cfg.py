# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import MISSING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .physics_manager import PhysicsManager


@configclass
class PhysicsManagerCfg:
    """Abstract base configuration for physics managers.

    This base class contains common simulation parameters shared across
    all physics backends. Subclasses should override :meth:`create_manager`
    to return the appropriate physics manager class.
    """

    # ------------------------------------------------------------------
    # Common Simulation Parameters
    # ------------------------------------------------------------------

    class_type: type[PhysicsManager] = MISSING

    dt: float = 1.0 / 60.0
    """The physics simulation time-step (in seconds). Default is 0.0167 seconds."""

    device: str = "cuda:0"
    """The device to run the simulation on. Default is ``"cuda:0"``.

    Valid options are:

    - ``"cpu"``: Use CPU.
    - ``"cuda"``: Use GPU, where the device ID is inferred from config.
    - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
    """

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81)."""

    physics_prim_path: str = "/physicsScene"
    """The prim path where the USD PhysicsScene is created. Default is "/physicsScene"."""
