# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-dispatching surface-gripper asset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils.backend_utils import FactoryBase
from .base_surface_gripper import BaseSurfaceGripper

if TYPE_CHECKING:
    from ...sim import SimulationContext
    from .surface_gripper_cfg import SurfaceGripperCfg


class SurfaceGripper(FactoryBase, BaseSurfaceGripper):
    """Create the active backend's surface gripper.

    Configure this asset with :class:`SurfaceGripperCfg` and access it through
    :attr:`~isaaclab.scene.InteractiveScene.surface_grippers`. See
    :class:`BaseSurfaceGripper` for command, measurement and lifecycle contracts.

    Newton supports one four-pad tool per replicated world and single-box payloads.
    It automatically acquires an eligible payload unless ``set_target_ids`` restricts
    acquisition. PhysX uses surface-gripper prims already authored in the robot USD.
    """

    def __new__(cls, cfg: SurfaceGripperCfg, *, sim: SimulationContext | None = None) -> BaseSurfaceGripper:
        """Select the implementation using the supplied or active simulation context."""
        return super().__new__(cls, cfg, sim=sim)
