# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for surface-gripper assets."""

from typing import TYPE_CHECKING

from ...utils import configclass
from ..asset_base_cfg import AssetBaseCfg

if TYPE_CHECKING:
    from .surface_gripper import SurfaceGripper


@configclass
class SurfaceGripperCfg(AssetBaseCfg):
    """Configure a surface gripper using the active physics backend.

    On Newton, ``prim_path`` selects the rigid tool body and the tool geometry below
    defines four compliant pads with fixed force and seal thresholds. Payloads must
    have exactly one collider, of box type.
    PhysX uses a surface-gripper prim and the four optional PhysX properties below.
    """

    class_type: type["SurfaceGripper"] | str = "{DIR}.surface_gripper:SurfaceGripper"
    cloning_contexts: tuple[type | str, ...] | None = ()

    max_grip_distance: float | None = None
    """PhysX maximum grip distance [m]; ``None`` preserves the authored value."""

    coaxial_force_limit: float | None = None
    """PhysX axial force limit [N]; ``None`` preserves the authored value."""

    shear_force_limit: float | None = None
    """PhysX shear force limit [N]; ``None`` preserves the authored value."""

    retry_interval: float | None = None
    """PhysX acquisition retry interval [s]; ``None`` preserves the authored value."""

    target_prim_path: str = ".*"
    """Newton payload-body expression. All eligible boxes are candidates by default."""

    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Newton pad-center translation in the tool frame [m]; local +Z points toward payloads."""

    pad_spacing: float = 0.007
    """Newton pad-center offsets from the tool center along local X and Y [m]."""

    pad_radius: float = 0.004
    """Newton pad radius [m]."""
