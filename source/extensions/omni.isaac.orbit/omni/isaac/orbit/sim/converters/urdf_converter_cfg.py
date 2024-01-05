# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing_extensions import Literal

from omni.isaac.orbit.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from omni.isaac.orbit.utils import configclass


@configclass
class UrdfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for UrdfConverter."""

    link_density = 0.0
    """Default density used for links. Defaults to 0.

    This setting is only effective if ``"inertial"`` properties are missing in the URDF.
    """

    import_inertia_tensor: bool = True
    """Import the inertia tensor from urdf. Defaults to True.

    If the ``"inertial"`` tag is missing, then it is imported as an identity.
    """

    convex_decompose_mesh = False
    """Decompose a convex mesh into smaller pieces for a closer fit. Defaults to False."""

    fix_base: bool = MISSING
    """Create a fix joint to the root/base link. Defaults to True."""

    merge_fixed_joints: bool = False
    """Consolidate links that are connected by fixed joints. Defaults to False."""

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""

    default_drive_type: Literal["none", "position", "velocity"] = "none"
    """The drive type used for joints. Defaults to ``"none"``.

    The drive type dictates the loaded joint PD gains and USD attributes for joint control:

    * ``"none"``: The joint stiffness and damping are set to 0.0.
    * ``"position"``: The joint stiff and damping are set based on the URDF file or provided configuration.
    * ``"velocity"``: The joint stiff is set to zero and damping is based on the URDF file or provided configuration.
    """

    default_drive_stiffness: float = 0.0
    """The default stiffness of the joint drive. Defaults to 0.0."""

    default_drive_damping: float = 0.0
    """The default damping of the joint drive. Defaults to 0.0.

    Note:
        If set to zero, the values parsed from the URDF joint tag ``"<dynamics><damping>"`` are used.
        Otherwise, it is overridden by the configured value.
    """
