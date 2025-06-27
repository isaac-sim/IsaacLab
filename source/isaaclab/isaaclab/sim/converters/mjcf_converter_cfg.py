# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.utils import configclass


@configclass
class MjcfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MjcfConverter."""

    link_density = 0.0
    """Default density used for links. Defaults to 0.

    This setting is only effective if ``"inertial"`` properties are missing in the MJCF.
    """

    import_inertia_tensor: bool = True
    """Import the inertia tensor from mjcf. Defaults to True.

    If the ``"inertial"`` tag is missing, then it is imported as an identity.
    """

    fix_base: bool = MISSING
    """Create a fix joint to the root/base link. Defaults to True."""

    import_sites: bool = True
    """Import the sites from the MJCF. Defaults to True."""

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""
