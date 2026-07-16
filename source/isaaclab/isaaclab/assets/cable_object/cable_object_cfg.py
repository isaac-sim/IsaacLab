# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from ..asset_base_cfg import AssetBaseCfg


@configclass
class CableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a cable object."""

    class_type: type | str = "isaaclab_contrib.cable.cable_object:CableObject"
