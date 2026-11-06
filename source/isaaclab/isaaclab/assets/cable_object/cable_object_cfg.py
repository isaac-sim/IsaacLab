# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils import configclass
from ..asset_base_cfg import AssetBaseCfg

if TYPE_CHECKING:
    from .cable_object import CableObject


@configclass
class CableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a cable object.

    The inherited :attr:`init_state` sets the cable's spawn position and orientation.
    """

    class_type: type[CableObject] | str = "{DIR}.cable_object:CableObject"
