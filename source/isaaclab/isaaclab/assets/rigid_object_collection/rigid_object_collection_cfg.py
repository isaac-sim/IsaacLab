# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .rigid_object_collection import RigidObjectCollection


@configclass
class RigidObjectCollectionCfg:
    """Configuration parameters for a rigid object collection."""

    class_type: type["RigidObjectCollection"] | str = "{DIR}.rigid_object_collection:RigidObjectCollection"
    """The associated asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """

    rigid_objects: dict[str, RigidObjectCfg] = MISSING
    """Dictionary of rigid object configurations to spawn.

    The keys are the names for the objects, which are used as unique identifiers throughout the code.
    """
