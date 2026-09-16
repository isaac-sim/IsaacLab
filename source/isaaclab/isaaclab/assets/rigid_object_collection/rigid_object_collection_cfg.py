# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .rigid_object_collection import RigidObjectCollection


@dataclass
class RigidObjectCollectionCfg:
    """Configuration parameters for a rigid object collection."""

    class_type: type["RigidObjectCollection"] | str = config_field(
        "{DIR}.rigid_object_collection:RigidObjectCollection"
    )
    """The associated asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """

    rigid_objects: dict[str, RigidObjectCfg] = config_field(MISSING)
    """Dictionary of rigid object configurations to spawn.

    The keys are the names for the objects, which are used as unique identifiers throughout the code.

    .. note::

       With the Newton backend, the configured root prim paths must share a combined prefix-and-suffix
       pattern that does not match other rigid objects in the same world.
    """
