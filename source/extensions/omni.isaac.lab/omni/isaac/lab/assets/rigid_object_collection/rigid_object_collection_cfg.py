# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg

from .rigid_object_collection import RigidObjectCollection


@configclass
class RigidObjectCollectionCfg:
    """Configuration parameters for a rigid object collection."""


    class_type: type = RigidObjectCollection
    """The associated asset class.

    The class should inherit from :class:`omni.isaac.lab.assets.asset_base.AssetBase`.
    """

    rigid_objects: dict[str, RigidObjectCfg] = MISSING
    """Dictionary of rigid object configurations to spawn.
    
    The keys are the names of the objects.
    """

