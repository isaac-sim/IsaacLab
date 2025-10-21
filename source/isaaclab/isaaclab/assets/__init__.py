# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different assets, such as rigid objects and articulations.

An asset is a physical object that can be spawned in the simulation. The class handles both
the spawning of the asset into the USD stage as well as initialization of necessary physics
handles to interact with the asset.

Upon construction of the asset instance, the prim corresponding to the asset is spawned into the
USD stage if the spawn configuration is not None. The spawn configuration is defined in the
:attr:`AssetBaseCfg.spawn` attribute. In case the configured :attr:`AssetBaseCfg.prim_path` is
an expression, then the prim is spawned at all the matching paths. Otherwise, a single prim is
spawned at the configured path. For more information on the spawn configuration, see the
:mod:`isaaclab.sim.spawners` module.

The asset class also registers callbacks for the stage play/stop events. These are used to
construct the physics handles for the asset as the physics engine is only available when the
stage is playing. Additionally, the class registers a callback for debug visualization of the
asset. This can be enabled by setting the :attr:`AssetBaseCfg.debug_vis` attribute to True.

The asset class follows the following naming convention for its methods:

* **set_xxx()**: These are used to only set the buffers into the :attr:`data` instance. However, they
  do not write the data into the simulator. The writing of data only happens when the
  :meth:`write_data_to_sim` method is called.
* **write_xxx_to_sim()**: These are used to set the buffers into the :attr:`data` instance and write
  the corresponding data into the simulator as well.
* **update(dt)**: These are used to update the buffers in the :attr:`data` instance. This should
  be called after a simulation step is performed.

The main reason to separate the ``set`` and ``write`` operations is to provide flexibility to the
user when they need to perform a post-processing operation of the buffers before applying them
into the simulator. A common example for this is dealing with explicit actuator models where the
specified joint targets are not directly applied to the simulator but are instead used to compute
the corresponding actuator torques.
"""

from .articulation import Articulation, ArticulationCfg, ArticulationData
from .asset_base import AssetBase
from .asset_base_cfg import AssetBaseCfg
from .deformable_object import DeformableObject, DeformableObjectCfg, DeformableObjectData
from .rigid_object import RigidObject, RigidObjectCfg, RigidObjectData
from .rigid_object_collection import RigidObjectCollection, RigidObjectCollectionCfg, RigidObjectCollectionData
from .surface_gripper import SurfaceGripper, SurfaceGripperCfg
