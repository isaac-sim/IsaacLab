# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing simulation-specific functionalities.

These include:

* Ability to spawn different objects and materials into Omniverse
* Define and modify various schemas on USD prims
* Converters to obtain USD file from other file formats (such as URDF, OBJ, STL, FBX)
* Utility class to control the simulator

.. note::
    Currently, only a subset of all possible schemas and prims in Omniverse are supported.
    We are expanding the these set of functions on a need basis. In case, there are
    specific prims or schemas that you would like to include, please open an issue on GitHub
    as a feature request elaborating on the required application.

To make it convenient to use the module, we recommend importing the module as follows:

.. code-block:: python

    import isaaclab.sim as sim_utils

"""

import warnings

from .converters import *  # noqa: F401, F403
from .schemas import *  # noqa: F401, F403
from .simulation_cfg import RenderCfg, SimulationCfg  # noqa: F401, F403
from .simulation_context import SimulationContext, build_simulation_context  # noqa: F401, F403
from .spawners import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .views import *  # noqa: F401, F403

# Deprecated alias for PhysxCfg -> PhysxCfg
# This supports old code that uses `from isaaclab.sim import PhysxCfg`
try:
    from isaaclab_physx.physics import PhysxCfg as _PhysxCfg

    class PhysxCfg(_PhysxCfg):
        """DEPRECATED: Use PhysxCfg from isaaclab_physx.physics instead."""

        def __init__(self, *args, **kwargs):
            warnings.warn(
                "PhysxCfg is deprecated. Use PhysxCfg from isaaclab_physx.physics instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

except ImportError:
    # isaaclab_physx not installed
    PhysxCfg = None  # type: ignore
