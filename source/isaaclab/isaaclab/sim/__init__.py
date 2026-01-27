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

from .converters import *  # noqa: F401, F403
from .schemas import *  # noqa: F401, F403
from .simulation_cfg import PhysxCfg, RenderCfg, SimulationCfg  # noqa: F401, F403
from .simulation_context import SimulationContext, build_simulation_context  # noqa: F401, F403
from .simulation_manager import SimulationManager, IsaacEvents  # noqa: F401, F403
from .spawners import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .views import *  # noqa: F401, F403

# Monkey-patch Isaac Sim's SimulationManager to use Isaac Lab's implementation
# This ensures all code (including Isaac Sim internals) uses our manager
try:
    import isaacsim.core.simulation_manager as _isaacsim_sim_manager
    import isaacsim.core.simulation_manager.impl.simulation_manager as _isaacsim_sim_manager_impl

    # Get reference to the ORIGINAL Isaac Sim SimulationManager before patching
    _OriginalSimManager = _isaacsim_sim_manager_impl.SimulationManager

    # Disable all default callbacks from Isaac Sim's manager to prevent double-dispatch
    # These callbacks were already set up when the extension loaded, so we need to disable them
    _OriginalSimManager.enable_all_default_callbacks(False)

    # Replace the class in both the module and impl module
    _isaacsim_sim_manager.SimulationManager = SimulationManager
    _isaacsim_sim_manager.IsaacEvents = IsaacEvents
    _isaacsim_sim_manager_impl.SimulationManager = SimulationManager
    _isaacsim_sim_manager_impl.IsaacEvents = IsaacEvents
except ImportError:
    pass  # Isaac Sim extension not loaded yet, will be patched when available
