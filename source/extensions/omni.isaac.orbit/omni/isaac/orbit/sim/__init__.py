# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains simulation-specific functionalities.

These include:

* Ability to spawn different objects and materials into Omniverse
* Define and modify various schemas on USD prims
* Loaders to obtain USD file from other file formats (such as URDF)
* Utility class to control the simulator

.. note::
    Currently, only a subset of all possible schemas and prims in Omniverse are supported.
    We are expanding the these set of functions on a need basis. In case, there are
    specific prims or schemas that you would like to include, please open an issue on GitHub
    as a feature request elaborating on the required application.

To make it convenient to use the module, we recommend importing the module as follows:

.. code-block:: python

    import omni.isaac.orbit.sim as sim_utils

"""

from .loaders import *  # noqa: F401, F403
from .schemas import *  # noqa: F401, F403
from .simulation_cfg import PhysxCfg, SimulationCfg  # noqa: F401, F403
from .simulation_context import SimulationContext  # noqa: F401, F403
from .spawners import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
