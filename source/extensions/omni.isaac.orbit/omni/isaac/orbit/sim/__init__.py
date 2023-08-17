# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains utilities for simulation.

To make it convenient to use the module, we recommend importing the module as follows:

.. code-block:: python

    import omni.isaac.orbit.sim as sim_utils

"""

from .loaders import UrdfLoader, UrdfLoaderCfg
from .schemas import *  # noqa: F401, F403
from .simulation_cfg import PhysicsMaterialCfg, PhysxCfg, SimulationCfg
from .simulation_context import SimulationContext

__all__ = ["PhysicsMaterialCfg", "PhysxCfg", "SimulationCfg", "SimulationContext"]
