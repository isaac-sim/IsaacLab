# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module provides the ``SimulationContext`` class.

The :class:`SimulationContext` inherits from the :class:`omni.isaac.core.simulation_context.SimulationContext` class
to provide additional functionality for the Orbit extension. This includes configuring the simulation through
the configuration class :class:`SimulationCfg` and providing a context manager for the simulation.
"""

from .simulation_cfg import PhysicsMaterialCfg, PhysxCfg, SimulationCfg
from .simulation_context import SimulationContext

__all__ = ["PhysicsMaterialCfg", "PhysxCfg", "SimulationCfg", "SimulationContext"]
