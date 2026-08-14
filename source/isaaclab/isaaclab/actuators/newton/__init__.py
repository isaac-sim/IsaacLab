# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native actuator integration for Isaac Lab.

Public API surface:

* :class:`~isaaclab.actuators.newton.adapter.NewtonActuatorAdapter` —
  the actuator adapter used by Newton and the host adapters. Newton
  constructs it directly from ``model.actuators``; PhysX and OVPhysX use
  :meth:`~NewtonActuatorAdapter.from_usd` to build actuators from authored
  ``NewtonActuator`` USD prims.
* :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
  — flat-array wrapper that satisfies the Newton actuator
  ``sim_state`` / ``sim_control`` protocol on PhysX and OVPhysX.
* :func:`~isaaclab.actuators.newton.kernels.build_implicit_dof_mask` —
  builds the per-DOF implicit-actuator mask consumed by the in-graph
  post-actuator kernel.

USD authoring lives on the schema side as
:func:`~isaaclab.sim.schemas.define_actuator_properties`; each backend calls
it through :meth:`ArticulationCfg._post_spawn`.
"""

from .adapter import NewtonActuatorAdapter
from .kernels import build_implicit_dof_mask
from .physx_wrapper import PhysxActuatorWrapper

__all__ = [
    "NewtonActuatorAdapter",
    "PhysxActuatorWrapper",
    "build_implicit_dof_mask",
]
