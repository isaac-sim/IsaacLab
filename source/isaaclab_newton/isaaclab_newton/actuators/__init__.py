# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native actuator integration for Isaac Lab.

Public API surface:

* :class:`~isaaclab_newton.actuators.adapter.NewtonActuatorAdapter` —
  the actuator adapter used by both backends. The Newton backend
  constructs it directly from ``model.actuators``; the PhysX backend
  uses :meth:`~NewtonActuatorAdapter.from_usd` to build the actuators
  from authored ``NewtonActuator`` USD prims.
* :class:`~isaaclab_newton.actuators.physx_wrapper.PhysxActuatorWrapper`
  — flat-array wrapper that satisfies the Newton actuator
  ``sim_state`` / ``sim_control`` protocol on the PhysX backend.
* :func:`~isaaclab_newton.actuators.kernels.build_implicit_dof_mask` —
  builds the per-DOF implicit-actuator mask consumed by the in-graph
  post-actuator kernel.

USD authoring lives on the schema side as
:func:`~isaaclab.sim.schemas.define_actuator_properties`; both backends
call into it via :meth:`ArticulationCfg._post_spawn`.
"""

from .adapter import NewtonActuatorAdapter, build_newton_actuator_defaults
from .kernels import build_implicit_dof_mask
from .physx_wrapper import PhysxActuatorWrapper

__all__ = [
    "NewtonActuatorAdapter",
    "PhysxActuatorWrapper",
    "build_implicit_dof_mask",
    "build_newton_actuator_defaults",
]
