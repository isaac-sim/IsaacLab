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
* :func:`~isaaclab.actuators.newton.adapter.read_group_parameter` /
  :func:`~isaaclab.actuators.newton.adapter.write_group_parameter` —
  group-scoped, user-ordered access to Newton actuator parameters through
  the selection API; the raw alternative is the Newton actuator object
  returned by the actuator collection mapping.
* :class:`~isaaclab.actuators.newton.bam_component.ControllerBam` — the
  Newton-native BAM servo model. Importing this package registers it under
  the ``NewtonBamControlAPI`` USD schema token.

USD authoring lives on the schema side as
:func:`~isaaclab.sim.schemas.define_actuator_properties`; each backend calls
it through :meth:`ArticulationCfg._post_spawn`.
"""

from .adapter import NewtonActuatorAdapter, read_group_parameter, write_group_parameter
from .bam_component import (
    BAM_CONTROL_API,
    ControllerBam,
    apply_bam_startup_sampling,
    register_bam_actuator_component,
)
from .kernels import build_implicit_dof_mask
from .physx_wrapper import PhysxActuatorWrapper

__all__ = [
    "BAM_CONTROL_API",
    "ControllerBam",
    "NewtonActuatorAdapter",
    "PhysxActuatorWrapper",
    "apply_bam_startup_sampling",
    "build_implicit_dof_mask",
    "read_group_parameter",
    "register_bam_actuator_component",
    "write_group_parameter",
]
