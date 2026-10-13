# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .joint_impedance import NewtonJointImpedanceController


@configclass
class NewtonJointImpedanceControllerCfg:
    """Configuration for :class:`NewtonJointImpedanceController`.

    The fields map one-to-one onto :class:`newton.controllers.ControllerJointImpedanceModelFree`, whose defaults
    they mirror. Gains are a scalar, a per-joint sequence, or ``None`` to read them live from
    :meth:`~NewtonJointImpedanceController.compute` (variable impedance). Their units are [1/s²] and [1/s] with
    :attr:`use_inertia_decoupling`, otherwise [N/m or N·m/rad] and [N·s/m or N·m·s/rad].
    """

    class_type: type[NewtonJointImpedanceController] | str = "{DIR}.joint_impedance:NewtonJointImpedanceController"
    """The associated controller class."""

    stiffness: float | Sequence[float] | None = MISSING
    """Position-error gain Kp."""

    damping: float | Sequence[float] | None = MISSING
    """Velocity-error gain Kd."""

    use_gravity_compensation: bool = True
    """Whether to add the gravity generalized forces to the output."""

    use_coriolis_compensation: bool = True
    """Whether to add the Coriolis generalized forces to the output."""

    use_inertia_decoupling: bool = True
    """Whether to premultiply the PD acceleration by the joint-space mass matrix."""

    use_qdd_feedforward: bool = False
    """Whether to add a desired joint acceleration as feedforward."""
