# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import config_field

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@dataclass
class XPBDSolverCfg(NewtonSolverCfg):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant
          constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16).
          Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid
          body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics
          Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU,
          Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    """

    class_type: type[NewtonManager] | str = config_field("{DIR}.xpbd_manager:NewtonXPBDManager")
    """Manager class for the XPBD solver."""

    solver_type: str = config_field("xpbd")
    """Solver type. Can be "xpbd"."""

    iterations: int = config_field(2)
    """Number of solver iterations."""

    soft_body_relaxation: float = config_field(0.9)
    """Relaxation parameter for soft body simulation."""

    soft_contact_relaxation: float = config_field(0.9)
    """Relaxation parameter for soft contact simulation."""

    joint_linear_relaxation: float = config_field(0.7)
    """Relaxation parameter for joint linear simulation."""

    joint_angular_relaxation: float = config_field(0.4)
    """Relaxation parameter for joint angular simulation."""

    joint_linear_compliance: float = config_field(0.0)
    """Compliance parameter for joint linear simulation."""

    joint_angular_compliance: float = config_field(0.0)
    """Compliance parameter for joint angular simulation."""

    rigid_contact_relaxation: float = config_field(0.8)
    """Relaxation parameter for rigid contact simulation."""

    rigid_contact_con_weighting: bool = config_field(True)
    """Whether to use contact constraint weighting for rigid contact simulation."""

    angular_damping: float = config_field(0.0)
    """Angular damping parameter for rigid contact simulation."""

    enable_restitution: bool = config_field(False)
    """Whether to enable restitution for rigid contact simulation."""
