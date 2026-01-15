# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets.articulation.articulation_data import ArticulationData


class MultirotorData(ArticulationData):
    """Data container for a multirotor articulation.

    This class extends the base articulation data container to include multirotor-specific
    data such as thruster states and forces.
    """

    thruster_names: list[str] = None
    """List of thruster names in the multirotor."""

    default_thruster_rps: torch.Tensor = None
    """Default thruster RPS state of all thrusters. Shape is (num_instances, num_thrusters).

    This quantity is configured through the :attr:`isaaclab.assets.MultirotorCfg.init_state` parameter.
    """

    thrust_target: torch.Tensor = None
    """Thrust targets commanded by the user. Shape is (num_instances, num_thrusters).

    This quantity contains the target thrust values set by the user through the
    :meth:`isaaclab.assets.Multirotor.set_thrust_target` method.
    """

    ##
    # Thruster commands
    ##

    computed_thrust: torch.Tensor = None
    """Computed thrust from the actuator model (before clipping). Shape is (num_instances, num_thrusters).

    This quantity contains the thrust values computed by the thruster actuator models
    before any clipping or saturation is applied.
    """

    applied_thrust: torch.Tensor = None
    """Applied thrust from the actuator model (after clipping). Shape is (num_instances, num_thrusters).

    This quantity contains the final thrust values that are applied to the simulation
    after all actuator model processing, including clipping and saturation.
    """
