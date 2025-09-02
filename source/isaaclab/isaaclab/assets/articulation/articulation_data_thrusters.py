# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.assets.articulation.articulation_data import ArticulationData


class ArticulationDataWithThrusters(ArticulationData):
    """Data container for an articulation with thrusters.
    """
    
    thruster_names: list[str] = None
    """List of thruster joint names."""

    default_thruster_rps: torch.Tensor = None
    """Default thruster state of all thrusters. Shape is (num_instances, num_thrusters).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationWithThrustersCfg.init_state` parameter.
    """
    
    thrust_target: torch.Tensor = None
    """Thrust targets commanded by the user. Shape is (num_instances, num_thrusters).
    """
    
    ##
    # Thruster commands
    ##

    computed_thrust: torch.Tensor = None
    """Computed thrust from the actuator model (before clipping). Shape is (num_instances, num_thrusters).
    """

    applied_thrust: torch.Tensor = None
    """Applied thrust applied from the actuator model (after clipping). Shape is (num_instances, num_thrusters).
    """
    