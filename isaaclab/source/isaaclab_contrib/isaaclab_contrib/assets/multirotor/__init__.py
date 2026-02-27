# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for multirotor assets.

This module provides specialized classes for simulating multirotor vehicles (drones,
quadcopters, hexacopters, etc.) in Isaac Lab. It extends the base articulation
framework to support thrust-based control through individual rotor/propeller actuators.

Key Components:
    - :class:`Multirotor`: Asset class for multirotor vehicles with thruster control
    - :class:`MultirotorCfg`: Configuration class for multirotors
    - :class:`MultirotorData`: Data container for multirotor state information

Example:
    .. code-block:: python

        from isaaclab_contrib.assets import Multirotor, MultirotorCfg
        from isaaclab_contrib.actuators import ThrusterCfg
        import isaaclab.sim as sim_utils

        # Configure multirotor
        cfg = MultirotorCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path="path/to/quadcopter.usd"),
            actuators={
                "thrusters": ThrusterCfg(
                    thruster_names_expr=["rotor_[0-3]"],
                    thrust_range=(0.0, 10.0),
                )
            },
        )

        # Create multirotor instance
        multirotor = Multirotor(cfg)

.. seealso::
    - :mod:`isaaclab_contrib.actuators`: Thruster actuator models
    - :mod:`isaaclab_contrib.mdp.actions`: Thrust action terms for RL
"""

from .multirotor import Multirotor
from .multirotor_cfg import MultirotorCfg
from .multirotor_data import MultirotorData
