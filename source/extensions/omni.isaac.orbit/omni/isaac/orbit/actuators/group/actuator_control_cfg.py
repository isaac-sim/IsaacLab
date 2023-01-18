# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Dict, List, Optional

from omni.isaac.orbit.utils import configclass


@configclass
class ActuatorControlCfg:
    """Configuration for the joint-level controller used by the group.

    This configuration is used by the ActuatorGroup class to configure the commands types and their
    respective scalings and offsets to apply on the input actions over the actuator group. If the
    scales and offsets are set to :obj:`None`, then no scaling or offset is applied on the commands.

    Depending on the actuator model type, the gains are set either into the simulator (implicit) or to
    the actuator model class (explicit).

    The command types are processed as a list of strings. Each string has two sub-strings joined by
    underscore:

    - type of command mode: "p" (position), "v" (velocity), "t" (torque)
    - type of command resolving: "abs" (absolute), "rel" (relative)

    For instance, the command type "p_abs" defines a position command in absolute mode, while "v_rel"
    defines a velocity command in relative mode. For more information on the command types, see the
    documentation of the :class:`ActuatorGroup` class.
    """

    command_types: List[str] = MISSING
    """
    Command types applied on the DOF in the group.

    Note:
        The first string in the list defines the type of DOF drive mode in simulation. It must contain either
        "p" (position-controlled), "v" (velocity-controlled), or "t" (torque-controlled).
    """

    stiffness: Optional[Dict[str, float]] = None
    """
    Stiffness gains of the DOFs in the group. Defaults to :obj:`None`.

    If :obj:`None`, these are loaded from the articulation prim.
    """

    damping: Optional[Dict[str, float]] = None
    """
    Damping gains of the DOFs in the group. Defaults to :obj:`None`.

    If :obj:`None`, these are loaded from the articulation prim.
    """

    dof_pos_offset: Optional[Dict[str, float]] = None
    """
    DOF position offsets used for processing commands. Defaults to :obj:`None`.

    If :obj:`None`, these are processed as zero, i.e. absolute commands.
    """

    dof_pos_scale: Optional[Dict[str, float]] = None
    """
    DOF position scaling factor used for processing commands. Defaults to :obj:`None`.

    If :obj:`None`, these are processed as ones, i.e. absolute commands.
    """

    dof_vel_scale: Optional[Dict[str, float]] = None
    """
    DOF velocity scaling factor used for processing commands. Defaults to :obj:`None`.

    If :obj:`None`, these are processed as ones, i.e. absolute commands.
    """

    dof_torque_scale: Optional[Dict[str, float]] = None
    """
    DOF torque scaling factor used for processing commands. Defaults to :obj:`None`.

    If :obj:`None`, these are processed as ones, i.e. absolute commands.
    """
