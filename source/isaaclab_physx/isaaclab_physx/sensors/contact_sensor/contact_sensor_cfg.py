# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.contact_sensor import ContactSensorCfg as BaseContactSensorCfg
from isaaclab.utils import configclass

from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(BaseContactSensorCfg):
    """Configuration for the PhysX contact sensor.

    Extends :class:`isaaclab.sensors.ContactSensorCfg` with PhysX-specific fields.
    """

    class_type: type = ContactSensor

    track_contact_points: bool = False
    """Whether to track the contact point locations. Defaults to False.

    .. note::
        Requires :attr:`filter_prim_paths_expr` to be set.
    """

    track_friction_forces: bool = False
    """Whether to track the friction forces at the contact points. Defaults to False.

    .. note::
        Requires :attr:`filter_prim_paths_expr` to be set.
    """

    max_contact_data_count_per_prim: int = 4
    """The maximum number of contacts per primitive to track. Default is 4.

    The total number of contacts allowed is ``max_contact_data_count_per_prim * num_envs * num_sensor_bodies``.

    .. note::
        If the environment is contact-rich, increase this parameter to avoid out of bounds memory
        errors and loss of contact data leading to inaccurate measurements.
    """
