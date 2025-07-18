# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ContactSensorData:
    """Data container for the contact reporting sensor."""

    pos_w: torch.Tensor | None = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
    """

    contact_pos_w: torch.Tensor | None = None
    """Average of the positions of contact points between sensor body and filter prim in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Collision pairs not in contact will result in nan.

    Note:
        If the :attr:`ContactSensorCfg.track_contact_points` is False, then this quantity is None.
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is an empty tensor.
        If the :attr:`ContactSensorCfg.max_contact_data_per_prim` is not specified or less than 1, then this quantity
            will not be calculated.
    """

    quat_w: torch.Tensor | None = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
    """

    net_forces_w: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, B, 3), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    net_forces_w_history: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, T, B, 3), where N is the number of sensors, T is the configured history length
    and B is the number of bodies in each sensor.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    force_matrix_w: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """

    force_matrix_w_history: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, T, B, M, 3), where N is the number of sensors, T is the configured history length,
    B is number of bodies in each sensor and M is the number of filtered bodies.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """

    last_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air before the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    current_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air since the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    last_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact before the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    current_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact since the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """
