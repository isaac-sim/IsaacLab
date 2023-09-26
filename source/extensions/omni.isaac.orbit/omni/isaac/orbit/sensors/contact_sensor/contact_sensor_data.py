# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ContactSensorData:
    """Data container for the contact reporting sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where ``N`` is the number of sensors.
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame.

    Shape is (N, 4), where ``N`` is the number of sensors.
    """

    net_forces_w: torch.Tensor = None
    """The net contact forces in world frame.

    Shape is (N, B, 3), where ``N`` is the number of sensors and ``B`` is the number of bodies in each sensor.
    """
    net_forces_w_history: torch.Tensor = None
    """The net contact forces in world frame.

    Shape is (N, T, B, 3), where ``N`` is the number of sensors, ``T`` is the configured history length
    and ``B`` is the number of bodies in each sensor.

    In the history dimension, the first index is the most recent and the last index is the oldest.
    """

    force_matrix_w: torch.Tensor = None
    """The contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, B, S, M, 3), where ``N`` is the number of sensors, ``B`` is number of bodies in each sensor,
    ``S`` is number of shapes per body and ``M`` is the number of filtered bodies.

    If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this tensor will be empty.
    """

    last_air_time: torch.Tensor = None
    """Time spent (in s) in the air before the last contact.

    Shape is (N,), where ``N`` is the number of sensors.
    """
    current_air_time: torch.Tensor = None
    """Time spent (in s) in the air since the last contact.

    Shape is (N,), where ``N`` is the number of sensors.
    """
