# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the contact sensor."""


from __future__ import annotations

from omni.isaac.orbit.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg


@configclass
class ContactSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor."""

    cls_name = "ContactSensor"

    filter_prim_paths_expr: list[str] = list()
    """The list of primitive paths to filter contacts with.

    For example, if you want to filter contacts with the ground plane, you can set this to
    ``["/World/ground_plane"]``. In this case, the contact sensor will only report contacts
    with the ground plane while using the :meth:`omni.isaac.core.prims.RigidContactView.get_contact_force_matrix`
    method.

    If an empty list is provided, then only net contact forces are reported.
    """

    history_length: int = 0
    """Number of past frames to store in the sensor buffers. Defaults to 0, which means that only
    the current data is stored."""
