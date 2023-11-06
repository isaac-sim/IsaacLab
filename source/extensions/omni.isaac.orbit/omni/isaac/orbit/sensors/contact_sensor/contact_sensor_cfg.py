# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.orbit.markers import VisualizationMarkersCfg
from omni.isaac.orbit.markers.config import CONTACT_SENSOR_MARKER_CFG
from omni.isaac.orbit.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor."""

    class_type: type = ContactSensor

    track_pose: bool = False
    """Whether to track the pose of the sensor's origin. Defaults to False."""

    track_air_time: bool = False
    """Whether to track the air time of the bodies (time between contacts). Defaults to False."""

    filter_prim_paths_expr: list[str] = list()
    """The list of primitive paths to filter contacts with.

    For example, if you want to filter contacts with the ground plane, you can set this to
    ``["/World/ground_plane"]``. In this case, the contact sensor will only report contacts
    with the ground plane while using the :meth:`omni.isaac.core.prims.RigidContactView.get_contact_force_matrix`
    method.

    If an empty list is provided, then only net contact forces are reported.
    """

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor")
    """The configuration object for the visualization markers. Defaults to CONTACT_SENSOR_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
