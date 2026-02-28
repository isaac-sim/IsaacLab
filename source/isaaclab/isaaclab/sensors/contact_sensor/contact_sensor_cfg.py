# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor.

    Sensing bodies are selected via :attr:`SensorBaseCfg.prim_path`. Filter bodies for
    per-partner force reporting are selected via :attr:`filter_prim_paths_expr`.

    Only body-level sensing and filtering are supported. For shape-level granularity,
    see ``NewtonContactSensorCfg`` in ``isaaclab_newton``.
    """

    class_type: type["ContactSensor"] | str = "{DIR}.contact_sensor:ContactSensor"

    track_pose: bool = False
    """Whether to track the pose of the sensor's origin. Defaults to False."""

    track_contact_points: bool = False
    """Whether to track the contact point locations. Defaults to False."""

    track_friction_forces: bool = False
    """Whether to track the friction forces at the contact points. Defaults to False."""

    max_contact_data_count_per_prim: int | None = None
    """The maximum number of contacts across all batches of the sensor to keep track of. Default is 4, where supported.

    This parameter sets the total maximum counts of the simulation across all bodies and environments. The total number
    of contacts allowed is max_contact_data_count_per_prim*num_envs*num_sensor_bodies.

    .. note::

        If the environment is very contact rich it is suggested to increase this parameter to avoid out of bounds memory
        errors and loss of contact data leading to inaccurate measurements.
    """

    track_air_time: bool = False
    """Whether to track the air/contact time of the bodies (time between contacts). Defaults to False."""

    force_threshold: float | None = None
    """The threshold on the norm of the contact force that determines whether two bodies are in collision or not.
    Defaults to None, in which case the sensor backend chooses an appropriate value.

    This value is only used for tracking the mode duration (the time in contact or in air),
    if :attr:`track_air_time` is True.
    """

    history_length: int = 0
    """Number of past frames to store in the sensor buffers. Defaults to 0, which means that only
    the current data is stored (no history)."""

    filter_prim_paths_expr: list[str] = []
    """List of body prim path expressions to filter contacts against. Defaults to empty,
    meaning contacts with all bodies are aggregated into the net force.

    If provided, a per-partner force matrix (:attr:`ContactSensorData.force_matrix_w`) is
    reported in addition to the net force. Each expression is matched against body prim paths
    in the scene.

    For shape-level filtering, see ``NewtonContactSensorCfg`` in ``isaaclab_newton``.

    .. note::
        Expressions can contain the environment namespace regex ``{ENV_REGEX_NS}``, which
        is replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` becomes ``/World/envs/env_.*/Object``.

    .. attention::
        Filtered contact reporting only works when :attr:`SensorBaseCfg.prim_path` matches a
        single primitive per environment. For many-to-many filtering, see
        ``NewtonContactSensorCfg`` in ``isaaclab_newton``.
    """

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor")
    """The configuration object for the visualization markers. Defaults to CONTACT_SENSOR_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
