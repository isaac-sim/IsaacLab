# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor."""

    class_type: type = ContactSensor

    track_pose: bool = False
    """Whether to track the pose of the sensor's origin. Defaults to False."""

    track_air_time: bool = False
    """Whether to track the air/contact time of the bodies (time between contacts). Defaults to False."""

    force_threshold: float = 0.0
    """The threshold on the norm of the contact force that determines whether two bodies are in collision or not.

    This value is only used for tracking the mode duration (the time in contact or in air),
    if :attr:`track_air_time` is True.
    """

    history_length: int = 0
    """Number of past frames to store in the sensor buffers. Defaults to 0, which means that only
    the current data is stored (no history)."""

    shape_path: list[str] | None = None
    """A list of expressions to filter contacts shapes with. Defaults to None. If both :attr:`body_names_expr` and
    :attr:`shape_names_expr` are None, the contact with all bodies/shapes is reported.

    Only one of :attr:`body_names_expr` or :attr:`shape_names_expr` can be provided.
    If both are provided, an error will be raised.

    We make an explicit difference between a body and a shape. A body is a rigid body, while a shape is a collision
    shape. A body can have multiple shapes. The shape option allows a more fine-grained control over the contact
    reporting.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.
    """

    filter_prim_paths_expr: list[str] | None = None
    """A list of expressions to filter contacts bodies with. Defaults to None. If both :attr:`contact_partners_body_expr` and
    :attr:`contact_partners_shape_expr` are None, the contact with all bodies/shapes is reported.

    Only one of :attr:`contact_partners_body_expr` or :attr:`contact_partners_shape_expr` can be provided.
    If both are provided, an error will be raised.

    The contact sensor allows reporting contacts between the primitive specified with either :attr:`body_names_expr` or
    :attr:`shape_names_expr` and other primitives in the scene. For instance, in a scene containing a robot, a ground
    plane and an object, you can obtain individual contact reports of the base of the robot with the ground plane and
    the object.

    We make an explicit difference between a body and a shape. A body is a rigid body, while a shape is a collision
    shape. A body can have multiple shapes. The shape option allows a more fine-grained control over the contact
    reporting.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.

    .. attention::
        The reporting of filtered contacts only works when the sensor primitive :attr:`prim_path` corresponds to a
        single primitive in that environment. If the sensor primitive corresponds to multiple primitives, the
        filtering will not work as expected. Please check :class:`~isaaclab.sensors.contact_sensor.ContactSensor`
        for more details.
    """

    filter_shape_paths_expr: list[str] | None = None
    """A list of expressions to filter contacts shapes with. Defaults to None. If both :attr:`contact_partners_body_expr` and
    :attr:`contact_partners_shape_expr` are None, the contact with all bodies/shapes is reported.

    Only one of :attr:`contact_partners_body_expr` or :attr:`contact_partners_shape_expr` can be provided.
    If both are provided, an error will be raised.

    The contact sensor allows reporting contacts between the primitive specified with either :attr:`body_names_expr` or
    :attr:`shape_names_expr` and other primitives in the scene. For instance, in a scene containing a robot, a ground
    plane and an object, you can obtain individual contact reports of the base of the robot with the ground plane and
    the object.


    We make an explicit difference between a body and a shape. A body is a rigid body, while a shape is a collision
    shape. A body can have multiple shapes. The shape option allows a more fine-grained control over the contact
    reporting.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.
    """

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor")
    """The configuration object for the visualization markers. Defaults to CONTACT_SENSOR_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
