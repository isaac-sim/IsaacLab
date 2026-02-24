# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.contact_sensor import ContactSensorCfg as BaseContactSensorCfg
from isaaclab.utils import DeferredClass, configclass


@configclass
class ContactSensorCfg(BaseContactSensorCfg):
    """Configuration for the Newton contact sensor.

    Extends :class:`isaaclab.sensors.ContactSensorCfg` with Newton-specific fields.
    """

    class_type: type | DeferredClass = DeferredClass("isaaclab_newton.sensors.contact_sensor.contact_sensor:ContactSensor")

    shape_path: list[str] | None = None
    """A list of shape path expressions to filter which shapes to sense contacts ON. Defaults to None,
    meaning all shapes on the bodies specified by :attr:`prim_path` are sensed.

    A body (rigid body) can have multiple collision shapes. This option allows more fine-grained
    control over which specific shapes to sense contacts on.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.
    """

    filter_shape_paths_expr: list[str] | None = None
    """A list of shape path expressions to filter contacts WITH. Defaults to None, meaning contacts
    with all shapes are reported.

    This is similar to :attr:`filter_prim_paths_expr` but filters at the shape level instead of body level.
    A body (rigid body) can have multiple collision shapes. This option allows more fine-grained
    control over which specific shapes to detect contacts with.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object/CollisionMesh`` will be replaced with
        ``/World/envs/env_.*/Object/CollisionMesh``.
    """
