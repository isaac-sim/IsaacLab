# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .sensor_base import SensorBase


@configclass
class SensorBaseCfg:
    """Configuration parameters for a sensor."""

    class_type: type["SensorBase"] = MISSING
    """The associated sensor class.

    The class should inherit from :class:`isaaclab.sensors.sensor_base.SensorBase`.
    """

    cloning_contexts: tuple[str | type, ...] | None = None
    """Cloning contexts for this sensor. Defaults to None.

    Sensors carry no physics of their own. :func:`~isaaclab.cloner.replicate` auto-adds
    :class:`~isaaclab.cloner.UsdReplicateContext` when the sensor has a ``spawn`` field
    and Kit is available.
    """

    prim_path: str = MISSING
    """Prim path (or expression) to the sensor.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot/sensor`` will be replaced with ``/World/envs/env_.*/Robot/sensor``.

    """

    update_period: float = 0.0
    """Update period of the sensor buffers (in seconds). Defaults to 0.0 (update every step)."""

    debug_vis: bool = False
    """Whether to visualize the sensor. Defaults to False."""
