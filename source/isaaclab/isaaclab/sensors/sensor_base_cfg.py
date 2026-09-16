# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .sensor_base import SensorBase


@dataclass
class SensorBaseCfg:
    """Configuration parameters for a sensor."""

    class_type: type["SensorBase"] = config_field(MISSING)
    """The associated sensor class.

    The class should inherit from :class:`isaaclab.sensors.sensor_base.SensorBase`.
    """

    cloning_contexts: tuple[str | type, ...] | None = config_field(())
    """Cloning contexts for this sensor. Defaults to no explicit cloning context.

    Sensors carry no physics of their own. When the sensor has a spawner, USD replication is
    added automatically under Kit. Listing :class:`~isaaclab.cloner.UsdReplicateContext`
    explicitly forces USD replication without Kit; see
    :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts`.
    """

    prim_path: str = config_field(MISSING)
    """Prim path (or expression) to the sensor.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot/sensor`` will be replaced with ``/World/envs/env_[^/]+/Robot/sensor``.

    """

    update_period: float = config_field(0.0)
    """Update period of the sensor buffers (in seconds). Defaults to 0.0 (update every step)."""

    debug_vis: bool = config_field(False)
    """Whether to visualize the sensor. Defaults to False."""
