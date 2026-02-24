# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory to create teleoperation devices from configuration.

.. deprecated::
    This module is deprecated. Please use :class:`isaaclab_teleop.IsaacTeleopDevice`
    instead of :func:`create_teleop_device`.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable
from typing import cast

from isaaclab.devices import DeviceBase, DeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase

# import logger
logger = logging.getLogger(__name__)


def create_teleop_device(
    device_name: str, devices_cfg: dict[str, DeviceCfg], callbacks: dict[str, Callable] | None = None
) -> DeviceBase:
    """Create a teleoperation device based on configuration.

    .. deprecated::
        Use :class:`isaaclab_teleop.IsaacTeleopDevice` instead.

    Args:
        device_name: The name of the device to create (must exist in devices_cfg)
        devices_cfg: Dictionary of device configurations
        callbacks: Optional dictionary of callbacks to register with the device
            Keys are the button/gesture names, values are callback functions

    Returns:
        The configured teleoperation device

    Raises:
        ValueError: If the device name is not found in the configuration
        ValueError: If the device configuration type is not supported
    """
    warnings.warn(
        "create_teleop_device is deprecated. Please use isaaclab_teleop.IsaacTeleopDevice instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if device_name not in devices_cfg:
        raise ValueError(f"Device '{device_name}' not found in teleop device configurations")

    device_cfg = devices_cfg[device_name]
    callbacks = callbacks or {}

    # Determine constructor from the configuration itself
    device_constructor = getattr(device_cfg, "class_type", None)
    if device_constructor is None:
        raise ValueError(
            f"Device configuration '{device_name}' does not declare class_type. "
            "Set cfg.class_type to the concrete DeviceBase subclass."
        )
    if not issubclass(device_constructor, DeviceBase):
        raise TypeError(f"class_type for '{device_name}' must be a subclass of DeviceBase; got {device_constructor}")

    # Try to create retargeters if they are configured
    retargeters = []
    if hasattr(device_cfg, "retargeters") and device_cfg.retargeters is not None:
        try:
            # Create retargeters based on configuration using per-config retargeter_type
            for retargeter_cfg in device_cfg.retargeters:
                retargeter_constructor = getattr(retargeter_cfg, "retargeter_type", None)
                if retargeter_constructor is None:
                    raise ValueError(
                        f"Retargeter configuration {type(retargeter_cfg).__name__} does not declare retargeter_type. "
                        "Set cfg.retargeter_type to the concrete RetargeterBase subclass."
                    )
                if not issubclass(retargeter_constructor, RetargeterBase):
                    raise TypeError(
                        f"retargeter_type for {type(retargeter_cfg).__name__} must be a subclass of RetargeterBase; got"
                        f" {retargeter_constructor}"
                    )
                retargeters.append(retargeter_constructor(retargeter_cfg))

        except NameError as e:
            raise ValueError(f"Failed to create retargeters: {e}")

    # Build constructor kwargs based on signature
    constructor_params = inspect.signature(device_constructor).parameters
    params: dict = {"cfg": device_cfg}
    if "retargeters" in constructor_params:
        params["retargeters"] = retargeters
    device = cast(DeviceBase, device_constructor(**params))

    # Register callbacks
    for key, callback in callbacks.items():
        device.add_callback(key, callback)

    logging.info(f"Created teleoperation device: {device_name}")
    return device
