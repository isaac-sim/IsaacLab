# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory to create teleoperation devices from configuration."""

import contextlib
import inspect
from collections.abc import Callable

import omni.log

from isaaclab.devices import DeviceBase, DeviceCfg
from isaaclab.devices.gamepad import Se2Gamepad, Se2GamepadCfg, Se3Gamepad, Se3GamepadCfg
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr.retargeters import (
    GR1T2Retargeter,
    GR1T2RetargeterCfg,
    GripperRetargeter,
    GripperRetargeterCfg,
    Se3AbsRetargeter,
    Se3AbsRetargeterCfg,
    Se3RelRetargeter,
    Se3RelRetargeterCfg,
)
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.devices.spacemouse import Se2SpaceMouse, Se2SpaceMouseCfg, Se3SpaceMouse, Se3SpaceMouseCfg

with contextlib.suppress(ModuleNotFoundError):
    # May fail if xr is not in use
    from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg

# Map device types to their constructor and expected config type
DEVICE_MAP: dict[type[DeviceCfg], type[DeviceBase]] = {
    Se3KeyboardCfg: Se3Keyboard,
    Se3SpaceMouseCfg: Se3SpaceMouse,
    Se3GamepadCfg: Se3Gamepad,
    Se2KeyboardCfg: Se2Keyboard,
    Se2GamepadCfg: Se2Gamepad,
    Se2SpaceMouseCfg: Se2SpaceMouse,
    OpenXRDeviceCfg: OpenXRDevice,
}


# Map configuration types to their corresponding retargeter classes
RETARGETER_MAP: dict[type[RetargeterCfg], type[RetargeterBase]] = {
    Se3AbsRetargeterCfg: Se3AbsRetargeter,
    Se3RelRetargeterCfg: Se3RelRetargeter,
    GripperRetargeterCfg: GripperRetargeter,
    GR1T2RetargeterCfg: GR1T2Retargeter,
}


def create_teleop_device(
    device_name: str, devices_cfg: dict[str, DeviceCfg], callbacks: dict[str, Callable] | None = None
) -> DeviceBase:
    """Create a teleoperation device based on configuration.

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
    if device_name not in devices_cfg:
        raise ValueError(f"Device '{device_name}' not found in teleop device configurations")

    device_cfg = devices_cfg[device_name]
    callbacks = callbacks or {}

    # Check if device config type is supported
    cfg_type = type(device_cfg)
    if cfg_type not in DEVICE_MAP:
        raise ValueError(f"Unsupported device configuration type: {cfg_type.__name__}")

    # Get the constructor for this config type
    constructor = DEVICE_MAP[cfg_type]

    # Try to create retargeters if they are configured
    retargeters = []
    if hasattr(device_cfg, "retargeters") and device_cfg.retargeters is not None:
        try:
            # Create retargeters based on configuration
            for retargeter_cfg in device_cfg.retargeters:
                cfg_type = type(retargeter_cfg)
                if cfg_type in RETARGETER_MAP:
                    retargeters.append(RETARGETER_MAP[cfg_type](retargeter_cfg))
                else:
                    raise ValueError(f"Unknown retargeter configuration type: {cfg_type.__name__}")

        except NameError as e:
            raise ValueError(f"Failed to create retargeters: {e}")

    # Check if the constructor accepts retargeters parameter
    constructor_params = inspect.signature(constructor).parameters
    if "retargeters" in constructor_params and retargeters:
        device = constructor(cfg=device_cfg, retargeters=retargeters)
    else:
        device = constructor(cfg=device_cfg)

    # Register callbacks
    for key, callback in callbacks.items():
        device.add_callback(key, callback)

    omni.log.info(f"Created teleoperation device: {device_name}")
    return device
