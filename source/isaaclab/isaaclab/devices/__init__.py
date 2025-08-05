# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package providing interfaces to different teleoperation devices.

Currently, the following categories of devices are supported:

* **Keyboard**: Standard keyboard with WASD and arrow keys.
* **Spacemouse**: 3D mouse with 6 degrees of freedom.
* **Gamepad**: Gamepad with 2D two joysticks and buttons. Example: Xbox controller.
* **OpenXR**: Uses hand tracking of index/thumb tip avg to drive the target pose. Gripping is done with pinching.

All device interfaces inherit from the :class:`DeviceBase` class, which provides a
common interface for all devices. The device interface reads the input data when
the :meth:`DeviceBase.advance` method is called. It also provides the function :meth:`DeviceBase.add_callback`
to add user-defined callback functions to be called when a particular input is pressed from
the peripheral device.
"""

from .device_base import DeviceBase, DeviceCfg, DevicesCfg
from .gamepad import Se2Gamepad, Se2GamepadCfg, Se3Gamepad, Se3GamepadCfg
from .keyboard import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg
from .openxr import OpenXRDevice, OpenXRDeviceCfg
from .retargeter_base import RetargeterBase, RetargeterCfg
from .spacemouse import Se2SpaceMouse, Se2SpaceMouseCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from .teleop_device_factory import create_teleop_device
