# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

from .device_base import DeviceBase
from .gamepad import Se2Gamepad, Se3Gamepad
from .keyboard import Se2Keyboard, Se3Keyboard
from .openxr import Se3HandTracking
from .spacemouse import Se2SpaceMouse, Se3SpaceMouse
