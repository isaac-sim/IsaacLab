# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR teleoperation devices (legacy).

.. deprecated::
    This package has moved to :mod:`isaaclab_teleop.deprecated.openxr`.
    Please migrate to :mod:`isaaclab_teleop` which provides the
    :class:`~isaaclab_teleop.IsaacTeleopDevice` as a replacement.

    Imports from this package will continue to work for backwards
    compatibility.  Individual class constructors emit
    :class:`DeprecationWarning` at instantiation time.
"""

try:

    from isaaclab_teleop.deprecated.openxr import (  # noqa: F401
        ManusVive,
        ManusViveCfg,
        OpenXRDevice,
        OpenXRDeviceCfg,
        XrAnchorRotationMode,
        XrCfg,
        remove_camera_configs,
    )
except ImportError:
    print("isaaclab_teleop is not installed. OpenXR teleoperation features will not be available.")
    pass
