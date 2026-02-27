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

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .xr_cfg import XrAnchorRotationMode, XrCfg, remove_camera_configs
    from .manus_vive import ManusVive, ManusViveCfg
    from .openxr_device import OpenXRDevice, OpenXRDeviceCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("xr_cfg", ["XrAnchorRotationMode", "XrCfg", "remove_camera_configs"]),
    ("manus_vive", ["ManusVive", "ManusViveCfg"]),
    ("openxr_device", ["OpenXRDevice", "OpenXRDeviceCfg"]),
)
