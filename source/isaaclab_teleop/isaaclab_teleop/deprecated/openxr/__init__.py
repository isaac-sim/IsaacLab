# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy OpenXR teleoperation devices.

.. deprecated::
    This package is deprecated. Please migrate to :mod:`isaaclab_teleop` which
    provides the :class:`~isaaclab_teleop.IsaacTeleopDevice` as a replacement
    for :class:`OpenXRDevice` and :class:`ManusVive`.

    XR configuration classes (:class:`XrCfg`, :class:`XrAnchorRotationMode`,
    :func:`remove_camera_configs`) have moved to :mod:`isaaclab_teleop.xr_cfg`.
    Anchor utilities (:class:`XrAnchorSynchronizer`) have moved to
    :mod:`isaaclab_teleop.xr_anchor_utils`.
"""

from .manus_vive import ManusVive, ManusViveCfg
from .openxr_device import OpenXRDevice, OpenXRDeviceCfg
from .xr_cfg import XrAnchorRotationMode, XrCfg, remove_camera_configs
