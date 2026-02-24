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

_NAMES = ["ManusVive", "ManusViveCfg", "OpenXRDevice", "OpenXRDeviceCfg", "XrAnchorRotationMode", "XrCfg", "remove_camera_configs"]
__all__ = _NAMES


def __getattr__(name):
    if name in _NAMES:
        import importlib

        mod = importlib.import_module("isaaclab_teleop.deprecated.openxr")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
