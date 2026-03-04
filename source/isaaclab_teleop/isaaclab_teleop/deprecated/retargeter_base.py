# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-export of legacy retargeter base classes.

The canonical definitions remain in :mod:`isaaclab.devices.retargeter_base`
because :class:`~isaaclab.devices.DeviceCfg` depends on them for all device
types (keyboard, gamepad, etc.).  This module simply re-exports them so that
code under ``isaaclab_teleop.deprecated`` can reference them locally.
"""

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg  # noqa: F401

__all__ = ["RetargeterBase", "RetargeterCfg"]
