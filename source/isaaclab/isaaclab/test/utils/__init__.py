# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-time helpers for Isaac Lab.

Exposes :func:`test_devices` for selecting the device list to parametrize tests
over. The set is ``scope ∩ budget``: ``scope`` is the call-site argument (the
devices the test is valid on), ``budget`` is the ``ISAACLAB_TEST_DEVICES`` env
var (the devices the run may use).
"""

from .devices import test_devices

__all__ = ["test_devices"]
