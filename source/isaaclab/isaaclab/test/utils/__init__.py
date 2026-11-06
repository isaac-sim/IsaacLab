# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-time helpers for Isaac Lab.

Exposes :class:`DeviceScope` and :func:`test_devices` for selecting the device
list to parametrize tests over, plus :func:`resolve_test_sim_device` for deriving
a Kit-backed test's device from the same runtime mask, and :func:`launch_test_simulation`
to start the simulation runtime for a test module. The selected
set is ``scope ∩ runtime``: ``scope`` is the call-site argument (the devices the
test is valid on), the runtime is the ``ISAACLAB_TEST_DEVICES`` env var (the
devices the run may use).
"""

from .devices import DeviceScope, resolve_test_sim_device, test_devices
from .simulation import launch_test_simulation

__all__ = ["DeviceScope", "launch_test_simulation", "resolve_test_sim_device", "test_devices"]
