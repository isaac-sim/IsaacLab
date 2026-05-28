# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-time helpers for Isaac Lab.

Currently exposes :func:`cuda_test_devices` for selecting the device list to
parametrize tests over, controllable via the ``ISAACLAB_TEST_DEVICES`` env var.
"""

from .devices import cuda_test_devices

__all__ = ["cuda_test_devices"]
