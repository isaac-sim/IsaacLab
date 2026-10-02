# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend and device parameters shared by the asset contracts."""

import pytest

from isaaclab.test.utils import DeviceScope, test_devices

from .backends import backends


@pytest.fixture(params=backends())
def backend(request: pytest.FixtureRequest) -> str:
    """Run each shared contract on every backend, skipping the ones unavailable in this process."""
    return request.param


@pytest.fixture(params=test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def device(request: pytest.FixtureRequest) -> str:
    """Exercise CPU buffers and CUDA staging."""
    return request.param
