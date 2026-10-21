# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend and device parameters shared by the asset contracts."""

import pytest
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices

from .backends import backends


@pytest.fixture(params=backends())
def backend(request: pytest.FixtureRequest) -> str:
    """Run each shared contract on every backend, skipping the ones unavailable in this process."""
    return request.param


@pytest.fixture(params=test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def device(request: pytest.FixtureRequest) -> str:
    """Exercise CPU buffers and the CUDA staging paths of PhysX and OVPhysX."""
    return request.param


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect Newton CUDA cases: Newton's asset code has no CUDA-specific path, so the CPU case covers it."""
    deselected = []
    for item in items:
        params = getattr(item, "callspec", None) and item.callspec.params
        if params and params.get("backend") == "newton" and params.get("device", "cpu") != "cpu":
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item not in deselected]


@pytest.fixture(autouse=True)
def _default_warp_device(request: pytest.FixtureRequest):
    """Allocate unqualified Warp arrays on the case's device, or on CPU when the case has none."""
    callspec = getattr(request.node, "callspec", None)
    with wp.ScopedDevice(callspec.params.get("device", "cpu") if callspec else "cpu"):
        yield
