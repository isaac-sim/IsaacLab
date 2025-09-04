# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import time

import pytest

from isaaclab.app import AppLauncher


@pytest.mark.isaacsim_ci
def test_kit_start_up_time():
    """Test kit start-up time."""
    start_time = time.time()
    app_launcher = AppLauncher(headless=True).app  # noqa: F841
    end_time = time.time()
    elapsed_time = end_time - start_time
    # we are doing some more imports on the automate side - will investigate using warp instead of numba cuda
    assert elapsed_time <= 12.0
