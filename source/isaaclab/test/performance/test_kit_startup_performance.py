# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import os
import time

from isaaclab.app import AppLauncher

_LOCAL_STARTUP_TIME_LIMIT = 15.0
_CI_STARTUP_TIME_LIMIT = 20.0


def test_kit_start_up_time():
    """Test kit start-up time."""
    start_time = time.time()
    app_launcher = AppLauncher(headless=True).app  # noqa: F841
    end_time = time.time()
    elapsed_time = end_time - start_time
    # GitHub Actions Docker jobs run with isolated writable runtime/cache mounts
    # for non-root users, which makes startup slightly colder than reused local caches.
    startup_time_limit = _CI_STARTUP_TIME_LIMIT if os.getenv("GITHUB_ACTIONS") == "true" else _LOCAL_STARTUP_TIME_LIMIT
    assert elapsed_time <= startup_time_limit, (
        f"Kit startup took {elapsed_time:.2f}s (limit {startup_time_limit:.2f}s)."
    )
