# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 probe: every preceding Kit case, then the Valiant color cases. Do not merge."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import json  # noqa: E402
import os  # noqa: E402

from tmp_valiant_teal_order import make_test  # noqa: E402

test_valiant_teal_order = make_test(
    os.environ.get("TEAL_ORDER_CASES", "all"),
    int(os.environ.get("TEAL_ORDER_REPEATS", "15")),
    os.environ.get("TEAL_ORDER_LABEL", "all"),
    json.loads(os.environ.get("TEAL_ORDER_SETTINGS", "{}")),
)
