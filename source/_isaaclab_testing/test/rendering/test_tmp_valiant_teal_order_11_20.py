# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 probe: preceding Kit cases 11-20, then the Valiant color cases. Do not merge."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

from tmp_valiant_teal_order import make_test  # noqa: E402

test_valiant_teal_order = make_test("11-20", 15, "pre11-20")
