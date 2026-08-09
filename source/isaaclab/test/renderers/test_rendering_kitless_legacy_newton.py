# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy OVRTX/Newton rendering in a bounded native-resource process."""

import pytest
from kitless_rendering_runner import make_kitless_test

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

test_rendering_scene_kitless = make_kitless_test("legacy", "newton")
