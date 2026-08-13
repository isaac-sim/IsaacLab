# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""All golden scenes rendered through Kit-less renderer/backend pairs."""

import pytest
from kitless_rendering_runner import generate_kitless_test_cases

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci, pytest.mark.cold_cache]

test_rendering = generate_kitless_test_cases()
