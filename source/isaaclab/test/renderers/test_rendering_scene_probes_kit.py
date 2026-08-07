# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Specialized task-free scenes through Kit-compatible renderers."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest  # noqa: E402
from rendering_runner import make_kit_test  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci

test_rendering_scene_probes = make_kit_test(scene_probes=True)
