# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical scene rendering through Kit-compatible renderer/backend pairs."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest  # noqa: E402
from rendering_cases import KIT_CASES, RenderCase  # noqa: E402
from rendering_runner import run_rendering_case  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


@pytest.mark.parametrize("case", KIT_CASES, ids=[case.id for case in KIT_CASES])
def test_rendering_scene(case: RenderCase, request: pytest.FixtureRequest) -> None:
    """All compatible AOVs share one task-free scene construction."""
    run_rendering_case(case, request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
