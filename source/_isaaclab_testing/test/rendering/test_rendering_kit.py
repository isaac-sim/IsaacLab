# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden scenes rendered through Kit-compatible renderer/backend pairs."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import pytest  # noqa: E402
from rendering_cases import KIT_RENDERING_CASES, RenderCase  # noqa: E402
from rendering_runner import run_rendering_case  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


@pytest.mark.parametrize("case", KIT_RENDERING_CASES, ids=[case.id for case in KIT_RENDERING_CASES])
def test_rendering(case: RenderCase, request: pytest.FixtureRequest) -> None:
    """Compare every compatible camera output from one scene construction."""
    run_rendering_case(case, request)
