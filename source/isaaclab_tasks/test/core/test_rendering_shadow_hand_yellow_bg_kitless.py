# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less golden render test: Shadow Hand with yellow camera background via OVRTX (RGB only)."""

import os

# OVRTX uses Vulkan, which requires a display socket even in headless mode.
os.environ.setdefault("DISPLAY", ":0")

import pytest  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    make_require_ovlibs_install_fixture,
    rendering_test_shadow_hand_yellow_bg,
)

pytestmark = [pytest.mark.isaacsim_ci]

_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize("physics_backend,renderer", [("newton", "ovrtx")])
def test_rgb_yellow_background_ovrtx(physics_backend, renderer):
    """Kit-less golden render test: RGB output with yellow background via OVRTX."""
    rendering_test_shadow_hand_yellow_bg(physics_backend, renderer, comparison_scores=[])
