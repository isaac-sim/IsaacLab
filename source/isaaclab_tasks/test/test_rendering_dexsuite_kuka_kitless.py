# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Dexsuite Kuka-Allegro Lift backend combinations."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_require_ovrtx_install_fixture,
    rendering_test_dexsuite_kuka,
)

pytestmark = pytest.mark.isaacsim_ci

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovrtx_install_fixture = make_require_ovrtx_install_fixture()


@pytest.mark.parametrize("physics_backend,renderer,data_type", KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS)
def test_rendering_dexsuite_kuka_kitless(physics_backend, renderer, data_type):
    """Camera output must match golden images (Dexsuite Kuka-Allegro Lift, single camera)."""
    rendering_test_dexsuite_kuka(physics_backend, renderer, data_type, _COMPARISON_SCORES)
