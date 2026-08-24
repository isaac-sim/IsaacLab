# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless regressions for per-link Kuka visual-material randomization."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_kitless_visual_material_rendering_params,
    make_require_ovlibs_install_fixture,
    rendering_test_kuka_visual_material_randomization,
)

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.arm_ci,
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.kitless,
]

_RENDERING_PARAMS = make_kitless_visual_material_rendering_params()
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize("ovstage_variant,physics_backend,renderer", _RENDERING_PARAMS, indirect=["ovstage_variant"])
def test_rendering_kuka_visual_material_randomization_kitless(ovstage_variant, physics_backend, renderer):
    """Runtime writes must reach Newton and both OVRTX stage implementations."""
    rendering_test_kuka_visual_material_randomization(physics_backend, renderer, _COMPARISON_SCORES)
