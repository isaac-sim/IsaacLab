# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for 3D Gaussian scenes and objects with the OVRTX renderer."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    GAUSSIAN_SPLATS_KITLESS_RENDERING_PARAMS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_require_ovlibs_install_fixture,
    rendering_test_gaussian_splats,
)

pytestmark = pytest.mark.arm_ci

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize(
    "ovstage_variant,scene_name,physics_backend,renderer",
    GAUSSIAN_SPLATS_KITLESS_RENDERING_PARAMS,
    indirect=["ovstage_variant"],
)
def test_rendering_gaussian_splats_kitless(ovstage_variant, scene_name, physics_backend, renderer):
    """Gaussian scene RGB output must match golden images (OVRTX)."""
    rendering_test_gaussian_splats(scene_name, physics_backend, renderer, _COMPARISON_SCORES)
