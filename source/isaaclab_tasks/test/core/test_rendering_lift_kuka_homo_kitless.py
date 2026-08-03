# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Lift KukaAllegro Lift backend combinations."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_kitless_rendering_params_lift,
    make_require_ovlibs_install_fixture,
    rendering_test_lift_kuka,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_RENDERING_PARAMS = make_kitless_rendering_params_lift()
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_type", _RENDERING_PARAMS, indirect=["ovstage_variant"]
)
def test_rendering_lift_kuka_homo_kitless(ovstage_variant, physics_backend, renderer, data_type):
    """Camera output must match golden images (Lift KukaAllegro Lift, single camera)."""
    rendering_test_lift_kuka(physics_backend, renderer, data_type, True, _COMPARISON_SCORES)
