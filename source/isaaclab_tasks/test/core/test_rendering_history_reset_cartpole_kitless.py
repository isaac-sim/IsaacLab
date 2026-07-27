# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less render history reset golden-image tests for the Cartpole OVRTX environment."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_require_ovlibs_install_fixture,
    rendering_test_cartpole_render_history_reset,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize("physics_backend", ["newton", "ovphysx"])
def test_rendering_history_reset_cartpole_kitless(physics_backend: str) -> None:
    """Post-reset frame must be near-black and match the golden image after reset_tile_history."""
    rendering_test_cartpole_render_history_reset(physics_backend, renderer="ovrtx", comparison_scores=_COMPARISON_SCORES)
