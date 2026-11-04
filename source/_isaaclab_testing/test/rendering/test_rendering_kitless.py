# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden scenes rendered without Kit."""

import pytest
from rendering_cases import KITLESS_RENDERING_CASES, RenderCase
from rendering_runner import run_rendering_case

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_CASES = [
    pytest.param(
        stage,
        case,
        id=f"{stage}-{case.id}",
        marks=pytest.mark.ov if case.physics == "ovphysx" or case.renderer == "ovrtx" else (),
    )
    for stage, case in KITLESS_RENDERING_CASES
]


@pytest.mark.parametrize("stage_variant,case", _CASES)
def test_rendering(
    stage_variant: str, case: RenderCase, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare both OVRTX scene readers against the same expected images."""
    monkeypatch.delenv("ISAAC_LAB_SHARE_OVSTAGE", raising=False)
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(stage_variant == "ovstage")))
    run_rendering_case(case, request, golden_namespace=None, artifact_namespace=stage_variant)
