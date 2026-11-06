# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden scenes rendered by OVRTX without Kit while OVPhysX advances physics."""

import pytest
from rendering_cases import RenderCase, kitless_rendering_cases
from rendering_runner import run_kitless_rendering_case

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci, pytest.mark.ov]

_CASES = [
    pytest.param(stage, case, id=f"{stage}-{case.id}", marks=case.marks)
    for stage, case in kitless_rendering_cases("ovphysx", "ovrtx")
]


@pytest.mark.parametrize("stage_variant,case", _CASES)
def test_rendering(
    stage_variant: str, case: RenderCase, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare both OVRTX scene readers against the same expected images."""
    run_kitless_rendering_case(stage_variant, case, request, monkeypatch)
