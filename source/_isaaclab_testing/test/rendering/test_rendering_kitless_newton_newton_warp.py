# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden scenes rendered by Newton Warp without Kit while Newton advances physics."""

import os

import pytest
import rendering_runner
from isaaclab_newton.renderers import NewtonOptixRendererCfg
from rendering_cases import RenderCase, kitless_rendering_cases
from rendering_runner import run_kitless_rendering_case

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_CASES = [
    pytest.param(stage, case, id=f"{stage}-{case.id}", marks=case.marks)
    for stage, case in kitless_rendering_cases("newton", "newton_warp")
]


@pytest.mark.parametrize("stage_variant,case", _CASES)
def test_rendering(
    stage_variant: str, case: RenderCase, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare every compatible camera output from one scene construction."""
    run_kitless_rendering_case(stage_variant, case, request, monkeypatch)


@pytest.mark.parametrize("stage_variant,case", [case for case in _CASES if case.values[1].scene == "rendering_scene"])
def test_rendering_optix(
    stage_variant: str, case: RenderCase, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newton OptiX renders the Newton Warp golden images.

    Skips without the ``optix`` extra, unless ``ISAACLAB_TEST_REQUIRE_OPTIX`` is set: the kitless legacy
    rendering CI job sets it so a broken OptiX install fails instead of passing with nothing run.
    """
    if os.environ.get("ISAACLAB_TEST_REQUIRE_OPTIX"):
        import warp_optix  # noqa: F401
    else:
        pytest.importorskip("warp_optix")
    monkeypatch.setattr(rendering_runner, "NewtonWarpRendererCfg", NewtonOptixRendererCfg)
    run_kitless_rendering_case(stage_variant, case, request, monkeypatch)
