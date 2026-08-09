# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Process bootstrap and golden-test factory for Kit-less rendering."""

from __future__ import annotations

import os
from typing import Any

# TODO: Remove once usd-core>=26.5 is the minimum; that release fixes the USD work-thread race.
# This must run before importing the shared runner, which imports scene and OpenUSD modules.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

from rendering_cases import RenderCase, select_kitless_cases, select_kitless_scene_probe_cases  # noqa: E402
from rendering_runner import run_rendering_case  # noqa: E402


def run_kitless_rendering_case(stage_variant: str, case: RenderCase, request: Any, monkeypatch: Any) -> None:
    """Configure one Kit-less runtime variant and delegate to the shared scene runner."""
    if stage_variant == "ovstage":
        monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    else:
        monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    if case.renderer == "ovrtx":
        monkeypatch.setenv("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "0")
        try:
            __import__("ovrtx")
        except ImportError as exc:
            raise AssertionError(f"OVRTX rendering requires the ovrtx extra: {exc}") from exc
    if case.physics == "ovphysx":
        try:
            __import__("ovphysx")
        except ImportError as exc:
            raise AssertionError(f"OVPhysX rendering requires the ovphysx extra: {exc}") from exc
    run_rendering_case(case, request, stage_variant=stage_variant)


def make_kitless_test(stage: str, physics: str, *, scene_probes: bool = False) -> Any:
    """Create one process-isolated native-renderer test from the central matrix."""
    import pytest

    selector = select_kitless_scene_probe_cases if scene_probes else select_kitless_cases
    cases = selector(stage, physics)

    @pytest.mark.parametrize("stage_variant,case", cases, ids=[f"{case_stage}-{case.id}" for case_stage, case in cases])
    def test_rendering_scene(
        stage_variant: str,
        case: RenderCase,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_kitless_rendering_case(stage_variant, case, request, monkeypatch)

    return test_rendering_scene
