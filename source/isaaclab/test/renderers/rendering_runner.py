# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden runner for one bundled rendering-scene case."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rendering_cases import RenderCase, select_kitless_cases

from isaaclab.test.utils.golden_image import camera_output_image, compare_to_golden
from isaaclab.test.utils.rendering import SEMANTIC_COLORS, build_rendering_scene
from isaaclab.utils.seed import configure_seed

_GOLDEN_DIR = Path(__file__).parent / "golden_images" / "rendering_scene"
_ARTIFACT_DIR = Path.cwd() / "tests" / "comparison-images" / "images"
_NO_SSIM = {
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "instance_segmentation",
    "instance_id_segmentation_fast",
    "motion_vectors",
}


def run_rendering_case(case: RenderCase, request: Any, *, stage_variant: str = "kit") -> None:
    """Build once, capture all compatible AOVs, and step physics once only for motion."""
    configure_seed(42, torch_deterministic=True)
    with build_rendering_scene(
        case.physics,
        renderer=case.renderer,
        data_types=case.aovs,
        num_envs=1,
        background_color=case.background_color,
    ) as runtime:
        runtime.stabilize_camera()
        outputs, info = runtime.camera_outputs()
        if "motion_vectors" in case.aovs:
            assert runtime.sim.get_physics_step_count() == 0
            runtime.step(render=False)
            runtime.render_camera()
            motion_outputs, _ = runtime.camera_outputs()
            assert runtime.sim.get_physics_step_count() == 1
            motion = motion_outputs["motion_vectors"]
            assert torch.count_nonzero(motion).item() > 0, "Motion vectors stayed zero after one physics step."
            outputs["motion_vectors"] = motion

        _validate_segmentation_metadata(info)
        failures = []
        for aov in case.aovs:
            label = f"{stage_variant}-{case.golden_id(aov)}"
            comparison = compare_to_golden(
                camera_output_image(outputs[aov], aov),
                _GOLDEN_DIR / f"{label}.png",
                label=label,
                artifact_dir=_ARTIFACT_DIR,
                max_diff_pct=0.75 if case.renderer == "newton_warp" else 3.0,
                min_ssim=None if aov in _NO_SSIM else (0.99 if case.renderer == "newton_warp" else 0.98),
                alpha_only=aov in {"instance_segmentation", "instance_id_segmentation_fast"},
            )
            comparison.record(request)
            if not comparison.passed:
                failures.append(f"{aov}: {comparison.error}")
        assert not failures, "\n".join([f"{case.id} failed:", *failures])


def run_kitless_rendering_case(stage_variant: str, case: RenderCase, request: Any, monkeypatch: Any) -> None:
    """Configure one kitless runtime variant and delegate to the shared scene runner."""
    monkeypatch.setenv("PXR_WORK_THREAD_LIMIT", "1")
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


def make_kitless_test(stage: str, physics: str) -> Any:
    """Create a tiny process-isolated pytest composition root for one native lifecycle partition."""
    import pytest

    cases = select_kitless_cases(stage, physics)

    @pytest.mark.parametrize("stage_variant,case", cases, ids=[f"{case_stage}-{case.id}" for case_stage, case in cases])
    def test_rendering_scene_kitless(
        stage_variant: str,
        case: RenderCase,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_kitless_rendering_case(stage_variant, case, request, monkeypatch)

    return test_rendering_scene_kitless


def _validate_segmentation_metadata(info: dict[str, Any] | None) -> None:
    if info is None or "semantic_segmentation" not in info:
        return
    metadata = info["semantic_segmentation"]
    assert metadata is not None and "idToLabels" in metadata
    labels = {
        entry["class"]
        for entry in metadata["idToLabels"].values()
        if isinstance(entry, dict) and "class" in entry and entry["class"] not in {"BACKGROUND", "UNLABELLED"}
    }
    expected = {name.split(":", 1)[1] for name in SEMANTIC_COLORS}
    assert labels and labels <= expected, f"Unexpected semantic labels: {sorted(labels - expected)}."
    assert "robot" in labels, f"The canonical robot label is absent from {sorted(labels)}."
