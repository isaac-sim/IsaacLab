# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden runner for one bundled rendering-scene case."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import torch
from rendering_cases import (
    KIT_CASES,
    SCENE_PROBE_KIT_CASES,
    RenderCase,
    select_kitless_cases,
    select_kitless_scene_probe_cases,
)
from rendering_scene_cfgs import make_rendering_scene_cfg

from isaaclab.test.utils.golden_image import camera_output_image, compare_to_golden
from isaaclab.test.utils.rendering import SEMANTIC_COLORS, build_rendering_scene
from isaaclab.utils.seed import configure_seed

_GOLDEN_ROOT = Path(__file__).parent / "golden_images"
_ARTIFACT_DIR = Path.cwd() / "tests" / "comparison-images" / "images"
_NO_SSIM = {
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "instance_segmentation",
    "instance_id_segmentation_fast",
}
_SEMANTIC_ONLY_AOVS = {"motion_vectors"}
_MAX_DIFF_PCT = {
    # RTX edges vary by 8.43% between L40S and RTX 5090 while retaining SSIM 0.993.
    "shadow_hand": 10.0,
    # Four textured task views vary by 13.82% between L40S and RTX 5090 while retaining SSIM 0.985.
    "kuka_heterogeneous": 15.0,
    "franka_cloth": 8.0,
    # The 2x2 task layout quadruples RTX-antialiased table/object edges; SSIM still gates structure.
    "franka_soft": 12.0,
}


def run_rendering_case(case: RenderCase, request: Any, *, stage_variant: str = "kit") -> None:
    """Build once, capture all compatible AOVs, and step physics once only for motion."""
    configure_seed(42, torch_deterministic=True)
    (
        scene_cfg,
        camera_eye,
        camera_target,
        required_labels,
        physics_cfg,
        preserve_fixed_articulation_roots,
    ) = make_rendering_scene_cfg(case.scene, case.physics)
    with build_rendering_scene(
        scene_cfg,
        case.physics,
        renderer=case.renderer,
        data_types=case.aovs,
        background_color=case.background_color,
        camera_eye=camera_eye,
        camera_target=camera_target,
        physics_cfg=physics_cfg,
        preserve_fixed_articulation_roots=preserve_fixed_articulation_roots,
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
            motion = motion if torch.is_tensor(motion) else motion.torch
            assert torch.isfinite(motion).all(), "Motion vectors contain non-finite values."
            magnitude = motion[..., :2].abs().amax(dim=-1)
            peak = magnitude.amax(dim=(1, 2))
            raw_moving_pixels = (magnitude > 1.0e-6).flatten(1).sum(dim=1)
            support_pixels = (magnitude >= peak[:, None, None] * 0.99).flatten(1).sum(dim=1)
            view_pixels = magnitude.shape[1] * magnitude.shape[2]
            assert torch.all(peak > 1.0e-6), f"Motion-vector peaks stayed zero after one step: {peak.tolist()}"
            assert torch.all(raw_moving_pixels > 20), f"Too few moving pixels: {raw_moving_pixels.tolist()}"
            assert torch.all(support_pixels > 0), f"No high-confidence motion: {support_pixels.tolist()}"
            assert torch.all(support_pixels < view_pixels // 2), f"Motion is not localized: {support_pixels.tolist()}"

        _validate_segmentation(outputs, info, required_labels)
        failures = []
        for aov in case.aovs:
            if aov in _SEMANTIC_ONLY_AOVS:
                continue
            golden_label = f"{stage_variant}-{case.golden_id(aov)}"
            artifact_label = f"{case.scene}-{golden_label}"
            canonical_rtx_rgb = case.scene == "rendering_scene" and case.renderer == "isaac_rtx" and aov == "rgb"
            rtx_max_diff = 8.0 if canonical_rtx_rgb else _MAX_DIFF_PCT.get(case.scene, 3.0)
            comparison = compare_to_golden(
                camera_output_image(outputs[aov], aov),
                _GOLDEN_ROOT / case.scene / f"{golden_label}.png",
                label=artifact_label,
                artifact_dir=_ARTIFACT_DIR,
                max_diff_pct=(0.75 if case.renderer == "newton_warp" else rtx_max_diff),
                min_ssim=(
                    None
                    if aov in _NO_SSIM
                    else 0.95
                    if case.scene == "kuka_heterogeneous"
                    else 0.975
                    if canonical_rtx_rgb
                    else 0.98
                ),
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


def make_kit_test(*, scene_probes: bool = False) -> Any:
    """Create one Kit test function from the centrally owned case matrix."""
    import pytest

    cases = SCENE_PROBE_KIT_CASES if scene_probes else KIT_CASES

    @pytest.mark.parametrize("case", cases, ids=[case.id for case in cases])
    def test_rendering_scene(case: RenderCase, request: pytest.FixtureRequest) -> None:
        run_rendering_case(case, request)

    return test_rendering_scene


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


def _validate_segmentation(
    outputs: dict[str, torch.Tensor], info: dict[str, Any] | None, required_labels: frozenset[str]
) -> None:
    semantic = outputs.get("semantic_segmentation")
    if semantic is None:
        return
    assert info is not None and "semantic_segmentation" in info
    metadata = info["semantic_segmentation"]
    assert metadata is not None and "idToLabels" in metadata
    id_to_labels = metadata["idToLabels"]
    channels = min(semantic.shape[-1], 4)
    for label in required_labels:
        colors = [key for key, entry in id_to_labels.items() if isinstance(entry, dict) and entry.get("class") == label]
        assert colors, f"The semantic metadata does not contain required label {label!r}."
        rendered = False
        for value in colors:
            value = ast.literal_eval(value) if isinstance(value, str) else value
            color = torch.tensor(value[:channels], device=semantic.device, dtype=semantic.dtype)
            if semantic.is_floating_point() and semantic.max().item() <= 1.0:
                color = color / 255.0
            rendered |= bool(torch.any(torch.all(semantic[..., :channels] == color, dim=-1)).item())
        assert rendered, f"The semantic output does not contain required label {label!r}."

    labels = {
        entry["class"]
        for entry in id_to_labels.values()
        if isinstance(entry, dict) and "class" in entry and entry["class"] not in {"BACKGROUND", "UNLABELLED"}
    }
    expected = {name.split(":", 1)[1] for name in SEMANTIC_COLORS}
    assert labels <= expected, f"Unexpected semantic labels: {sorted(labels - expected)}."
