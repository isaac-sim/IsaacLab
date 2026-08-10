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
from golden_image import camera_output_image, compare_to_golden
from rendering_cases import KIT_RENDERING_CASES, RenderCase
from rendering_runtime import SEMANTIC_COLORS, build_rendering_scene
from rendering_scene_cfgs import make_rendering_scene_spec

from isaaclab.renderers.output_contract import RenderBufferKind
from isaaclab.utils.seed import configure_seed

_GOLDEN_ROOT = Path(__file__).parent / "golden_images" / "renderers"
_ARTIFACT_DIR = Path.cwd() / "tests" / "comparison-images" / "images"
_NO_SSIM = {
    RenderBufferKind.DEPTH,
    RenderBufferKind.DISTANCE_TO_CAMERA,
    RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
    RenderBufferKind.INSTANCE_SEGMENTATION,
    RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST,
    RenderBufferKind.MOTION_VECTORS,
}
_ALPHA_ONLY_AOVS = {RenderBufferKind.INSTANCE_SEGMENTATION, RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST}


def run_rendering_case(
    case: RenderCase,
    request: Any,
    *,
    golden_namespace: str | None = "kit",
    artifact_namespace: str | None = None,
) -> None:
    """Build once, capture all compatible AOVs, and step physics once only for motion."""
    configure_seed(42, torch_deterministic=True)
    scene = make_rendering_scene_spec(case.scene, case.physics)
    with build_rendering_scene(
        scene.cfg,
        case.physics,
        renderer=case.renderer,
        data_types=case.aovs,
        background_color=case.background_color,
        camera_eye=scene.camera_eye,
        camera_target=scene.camera_target,
        physics_cfg=scene.physics_cfg,
        preserve_fixed_articulation_roots=scene.preserve_fixed_articulation_roots,
    ) as runtime:
        runtime.stabilize_camera()
        outputs, info = runtime.camera_outputs()
        if RenderBufferKind.MOTION_VECTORS in case.aovs:
            assert runtime.sim.get_physics_step_count() == 0
            runtime.step(render=False)
            runtime.render_camera()
            motion_outputs, _ = runtime.camera_outputs()
            assert runtime.sim.get_physics_step_count() == 1
            motion = motion_outputs[RenderBufferKind.MOTION_VECTORS]
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
            outputs[RenderBufferKind.MOTION_VECTORS] = motion

        _validate_segmentation(outputs, info, scene.required_labels)
        failures = []
        for aov in case.golden_aovs:
            image_max_diff_pct, min_ssim = scene.image_tolerance(case.renderer, aov)
            golden_filename = case.golden_filename(aov, golden_namespace)
            artifact_label = "-".join(
                part for part in (case.scene, artifact_namespace, golden_filename.removesuffix(".png")) if part
            )
            comparison = compare_to_golden(
                camera_output_image(outputs[aov], aov),
                _GOLDEN_ROOT / case.scene / golden_filename,
                label=artifact_label,
                artifact_dir=_ARTIFACT_DIR,
                max_diff_pct=0.75 if case.renderer == "newton_warp" else image_max_diff_pct,
                min_ssim=None if aov in _NO_SSIM else min_ssim,
                # OVRTX's numeric semantic IDs vary by USD reader; metadata validates their labels separately.
                alpha_only=aov in _ALPHA_ONLY_AOVS
                or (case.renderer == "ovrtx" and aov == RenderBufferKind.SEMANTIC_SEGMENTATION),
            )
            comparison.record(request)
            if not comparison.passed:
                failures.append(f"{aov}: {comparison.error}")
        assert not failures, "\n".join([f"{case.id} failed:", *failures])


def make_kit_test() -> Any:
    """Create one Kit test function from the centrally owned case matrix."""
    import pytest

    @pytest.mark.parametrize("case", KIT_RENDERING_CASES, ids=[case.id for case in KIT_RENDERING_CASES])
    def test_rendering_scene(case: RenderCase, request: pytest.FixtureRequest) -> None:
        run_rendering_case(case, request)

    return test_rendering_scene


def _validate_segmentation(
    outputs: dict[str, torch.Tensor], info: dict[str, Any] | None, required_labels: frozenset[str]
) -> None:
    semantic = outputs.get(RenderBufferKind.SEMANTIC_SEGMENTATION)
    if semantic is None:
        return
    assert info is not None and RenderBufferKind.SEMANTIC_SEGMENTATION in info
    metadata = info[RenderBufferKind.SEMANTIC_SEGMENTATION]
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
