# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for visualizer golden image tests.

Adapts the dual-gate comparison system (per-pixel L2 norm + SSIM) from
``source/isaaclab_tasks/test/rendering_test_utils.py`` for visualizer RGB
frame output.  Covers :class:`~isaaclab_visualizers.kit.KitVisualizer`
(RTX viewport / tiled camera) and
:class:`~isaaclab_visualizers.newton.NewtonVisualizer` (OpenGL frame / tiled
camera).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops

_GOLDEN_IMAGES_DIRECTORY = Path(__file__).parent / "golden_images"

# Euclidean distance between two RGB pixels below which they are treated as equal.
_PIXEL_L2_NORM_DIFFERENCE_THRESHOLD = 10.0

# Kit RTX viewport inherits RTX renderer variability; use the cartpole renderer
# baseline (1.0 %).  Newton GL is more deterministic so we tighten to 0.5 %.
MAX_DIFF_PCT_BY_VISUALIZER: dict[str, float] = {
    "kit": 1.0,
    "newton": 0.5,
}

_SSIM_THRESHOLD_BY_VISUALIZER: dict[str, float] = {
    "kit": 0.985,
    "newton": 0.990,
}

# Tiled rendering composites N camera views; RTX anti-aliasing noise accumulates
# across tiles so the effective per-frame variance is higher than single-viewport.
# Override only the combinations that need extra headroom.
# Keys may be "visualizer-mode" (scene-agnostic) or "scene-visualizer-mode" (scene-specific).
# Scene-specific keys take precedence over scene-agnostic ones.
_MAX_DIFF_PCT_OVERRIDES: dict[str, float] = {
    "kit-tiled": 2.0,
    # Cartpole Kit viewport runs after tiled tests which leave residual RTX TAA history;
    # observed 1.35–5.83% diff vs the clean-state golden. The default 1% is too tight.
    "cartpole-kit-viewport": 7.0,
    # AnymalD Kit (RTX) produces higher inter-run variance than cartpole due to
    # complex geometry and materials; observed ~5.5% tiled, ~3.7% viewport on CI.
    "anymal_d-kit-tiled": 6.5,
    "anymal_d-kit-viewport": 4.5,
    # Shadow hand Kit (RTX) variance is higher still — many thin finger surfaces
    # amplify RTX anti-aliasing noise; observed ~24% tiled, ~7.5% viewport on CI.
    # Thresholds are set to catch catastrophic failures (black frames, wrong geometry)
    # rather than subtle rendering differences.
    "shadow_hand-kit-tiled": 30.0,
    "shadow_hand-kit-viewport": 9.0,
    # Franka cloth Kit RTX tiled: VBD cloth non-determinism between runs; observed ~4.2% tiled.
    "franka_cloth-kit-tiled": 5.5,
    # Franka cloth Kit RTX viewport: cloth VBD solver non-determinism combined with
    # RTX warm-up variance produces large inter-run pixel diffs; observed ~41% viewport.
    "franka_cloth-kit-viewport": 45.0,
}

_SSIM_THRESHOLD_OVERRIDES: dict[str, float] = {
    # Observed ~0.9650 on CI; 0.960 gives headroom without masking real regressions.
    "kit-tiled": 0.960,
    # Observed 0.8809 for cartpole Kit viewport after prior tiled tests (TAA history contamination).
    "cartpole-kit-viewport": 0.87,
    # Observed 0.9539 (tiled) and 0.9678 (viewport) for AnymalD Kit on CI.
    "anymal_d-kit-tiled": 0.945,
    "anymal_d-kit-viewport": 0.960,
    # Observed 0.8391 (tiled) and 0.9214 (viewport) for shadow hand Kit on CI.
    "shadow_hand-kit-tiled": 0.80,
    "shadow_hand-kit-viewport": 0.90,
    # Observed 0.6896 for franka cloth Kit viewport; cloth VBD non-determinism.
    "franka_cloth-kit-viewport": 0.65,
}

_COMPARISON_IMAGES_DIR = os.path.join(os.getcwd(), "tests", "comparison-images")
_COMPARISON_IMAGE_SUBDIR = "images"

# RTX variability makes golden checks flaky on CI; match the renderer test retry count.
FLAKY_MARK = pytest.mark.flaky(max_runs=3, min_passes=1)


# ---------------------------------------------------------------------------
# Image-comparison primitives
# ---------------------------------------------------------------------------


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM between two ``(1, C, H, W)`` float tensors in ``[0, 1]``."""
    c1 = 0.01**2
    c2 = 0.03**2
    channels = img1.shape[1]
    pad = window_size // 2

    kernel = torch.ones(channels, 1, window_size, window_size, device=img1.device, dtype=img1.dtype) / (
        window_size * window_size
    )

    mu1 = torch.nn.functional.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = torch.nn.functional.conv2d(img2, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def _pixel_diff_percentage(
    result_image: Image.Image,
    golden_image: Image.Image,
    pixel_diff_threshold: float = _PIXEL_L2_NORM_DIFFERENCE_THRESHOLD,
) -> float:
    """Return percentage of pixels whose L2 norm difference exceeds ``pixel_diff_threshold``."""
    diff_array = np.array(ImageChops.difference(result_image, golden_image))
    l2_norm_array = np.linalg.norm(diff_array, axis=2)
    num_different_pixels = np.sum(l2_norm_array > pixel_diff_threshold)
    return 100.0 * num_different_pixels / l2_norm_array.size


def _compare_images(
    result_image: Image.Image,
    golden_image: Image.Image,
    max_different_pixels_percentage: float,
    ssim_threshold: float,
) -> tuple[bool, str | None, float, float]:
    """Compare ``result_image`` against ``golden_image`` with L2 + SSIM dual gate.

    Args:
        result_image: PIL RGB image captured from the visualizer.
        golden_image: PIL RGB reference image loaded from the golden store.
        max_different_pixels_percentage: Maximum allowed fraction of differing pixels.
        ssim_threshold: Minimum required SSIM score.

    Returns:
        ``(passed, error_message, diff_pct, ssim_score)``
    """
    if result_image.size != golden_image.size:
        return False, f"Size mismatch: expected {golden_image.size}, got {result_image.size}.", 0.0, 0.0

    if result_image.mode != golden_image.mode:
        return False, f"Mode mismatch: expected {golden_image.mode}, got {result_image.mode}.", 0.0, 0.0

    diff_pct = _pixel_diff_percentage(result_image, golden_image)

    result_tensor = torch.from_numpy(np.array(result_image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    golden_tensor = torch.from_numpy(np.array(golden_image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    ssim_score = _ssim(result_tensor, golden_tensor)

    if diff_pct > max_different_pixels_percentage:
        return (
            False,
            f"Pixel diff ({diff_pct:.2f}%) exceeds threshold of {max_different_pixels_percentage:.2f}%."
            f" SSIM={ssim_score:.4f}.",
            diff_pct,
            ssim_score,
        )

    if ssim_score < ssim_threshold:
        return (
            False,
            f"SSIM ({ssim_score:.4f}) below threshold of {ssim_threshold:.4f}. Different pixels: {diff_pct:.2f}%.",
            diff_pct,
            ssim_score,
        )

    return True, None, diff_pct, ssim_score


def _save_comparison_image(img: Image.Image, filename: str) -> str:
    """Save ``img`` under the comparison images directory and return the path."""
    path = os.path.join(_COMPARISON_IMAGES_DIR, _COMPARISON_IMAGE_SUBDIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------


def validate_visualizer_frame(
    test_name: str,
    physics_backend: str,
    visualizer_type: str,
    mode: str,
    frame: np.ndarray,
    comparison_scores: list[dict],
) -> None:
    """Compare a captured visualizer RGB frame against its golden image.

    On the first call for a new ``(test_name, physics_backend, visualizer_type, mode)``
    combination, the frame is saved as the new golden and the test is failed so that
    the generated file can be reviewed and committed before subsequent runs pass.

    Args:
        test_name: Scene name used as the golden-image subdirectory (e.g. ``"cartpole"``).
        physics_backend: Physics backend label (``"physx"`` or ``"newton"``).
        visualizer_type: Visualizer label (``"kit"`` or ``"newton"``).
        mode: Capture mode (``"viewport"`` or ``"tiled"``).
        frame: HxWx3 (or HxWxC) ``uint8``-range numpy array captured from the visualizer.
        comparison_scores: Module-level list that accumulates per-combination score entries
            for the HTML report and JUnit XML properties.
    """
    golden_image_dir = _GOLDEN_IMAGES_DIRECTORY / test_name
    golden_image_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_image_dir / f"{physics_backend}-{visualizer_type}-{mode}.png"

    rgb = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
    result_image = Image.fromarray(rgb)

    def _lookup_threshold(overrides: dict, scene: str, viz: str, m: str, fallback: dict, default: float) -> float:
        return overrides.get(f"{scene}-{viz}-{m}", overrides.get(f"{viz}-{m}", fallback.get(viz, default)))

    if not golden_path.exists():
        result_image.save(golden_path)
        _thresh = _lookup_threshold(
            _MAX_DIFF_PCT_OVERRIDES, test_name, visualizer_type, mode, MAX_DIFF_PCT_BY_VISUALIZER, 1.0
        )
        _ssim_thresh = _lookup_threshold(
            _SSIM_THRESHOLD_OVERRIDES, test_name, visualizer_type, mode, _SSIM_THRESHOLD_BY_VISUALIZER, 0.985
        )
        comparison_scores.append(
            {
                "test": test_name,
                "backend": physics_backend,
                "visualizer": visualizer_type,
                "mode": mode,
                "diff_pct": 0.0,
                "ssim": 0.0,
                "threshold": _thresh,
                "ssim_threshold": _ssim_thresh,
                "passed": False,
                "img_result_path": None,
                "img_golden_path": None,
            }
        )
        pytest.fail(
            f"Golden image not found for {test_name}/{physics_backend}-{visualizer_type}-{mode}.png.\n"
            f"Saved the current frame as the new golden at:\n  {golden_path}\n"
            "Review the image, commit it, then re-run the test."
        )

    try:
        golden_image = Image.open(golden_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        comparison_scores.append(
            {
                "test": test_name,
                "backend": physics_backend,
                "visualizer": visualizer_type,
                "mode": mode,
                "diff_pct": 0.0,
                "ssim": 0.0,
                "threshold": _lookup_threshold(
                    _MAX_DIFF_PCT_OVERRIDES, test_name, visualizer_type, mode, MAX_DIFF_PCT_BY_VISUALIZER, 1.0
                ),
                "ssim_threshold": _lookup_threshold(
                    _SSIM_THRESHOLD_OVERRIDES, test_name, visualizer_type, mode, _SSIM_THRESHOLD_BY_VISUALIZER, 0.985
                ),
                "passed": False,
                "img_result_path": None,
                "img_golden_path": None,
            }
        )
        pytest.fail(f"Failed to open golden image at {golden_path}: {exc}")

    max_diff_pct = _lookup_threshold(
        _MAX_DIFF_PCT_OVERRIDES, test_name, visualizer_type, mode, MAX_DIFF_PCT_BY_VISUALIZER, 1.0
    )
    ssim_threshold = _lookup_threshold(
        _SSIM_THRESHOLD_OVERRIDES, test_name, visualizer_type, mode, _SSIM_THRESHOLD_BY_VISUALIZER, 0.985
    )
    result_image_rgb = result_image.convert("RGB")

    succeeded, error_message, diff_pct, ssim_score = _compare_images(
        result_image_rgb, golden_image, max_diff_pct, ssim_threshold=ssim_threshold
    )

    entry: dict = {
        "test": test_name,
        "backend": physics_backend,
        "visualizer": visualizer_type,
        "mode": mode,
        "diff_pct": diff_pct,
        "ssim": ssim_score,
        "threshold": max_diff_pct,
        "ssim_threshold": ssim_threshold,
        "passed": succeeded,
        "img_result_path": None,
        "img_golden_path": None,
    }

    if diff_pct > 0:
        prefix = f"{test_name}-{physics_backend}-{visualizer_type}-{mode}"
        entry["img_result_path"] = _save_comparison_image(result_image_rgb, f"{prefix}-actual.png")
        entry["img_golden_path"] = _save_comparison_image(golden_image, f"{prefix}-golden.png")

    comparison_scores.append(entry)

    if not succeeded:
        pytest.fail(
            f"{test_name} (physics={physics_backend}, visualizer={visualizer_type}, mode={mode}) failed:\n"
            f"  {error_message}\n"
            f"Comparison images written to {_COMPARISON_IMAGES_DIR}."
        )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html_report(comparison_scores: list[dict], report_filename: str) -> None:
    """Write an HTML comparison report sorted by pixel-diff% descending.

    Args:
        comparison_scores: Accumulated score entries from :func:`validate_visualizer_frame`.
        report_filename: Output filename placed under ``tests/comparison-images/``.
    """
    if not comparison_scores:
        return

    os.makedirs(_COMPARISON_IMAGES_DIR, exist_ok=True)
    report_path = os.path.join(_COMPARISON_IMAGES_DIR, report_filename)
    sorted_scores = sorted(comparison_scores, key=lambda e: -e["diff_pct"])

    rows = []
    for entry in sorted_scores:
        status_class = "pass" if entry["passed"] else "fail"
        status_text = status_class.upper()

        actual_img_html = ""
        golden_img_html = ""
        if entry.get("img_result_path"):
            actual_fname = os.path.relpath(entry["img_result_path"], _COMPARISON_IMAGES_DIR)
            golden_fname = os.path.relpath(entry["img_golden_path"], _COMPARISON_IMAGES_DIR)
            actual_img_html = f'<a href="{actual_fname}"><img src="{actual_fname}" width="120" loading="lazy"></a>'
            golden_img_html = f'<a href="{golden_fname}"><img src="{golden_fname}" width="120" loading="lazy"></a>'

        rows.append(
            f'<tr class="{status_class}">'
            f"<td>{entry['test']}</td>"
            f"<td>{entry['backend']}</td>"
            f"<td>{entry['visualizer']}</td>"
            f"<td>{entry['mode']}</td>"
            f"<td>{entry['diff_pct']:.2f}</td>"
            f"<td>{entry['threshold']:.1f}</td>"
            f"<td>{entry['ssim']:.4f}</td>"
            f"<td>{entry['ssim_threshold']:.4f}</td>"
            f'<td class="status-{status_class}">{status_text}</td>'
            f"<td>{actual_img_html}</td>"
            f"<td>{golden_img_html}</td>"
            "</tr>"
        )

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Visualizer Golden Image - Comparison Report</title>\n"
        "<style>\n"
        "  body { font-family: sans-serif; font-size: 13px; margin: 16px; }\n"
        "  h1 { font-size: 1.3em; margin-bottom: 4px; }\n"
        "  p { margin-top: 4px; color: #555; }\n"
        "  table { border-collapse: collapse; width: 100%; }\n"
        "  th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: middle; }\n"
        "  th { background: #f0f0f0; white-space: nowrap; }\n"
        "  tr.fail { background: #fff0f0; }\n"
        "  tr.pass:hover, tr.fail:hover { filter: brightness(0.96); }\n"
        "  .status-pass { color: #2a7a2a; font-weight: bold; }\n"
        "  .status-fail { color: #cc0000; font-weight: bold; }\n"
        "  img { display: block; max-width: 120px; height: auto; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Visualizer Golden Image - Comparison Report</h1>\n"
        f"<p>Sorted by PixelDiff&nbsp;% (desc) &mdash; {len(sorted_scores)}&nbsp;total.</p>\n"
        "<table>\n"
        "<thead><tr>"
        "<th>Test</th>"
        "<th>Backend</th>"
        "<th>Visualizer</th>"
        "<th>Mode</th>"
        "<th>PixelDiff&nbsp;%</th>"
        "<th>PixelDiff Threshold&nbsp;%</th>"
        "<th>SSIM</th>"
        "<th>SSIM Threshold</th>"
        "<th>Status</th>"
        "<th>ACTUAL</th>"
        "<th>GOLDEN</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n"
        f"<p>Generated:&nbsp;{datetime.now().astimezone().isoformat(timespec='seconds')}.</p>\n"
        "</body>\n"
        "</html>\n"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(html)


# ---------------------------------------------------------------------------
# Pytest fixture factories  (mirrors rendering_test_utils.py factories)
# ---------------------------------------------------------------------------


def run_visualizer_golden_cartpole(
    physics_backend: str,
    visualizer_type: str,
    mode: str,
    comparison_scores: list[dict],
) -> None:
    """Run a golden-image test for one ``(physics_backend, visualizer_type, mode)`` combination.

    Imports :mod:`visualizer_integration_utils` lazily so this module remains importable before
    :class:`~isaaclab.app.AppLauncher` starts Isaac Sim.

    Args:
        physics_backend: ``"physx"`` or ``"newton"``.
        visualizer_type: ``"kit"`` (RTX viewport) or ``"newton"`` (OpenGL).
        mode: ``"viewport"`` (main viewer frame) or ``"tiled"`` (composite tiled camera).
        comparison_scores: Module-level accumulator forwarded to :func:`validate_visualizer_frame`.
    """
    import torch
    import visualizer_integration_utils as _viz_utils
    from isaaclab_visualizers.kit import KitVisualizer
    from isaaclab_visualizers.newton import NewtonVisualizer

    import isaaclab.sim as sim_utils

    def _get_active_visualizer(env, viz_type: str):
        cls = KitVisualizer if viz_type == "kit" else NewtonVisualizer
        matches = [v for v in env.sim.visualizers if isinstance(v, cls)]
        assert matches, f"Expected a {viz_type} visualizer in env.sim.visualizers."
        return matches[0]

    def _capture_frame(env, viz_type: str, capture_mode: str, backend: str, actions: torch.Tensor):
        if capture_mode == "tiled":
            viz = _get_active_visualizer(env, viz_type)
            return _viz_utils._capture_visualizer_tiled_camera_rgb(viz)

        if viz_type == "kit":
            kit_viz = _get_active_visualizer(env, "kit")
            return _viz_utils._capture_kit_viewport_with_pose_reapply(env, kit_viz)

        newton_viz = _get_active_visualizer(env, "newton")
        viewer = getattr(newton_viz, "_viewer", None)
        assert viewer is not None, "NewtonVisualizer did not create a viewer."
        _viz_utils._warm_newton_viewer(newton_viz, viewer)
        return _viz_utils._frame_to_numpy(viewer.get_frame())

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()
        tiled = mode == "tiled"
        env = _viz_utils._make_cartpole_camera_env(visualizer_type, physics_backend, tiled_camera=tiled)
        _viz_utils._configure_sim_for_visualizer_test(env)
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        # Reseed immediately before reset so the initial pole angle is reproducible
        # regardless of how many CUDA RNG samples prior tests consumed.
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        for _ in range(_viz_utils._START_BUFFER_STEPS):
            env.step(action=actions)

        frame = _capture_frame(env, visualizer_type, mode, physics_backend, actions)

        validate_visualizer_frame("cartpole", physics_backend, visualizer_type, mode, frame, comparison_scores)
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)


def run_visualizer_golden_shadow_hand(
    physics_backend: str,
    visualizer_type: str,
    mode: str,
    comparison_scores: list[dict],
) -> None:
    """Run a golden-image test for shadow hand + one ``(physics_backend, visualizer_type, mode)`` combination.

    Args:
        physics_backend: ``"physx"`` or ``"newton"``.
        visualizer_type: ``"kit"`` (RTX viewport) or ``"newton"`` (OpenGL).
        mode: ``"viewport"`` (main viewer frame) or ``"tiled"`` (composite tiled camera).
        comparison_scores: Module-level accumulator forwarded to :func:`validate_visualizer_frame`.
    """
    import torch
    import visualizer_integration_utils as _viz_utils
    from isaaclab_visualizers.kit import KitVisualizer
    from isaaclab_visualizers.newton import NewtonVisualizer

    import isaaclab.sim as sim_utils

    def _get_active_visualizer(env, viz_type: str):
        cls = KitVisualizer if viz_type == "kit" else NewtonVisualizer
        matches = [v for v in env.sim.visualizers if isinstance(v, cls)]
        assert matches, f"Expected a {viz_type} visualizer in env.sim.visualizers."
        return matches[0]

    def _capture_frame(env, viz_type: str, capture_mode: str, backend: str, actions: torch.Tensor):
        if capture_mode == "tiled":
            viz = _get_active_visualizer(env, viz_type)
            return _viz_utils._capture_visualizer_tiled_camera_rgb(viz)

        if viz_type == "kit":
            kit_viz = _get_active_visualizer(env, "kit")
            return _viz_utils._capture_kit_viewport_with_pose_reapply(
                env, kit_viz, resolution=_viz_utils._SHADOW_HAND_KIT_INTEGRATION_RENDER_RESOLUTION
            )

        newton_viz = _get_active_visualizer(env, "newton")
        viewer = getattr(newton_viz, "_viewer", None)
        assert viewer is not None, "NewtonVisualizer did not create a viewer."
        _viz_utils._warm_newton_viewer(newton_viz, viewer)
        return _viz_utils._frame_to_numpy(viewer.get_frame())

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()
        tiled = mode == "tiled"
        env = _viz_utils._make_shadow_hand_env(visualizer_type, physics_backend, tiled_camera=tiled)
        _viz_utils._configure_sim_for_visualizer_test(env)  # type: ignore[arg-type]
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        for _ in range(_viz_utils._START_BUFFER_STEPS):
            env.step(action=actions)

        frame = _capture_frame(env, visualizer_type, mode, physics_backend, actions)

        validate_visualizer_frame("shadow_hand", physics_backend, visualizer_type, mode, frame, comparison_scores)
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)


def run_visualizer_golden_anymal_d(
    physics_backend: str,
    visualizer_type: str,
    mode: str,
    comparison_scores: list[dict],
) -> None:
    """Run a golden-image test for AnymalD + one ``(physics_backend, visualizer_type, mode)`` combination.

    Args:
        physics_backend: ``"physx"`` or ``"newton"``.
        visualizer_type: ``"kit"`` (RTX viewport) or ``"newton"`` (OpenGL).
        mode: ``"viewport"`` (main viewer frame) or ``"tiled"`` (composite tiled camera).
        comparison_scores: Module-level accumulator forwarded to :func:`validate_visualizer_frame`.
    """
    import torch
    import visualizer_integration_utils as _viz_utils
    from isaaclab_visualizers.kit import KitVisualizer
    from isaaclab_visualizers.newton import NewtonVisualizer

    import isaaclab.sim as sim_utils

    def _get_active_visualizer(env, viz_type: str):
        cls = KitVisualizer if viz_type == "kit" else NewtonVisualizer
        matches = [v for v in env.sim.visualizers if isinstance(v, cls)]
        assert matches, f"Expected a {viz_type} visualizer in env.sim.visualizers."
        return matches[0]

    def _capture_frame(env, viz_type: str, capture_mode: str, backend: str, actions: torch.Tensor):
        if capture_mode == "tiled":
            viz = _get_active_visualizer(env, viz_type)
            return _viz_utils._capture_visualizer_tiled_camera_rgb(viz)

        if viz_type == "kit":
            kit_viz = _get_active_visualizer(env, "kit")
            return _viz_utils._capture_kit_viewport_with_pose_reapply(
                env, kit_viz, resolution=_viz_utils._ANYMAL_D_KIT_INTEGRATION_RENDER_RESOLUTION
            )

        newton_viz = _get_active_visualizer(env, "newton")
        viewer = getattr(newton_viz, "_viewer", None)
        assert viewer is not None, "NewtonVisualizer did not create a viewer."
        _viz_utils._warm_newton_viewer(newton_viz, viewer)
        return _viz_utils._frame_to_numpy(viewer.get_frame())

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()
        tiled = mode == "tiled"
        env = _viz_utils._make_anymal_d_env(visualizer_type, physics_backend, tiled_camera=tiled)
        _viz_utils._configure_sim_for_visualizer_test(env)  # type: ignore[arg-type]
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        for _ in range(_viz_utils._START_BUFFER_STEPS):
            env.step(action=actions)

        frame = _capture_frame(env, visualizer_type, mode, physics_backend, actions)

        validate_visualizer_frame("anymal_d", physics_backend, visualizer_type, mode, frame, comparison_scores)
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)


def run_visualizer_golden_franka_cloth(
    physics_backend: str,
    visualizer_type: str,
    mode: str,
    comparison_scores: list[dict],
) -> None:
    """Run a golden-image test for franka cloth + one ``(visualizer_type, mode)`` combination.

    Franka cloth uses Newton VBD cloth physics exclusively; *physics_backend* is
    accepted for API consistency with the other scene runners but must be ``"newton"``.

    Args:
        physics_backend: Must be ``"newton"``; franka cloth has no PhysX preset.
        visualizer_type: ``"kit"`` (RTX viewport) or ``"newton"`` (OpenGL).
        mode: ``"viewport"`` (main viewer frame) or ``"tiled"`` (composite tiled camera).
        comparison_scores: Module-level accumulator forwarded to :func:`validate_visualizer_frame`.
    """
    assert physics_backend == "newton", "FrankaCloth env has no PhysX preset; physics_backend must be 'newton'."

    import torch
    import visualizer_integration_utils as _viz_utils
    from isaaclab_visualizers.kit import KitVisualizer
    from isaaclab_visualizers.newton import NewtonVisualizer

    import isaaclab.sim as sim_utils

    def _get_active_visualizer(env, viz_type: str):
        cls = KitVisualizer if viz_type == "kit" else NewtonVisualizer
        matches = [v for v in env.sim.visualizers if isinstance(v, cls)]
        assert matches, f"Expected a {viz_type} visualizer in env.sim.visualizers."
        return matches[0]

    def _capture_frame(env, viz_type: str, capture_mode: str, actions: torch.Tensor):
        if capture_mode == "tiled":
            viz = _get_active_visualizer(env, viz_type)
            return _viz_utils._capture_visualizer_tiled_camera_rgb(viz)

        if viz_type == "kit":
            kit_viz = _get_active_visualizer(env, "kit")
            return _viz_utils._capture_kit_viewport_with_pose_reapply(
                env, kit_viz, resolution=_viz_utils._FRANKA_CLOTH_KIT_INTEGRATION_RENDER_RESOLUTION
            )

        newton_viz = _get_active_visualizer(env, "newton")
        viewer = getattr(newton_viz, "_viewer", None)
        assert viewer is not None, "NewtonVisualizer did not create a viewer."
        _viz_utils._warm_newton_viewer(newton_viz, viewer)
        return _viz_utils._frame_to_numpy(viewer.get_frame())

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()
        tiled = mode == "tiled"
        env = _viz_utils._make_franka_cloth_env(visualizer_type, tiled_camera=tiled)
        _viz_utils._configure_sim_for_visualizer_test(env)  # type: ignore[arg-type]
        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        for _ in range(_viz_utils._FRANKA_CLOTH_WARMUP_STEPS):
            env.step(action=actions)

        frame = _capture_frame(env, visualizer_type, mode, actions)

        validate_visualizer_frame("franka_cloth", physics_backend, visualizer_type, mode, frame, comparison_scores)
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)


def make_determinism_fixture():
    """Create an autouse fixture that seeds RNG determinism for each test."""

    @pytest.fixture(autouse=True)
    def _determinism_fixture():
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)

        yield

        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()

    return _determinism_fixture


def make_generate_html_report_fixture(comparison_scores: list[dict], report_filename: str):
    """Create a session-scoped fixture that writes the HTML report after all tests.

    Args:
        comparison_scores: Module-local comparison score list shared with test functions.
        report_filename: Output filename (placed under ``tests/comparison-images/``).
    """

    @pytest.fixture(scope="session", autouse=True)
    def _generate_html_report():
        yield
        generate_html_report(comparison_scores, report_filename)

    return _generate_html_report


def make_attach_comparison_properties_fixture(comparison_scores: list[dict]):
    """Create an autouse fixture that attaches per-combination scores as JUnit XML properties.

    Args:
        comparison_scores: Module-local comparison score list shared with test functions.
    """

    @pytest.fixture(autouse=True)
    def _attach_comparison_properties(request):
        initial_count = len(comparison_scores)
        yield
        for entry in comparison_scores[initial_count:]:
            label = f"{entry['backend']}-{entry['visualizer']}-{entry['mode']}"
            request.node.user_properties.append((f"diff_pct:{label}", f"{entry['diff_pct']:.2f}"))
            request.node.user_properties.append((f"ssim:{label}", f"{entry['ssim']:.4f}"))
            request.node.user_properties.append((f"threshold:{label}", f"{entry['threshold']:.1f}"))
            if entry.get("img_result_path"):
                request.node.user_properties.append((f"img_result:{label}", entry["img_result_path"]))
                request.node.user_properties.append((f"img_golden:{label}", entry["img_golden_path"]))

    return _attach_comparison_properties
