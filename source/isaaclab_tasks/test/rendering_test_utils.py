# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for rendering correctness tests."""

import os
from datetime import datetime
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops

# Directory containing golden images.
_GOLDEN_IMAGES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_images")

# Pixel L2 norm difference threshold. L2 norm difference is the Euclidean distance between two pixels:
#
#   d = sqrt((R1 - R2)^2 + (G1 - G2)^2 + (B1 - B2)^2)
#
# If the difference between two pixels is less than this threshold, consider them "equal" (i.e. within the tolerance).
#
_PIXEL_L2_NORM_DIFFERENCE_THRESHOLD = 10.0

# The max percentage of pixels allowed to differ. If the percentage exceeds this value, the test will fail.
# The value is set case by case based on the screen space taken up by the env in camera output images. It
# needs to be large enough to tolerate minor rendering noise while small enough to catch unexpected changes.
MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME = {
    "cartpole": 1.0,
    # Shadow-hand renderings (incl. ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0``) show up to
    # ~3.28 % per-pixel diff from anti-aliasing noise along the many finger/cube edges. 5.0 gives
    # headroom above that without masking real regressions, which the SSIM gate still catches.
    "shadow_hand": 5.0,
    # Texture aliasing artifacts on the ground (NVBUG#6116767)
    "dexsuite_kuka": 8.0,
}

# Minimum SSIM score below which two images are considered structurally different. SSIM is a perceptual metric
# robust to uniform per-pixel noise that penalises structural changes (geometry shifts, swapped colours, missing
# materials, etc.), so it complements the per-pixel L2 gate by catching regressions that survive a loosened pixel
# threshold.
_SSIM_THRESHOLD = 0.985

# Per-env SSIM overrides. Envs not listed fall back to ``_SSIM_THRESHOLD``. Loosened individually
# (not globally) to keep the strict gate active everywhere it already passes.
_SSIM_THRESHOLD_BY_ENV_NAME = {
    # Texture aliasing artifacts on the ground (NVBUG#6116767)
    "dexsuite_kuka": 0.95,
}

# Data types for which the SSIM gate is not enforced. SSIM assumes natural-image statistics and is unreliable on
# outputs where the per-pixel value distribution is highly non-uniform after normalisation (e.g. depth, where we
# divide by the max value so tiny absolute differences near the far plane dominate windowed variance). For these
# data types we still compute SSIM for reporting, but only the per-pixel L2 gate is used to decide pass/fail.
_SSIM_DISABLED_DATA_TYPES: set[str] = {"depth", "distance_to_camera", "distance_to_image_plane"}

# Directory for comparison images saved during the test session.
# Located under the pytest output root so it gets copied alongside test reports.
_COMPARISON_IMAGES_DIR = os.path.join(os.getcwd(), "tests", "comparison-images")
_COMPARISON_IMAGE_SUBDIR = "images"

# ---------------------------------------------------------------------------
# Parametrization: (physics_backend, renderer, data_type)
# ---------------------------------------------------------------------------

# OVRTX kitless paths can segfault on GitHub Actions runners; keep warp/Kit paths in CI.
_SKIP_ON_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
_SKIP_ON_GITHUB_ACTIONS_MARK = pytest.mark.skipif(
    _SKIP_ON_GITHUB_ACTIONS,
    reason="Skipped on GitHub Actions until the test can run on GitHub Actions.",
)

PHYSICS_RENDERER_AOV_COMBINATIONS = [
    # physx + isaacsim_rtx_renderer
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "rgb",
        id="physx-isaacsim_rtx-rgb",
    ),
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "albedo",
        id="physx-isaacsim_rtx-albedo",
    ),
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "depth",
        id="physx-isaacsim_rtx-depth",
    ),
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "simple_shading_constant_diffuse",
        id="physx-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "simple_shading_diffuse_mdl",
        id="physx-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "simple_shading_full_mdl",
        id="physx-isaacsim_rtx-simple_shading_full_mdl",
    ),
    pytest.param(
        "physx",
        "isaacsim_rtx_renderer",
        "semantic_segmentation",
        id="physx-isaacsim_rtx-semantic_segmentation",
    ),
    # newton + isaacsim_rtx_renderer
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "rgb",
        id="newton-isaacsim_rtx-rgb",
    ),
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "albedo",
        id="newton-isaacsim_rtx-albedo",
    ),
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "depth",
        id="newton-isaacsim_rtx-depth",
    ),
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "simple_shading_constant_diffuse",
        id="newton-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "simple_shading_diffuse_mdl",
        id="newton-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "simple_shading_full_mdl",
        id="newton-isaacsim_rtx-simple_shading_full_mdl",
    ),
    pytest.param(
        "newton",
        "isaacsim_rtx_renderer",
        "semantic_segmentation",
        id="newton-isaacsim_rtx-semantic_segmentation",
    ),
    # physx + newton_renderer (warp)
    pytest.param(
        "physx",
        "newton_renderer",
        "rgb",
        id="physx-newton_warp-rgb",
    ),
    pytest.param(
        "physx",
        "newton_renderer",
        "depth",
        id="physx-newton_warp-depth",
    ),
]

KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS = [
    # newton + ovrtx_renderer
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "rgb",
        id="newton-ovrtx-rgb",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "albedo",
        id="newton-ovrtx-albedo",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "depth",
        id="newton-ovrtx-depth",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "simple_shading_constant_diffuse",
        id="newton-ovrtx-simple_shading_constant_diffuse",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "simple_shading_diffuse_mdl",
        id="newton-ovrtx-simple_shading_diffuse_mdl",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "simple_shading_full_mdl",
        id="newton-ovrtx-simple_shading_full_mdl",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    pytest.param(
        "newton",
        "ovrtx_renderer",
        "semantic_segmentation",
        id="newton-ovrtx-semantic_segmentation",
        marks=_SKIP_ON_GITHUB_ACTIONS_MARK,
    ),
    # newton + newton_renderer (warp)
    pytest.param(
        "newton",
        "newton_renderer",
        "rgb",
        id="newton-newton_warp-rgb",
    ),
    pytest.param(
        "newton",
        "newton_renderer",
        "depth",
        id="newton-newton_warp-depth",
    ),
]


def maybe_save_stage(test_name: str, physics_backend: str, renderer: str, data_type: str) -> None:
    """If ``ISAAC_LAB_SAVE_STAGES`` is set, dump the current USD stage to that directory."""
    out_dir = os.environ.get("ISAAC_LAB_SAVE_STAGES")
    if not out_dir:
        return

    import isaaclab.sim as sim_utils

    os.makedirs(out_dir, exist_ok=True)
    safe_test_name = test_name.replace("/", "_")
    stage_path = os.path.join(out_dir, f"{safe_test_name}-{physics_backend}-{renderer}-{data_type}.usda")
    sim_utils.save_stage(stage_path, save_and_reload_in_place=False)
    print(f"[ISAAC_LAB_SAVE_STAGES] wrote {stage_path}")


def _apply_overrides_to_env_cfg(env_cfg: Any, override_args: list[str]) -> Any:
    """Apply override args to env_cfg using parse_overrides and apply_overrides."""
    from isaaclab_tasks.utils.hydra import apply_overrides, collect_presets, parse_overrides

    presets = {"env": collect_presets(env_cfg)}
    global_presets, preset_sel, preset_scalar, _ = parse_overrides(override_args, presets)
    hydra_cfg = {"env": env_cfg.to_dict()}
    env_cfg, _ = apply_overrides(env_cfg, None, hydra_cfg, global_presets, preset_sel, preset_scalar, presets)
    return env_cfg


def _physics_preset_name(physics_backend: str) -> str:
    """Translate the historical ``"newton"`` backend label (still used by golden-image
    filenames and ``pytest.param`` IDs) to the renamed Hydra preset
    ``"newton_mjwarp"``. Other labels (``"physx"`` etc.) pass through unchanged.
    """
    return "newton_mjwarp" if physics_backend == "newton" else physics_backend


def _normalize_tensor(tensor: torch.Tensor, data_type: str) -> torch.Tensor:
    """Convert camera output tensor to [0, 1] float32 for conversion to image."""
    normalized = tensor.float()

    if data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
        max_val = normalized.max()
        if max_val > 0:
            normalized = normalized / max_val
    elif data_type in {"albedo"}:
        normalized = normalized[..., :3] / 255.0
    else:
        normalized = normalized / 255.0

    return normalized


def _save_comparison_image(img: Image.Image, filename: str) -> str:
    """Save a PIL image under the comparison images directory."""
    path = os.path.join(_COMPARISON_IMAGES_DIR, _COMPARISON_IMAGE_SUBDIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="PNG")
    return path


def generate_html_report(comparison_scores: list[dict], report_filename: str) -> None:
    """Generate and save an HTML report of comparison scores."""
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

        ssim_checked = entry.get("ssim_checked", True)
        ssim_cell_class = "" if ssim_checked else ' class="ssim-disabled"'
        entry_ssim_threshold = entry.get("ssim_threshold", _SSIM_THRESHOLD)
        ssim_threshold_cell = f"{entry_ssim_threshold:.4f}" if ssim_checked else "N/A"
        ssim_title = "" if ssim_checked else ' title="SSIM gate disabled for this data type; score is informational."'

        rows.append(
            f'<tr class="{status_class}">'
            f"<td>{entry['test']}</td>"
            f"<td>{entry['backend']}</td>"
            f"<td>{entry['renderer']}</td>"
            f"<td>{entry['aov']}</td>"
            f"<td>{entry['diff_pct']:.2f}</td>"
            f"<td>{entry['threshold']:.1f}</td>"
            f"<td{ssim_cell_class}{ssim_title}>{entry['ssim']:.4f}</td>"
            f"<td{ssim_cell_class}{ssim_title}>{ssim_threshold_cell}</td>"
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
        "<title>Rendering Correctness - Image Comparison Report</title>\n"
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
        "  .ssim-disabled { color: #999; font-style: italic; }\n"
        "  img { display: block; max-width: 120px; height: auto; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Rendering Correctness - Image Comparison Report</h1>\n"
        f"<p>Sorted by PixelDiff&nbsp;% (desc) - {len(sorted_scores)}&nbsp; total.</p>\n"
        "<table>\n"
        "<thead><tr>"
        "<th>Test</th>"
        "<th>Backend</th>"
        "<th>Renderer</th>"
        "<th>AOV</th>"
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


def attach_comparison_properties(
    request: pytest.FixtureRequest, comparison_scores: list[dict], initial_count: int
) -> None:
    """Attach pixel-diff, SSIM scores, and failure images as JUnit XML properties."""
    for entry in comparison_scores[initial_count:]:
        label = f"{entry['backend']}-{entry['renderer']}-{entry['aov']}"
        request.node.user_properties.append((f"diff_pct:{label}", f"{entry['diff_pct']:.2f}"))
        ssim_value = f"{entry['ssim']:.4f}" if entry.get("ssim_checked", True) else f"{entry['ssim']:.4f} (N/A)"
        request.node.user_properties.append((f"ssim:{label}", ssim_value))
        request.node.user_properties.append((f"threshold:{label}", f"{entry['threshold']:.1f}"))
        if entry.get("img_result_path"):
            request.node.user_properties.append((f"img_result:{label}", entry["img_result_path"]))
            request.node.user_properties.append((f"img_golden:{label}", entry["img_golden_path"]))


def make_determinism_fixture():
    """Create an autouse fixture that enables determinism for each test."""

    @pytest.fixture(autouse=True)
    def _determinism_fixture():
        """Enable determinism for each test."""
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)

        yield

        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()

    return _determinism_fixture


def make_generate_html_report_fixture(comparison_scores: list[dict], report_filename: str):
    """Create a session fixture that writes the HTML report for one module.

    Args:
        comparison_scores: Module-local comparison score storage.
        report_filename: Output report filename.
    """

    @pytest.fixture(scope="session", autouse=True)
    def _generate_html_report():
        """Generate an HTML comparison report after all tests in the session complete."""
        yield
        generate_html_report(comparison_scores, report_filename)

    return _generate_html_report


def make_attach_comparison_properties_fixture(comparison_scores: list[dict]):
    """Create an autouse fixture that attaches JUnit properties for one module.

    Args:
        comparison_scores: Module-local comparison score storage.
    """

    @pytest.fixture(autouse=True)
    def _attach_comparison_properties(request):
        """Attach pixel-diff, SSIM scores, and failure images as JUnit XML properties."""
        initial_count = len(comparison_scores)
        yield
        attach_comparison_properties(request, comparison_scores, initial_count)

    return _attach_comparison_properties


def make_require_ovrtx_install_fixture():
    """Create an autouse fixture that fails fast when OVRTX is required but not installed.

    Only parametrized cases with ``renderer == "ovrtx_renderer"`` are checked (Newton
    Warp kitless cases do not need ``ov[ovrtx]``). Install with
    ``./isaaclab.sh -i 'ov[ovrtx]'`` (or the equivalent in your environment).
    """

    @pytest.fixture(autouse=True)
    def _require_ovrtx_install(request):
        callspec = getattr(request.node, "callspec", None)
        if callspec is None:
            return

        if callspec.params.get("renderer") != "ovrtx_renderer":
            return

        try:
            import ovrtx

            print(f"ovrtx version: {ovrtx.__version__}")
        except ImportError as exc:
            pytest.fail(
                "Kitless OVRTX rendering tests require the optional dependency ov[ovrtx]. "
                "Install with: ./isaaclab.sh -i 'ov[ovrtx]'\n"
                f"ImportError: {exc}"
            )

    return _require_ovrtx_install


def _make_grid(images: torch.Tensor) -> torch.Tensor:
    """Make a grid of images from a tensor of shape (B, H, W, C)."""
    from torchvision.utils import make_grid

    return make_grid(torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1), nrow=round(images.shape[0] ** 0.5))


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM between two (1, C, H, W) float tensors in [0, 1]."""
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
    """Compute the percentage of pixels whose L2 norm difference exceeds a threshold."""
    diff_array = np.array(ImageChops.difference(result_image, golden_image))
    l2_norm_array = np.linalg.norm(diff_array, axis=2)
    num_different_pixels = np.sum(l2_norm_array > pixel_diff_threshold)
    return 100.0 * num_different_pixels / l2_norm_array.size


def _compare_images(
    result_image: Image.Image,
    golden_image: Image.Image,
    max_different_pixels_percentage: float,
    check_ssim: bool = True,
    ssim_threshold: float = _SSIM_THRESHOLD,
) -> tuple[bool, str | None, float, float]:
    """Compare result and golden images against pixel and SSIM thresholds."""
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
            f"The percentage of different pixels ({diff_pct:.2f}%) exceeds the threshold of"
            f" {max_different_pixels_percentage:.2f}%. SSIM={ssim_score:.4f}.",
            diff_pct,
            ssim_score,
        )

    if check_ssim and ssim_score < ssim_threshold:
        return (
            False,
            f"SSIM ({ssim_score:.4f}) is below the threshold of {ssim_threshold:.4f}."
            f" Different pixels: {diff_pct:.2f}%.",
            diff_pct,
            ssim_score,
        )

    return True, None, diff_pct, ssim_score


def validate_camera_outputs(
    test_name: str,
    physics_backend: str,
    renderer: str,
    camera_outputs: dict[str, torch.Tensor],
    max_different_pixels_percentage: float,
    comparison_scores: list[dict],
) -> None:
    """Validate correctness and consistency of camera outputs."""
    assert len(camera_outputs) > 0, f"[{test_name}] No camera outputs produced by {physics_backend} + {renderer}."

    golden_image_dir = os.path.join(_GOLDEN_IMAGES_DIRECTORY, test_name)
    os.makedirs(golden_image_dir, exist_ok=True)

    ssim_threshold = _SSIM_THRESHOLD_BY_ENV_NAME.get(test_name, _SSIM_THRESHOLD)
    failed_data_types = {}

    for data_type, tensor in camera_outputs.items():
        condition = torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))
        corrected = torch.where(condition, torch.zeros_like(tensor), tensor)
        max_val = corrected.max()
        if max_val <= 0:
            failed_data_types[data_type] = f"Camera output '{data_type}' has no non-zero pixels."
            continue

        normalized = _normalize_tensor(corrected, data_type)
        grid = _make_grid(normalized)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        result_image = Image.fromarray(ndarr)

        golden_path = os.path.join(golden_image_dir, f"{physics_backend}-{renderer}-{data_type}.png")
        if not os.path.exists(golden_path):
            failed_data_types[data_type] = f"Golden image not found at {golden_path}."
            result_image.save(golden_path)
            continue

        try:
            golden_image = Image.open(golden_path)
        except Exception as error:  # noqa: BLE001
            failed_data_types[data_type] = f"Error opening golden image: {error}"
            continue

        check_ssim = data_type not in _SSIM_DISABLED_DATA_TYPES
        succeeded, error_message, diff_pct, ssim_score = _compare_images(
            result_image,
            golden_image,
            max_different_pixels_percentage,
            check_ssim=check_ssim,
            ssim_threshold=ssim_threshold,
        )

        entry = {
            "test": test_name,
            "backend": physics_backend,
            "renderer": renderer,
            "aov": data_type,
            "diff_pct": diff_pct,
            "ssim": ssim_score,
            "ssim_checked": check_ssim,
            "threshold": max_different_pixels_percentage,
            "ssim_threshold": ssim_threshold,
            "passed": succeeded,
            "img_result_path": None,
            "img_golden_path": None,
        }

        if diff_pct > 0:
            prefix = f"{test_name}-{physics_backend}-{renderer}-{data_type}"
            entry["img_result_path"] = _save_comparison_image(result_image, f"{prefix}-actual.png")
            entry["img_golden_path"] = _save_comparison_image(golden_image, f"{prefix}-golden.png")

        comparison_scores.append(entry)

        if not succeeded:
            failed_data_types[data_type] = error_message

    if failed_data_types:
        reason = f"{test_name} (physics={physics_backend}, renderer={renderer}) failed for the following data types:\n"
        for data_type, error_message in failed_data_types.items():
            reason += f"- {data_type}: {error_message}\n"
        reason += f"Comparison images were written to {_COMPARISON_IMAGES_DIR}."
        pytest.fail(reason)


def rendering_test_shadow_hand(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
) -> None:
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env import ShadowHandVisionEnv
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env_cfg import ShadowHandVisionEnvCfg

    override_args = [f"presets={_physics_preset_name(physics_backend)},{renderer},{data_type}"]

    env_cfg = ShadowHandVisionEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4

    if data_type == "depth":
        # Disable CNN forward pass as it cannot be meaningfully trained from depth alone and will raise a ValueError.
        env_cfg.feature_extractor.enabled = False

    env = None

    try:
        env = ShadowHandVisionEnv(env_cfg)
        maybe_save_stage("shadow_hand", physics_backend, renderer, data_type)

        validate_camera_outputs(
            "shadow_hand",
            physics_backend,
            renderer,
            env._tiled_camera.data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME["shadow_hand"],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None


def rendering_test_cartpole(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
) -> None:
    from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg

    env_cfg = CartpoleCameraPresetsEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(
        env_cfg, [f"presets={_physics_preset_name(physics_backend)},{renderer},{data_type}"]
    )

    env_cfg.scene.num_envs = 4

    env = None

    try:
        env = CartpoleCameraEnv(env_cfg)
        maybe_save_stage("cartpole", physics_backend, renderer, data_type)
        validate_camera_outputs(
            "cartpole",
            physics_backend,
            renderer,
            env._tiled_camera.data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME["cartpole"],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None


def rendering_test_dexsuite_kuka(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
) -> None:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
        DexsuiteKukaAllegroLiftEnvCfg,
    )

    override_args = [f"presets={_physics_preset_name(physics_backend)},{renderer},{data_type}64,single_camera,cube"]

    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4

    # Disable the observation point-cloud visualisation markers (/Visuals/ObservationPointCloud).
    # The underlying point sampling uses the global numpy/torch RNG, so marker positions shift
    # across processes and show up as random red dots in the rendered camera output. Since this
    # test only cares about rendering correctness of the scene itself, we hide the markers here
    # rather than making the sampler deterministic globally.
    point_cloud_term = getattr(env_cfg.observations.perception, "object_point_cloud", None)
    if point_cloud_term is not None:
        point_cloud_term.params["visualize"] = False

    # The success and failure markers are placed exactly at the same location. If both markers are
    # visible, the rendering order will determine which one is visible in the camera output. Hide
    # both markers to avoid this nondeterministic behavior.
    for marker_cfg in env_cfg.commands.object_pose.success_visualizer_cfg.markers.values():
        marker_cfg.visible = False

    env = None

    try:
        env = ManagerBasedRLEnv(env_cfg)
        maybe_save_stage("dexsuite_kuka", physics_backend, renderer, data_type)
        validate_camera_outputs(
            "dexsuite_kuka",
            physics_backend,
            renderer,
            env.scene.sensors["base_camera"].data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME["dexsuite_kuka"],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None
