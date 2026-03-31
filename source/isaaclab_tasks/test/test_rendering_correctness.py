# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rendering correctness.

Each test builds an environment with a given (physics_backend, renderer, data_type),
steps once, then checks if camera outputs are not blank (at least one non-zero
pixel) and consistent with golden images. Env-specific fixtures use parametrized
combinations; a separate test covers a list of registered task IDs that use
camera-based observations.
"""

# Launch Isaac Sim Simulator first.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

from isaaclab.envs.utils.spaces import sample_space  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from isaaclab_tasks.utils.hydra import (  # noqa: E402
    apply_overrides,
    collect_presets,
    parse_overrides,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directory containing golden images.
_GOLDEN_IMAGES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_images")

# Pixel L2 norm difference threshold. L2 norm difference is the Euclidean distance between two pixels:
#
#   d = sqrt((R1 - R2)^2 + (G1 - G2)^2 + (B1 - B2)^2)
#
# If the difference between two pixels is less than this threshold, consider them "equal" (i.e. within the tolerance).
#
_PIXEL_L2_NORM_DIFFERENCE_THRESHOLD = 10.0

_OVRTX_DISABLED = pytest.mark.skip(
    reason="OVRTX is optional and experimental feature and temporarily is excluded from testing."
)

# Directory for comparison images saved during the test session.
# Located under the pytest output root so it gets copied alongside test reports.
_COMPARISON_IMAGES_DIR = os.path.join(os.getcwd(), "tests", "comparison-images")

# Collects comparison scores from all golden-image comparisons during the session.
# Each entry: {"test": str, "backend": str, "renderer": str, "aov": str,
#              "ssim": float, "diff_pct": float, "passed": bool,
#              "img_result_path": str | None, "img_golden_path": str | None}
_COMPARISON_SCORES: list[dict] = []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Fixture to clear SimulationContext after each test.

    SimulationContext is a singleton; tests that create envs leave it set. Without
    cleanup, later tests can see stale context or fail when the instance is
    reused. The fixture runs after every test and calls clear_instance() so each
    test runs with a clean simulation context and tests stay isolated.
    """
    yield

    SimulationContext.clear_instance()


@pytest.fixture(scope="session", autouse=True)
def _generate_comparison_html_report():
    """Generate an HTML comparison report after all tests in the session complete."""
    yield
    try:
        _generate_html_report()
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(f"Failed to generate HTML comparison report: {exc}", stacklevel=1)


@pytest.fixture(autouse=True)
def _attach_comparison_properties(request):
    """Attach pixel-diff, SSIM scores, and failure images as JUnit XML properties."""
    initial_count = len(_COMPARISON_SCORES)
    yield
    for entry in _COMPARISON_SCORES[initial_count:]:
        label = f"{entry['backend']}-{entry['renderer']}-{entry['aov']}"
        request.node.user_properties.append((f"diff_pct:{label}", f"{entry['diff_pct']:.2f}"))
        request.node.user_properties.append((f"ssim:{label}", f"{entry['ssim']:.4f}"))
        request.node.user_properties.append((f"threshold:{label}", f"{entry['threshold']:.1f}"))
        if entry.get("img_result_path"):
            request.node.user_properties.append((f"img_result:{label}", entry["img_result_path"]))
            request.node.user_properties.append((f"img_golden:{label}", entry["img_golden_path"]))


# ---------------------------------------------------------------------------
# Parametrization: (physics_backend, renderer, data_type)
# ---------------------------------------------------------------------------

_PHYSICS_RENDERER_AOV_COMBINATIONS = [
    # physx + isaacsim_rtx_renderer
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "rgb"),
        id="physx-isaacsim_rtx-rgb",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "albedo"),
        id="physx-isaacsim_rtx-albedo",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "depth"),
        id="physx-isaacsim_rtx-depth",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "simple_shading_constant_diffuse"),
        id="physx-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl"),
        id="physx-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "simple_shading_full_mdl"),
        id="physx-isaacsim_rtx-simple_shading_full_mdl",
    ),
    pytest.param(
        ("physx", "isaacsim_rtx_renderer", "semantic_segmentation"),
        id="physx-isaacsim_rtx-semantic_segmentation",
    ),
    # physx + newton_renderer (warp)
    pytest.param(
        ("physx", "newton_renderer", "rgb"),
        id="physx-newton_warp-rgb",
    ),
    pytest.param(
        ("physx", "newton_renderer", "depth"),
        id="physx-newton_warp-depth",
    ),
    # newton + isaacsim_rtx_renderer
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "rgb"),
        id="newton-isaacsim_rtx-rgb",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "albedo"),
        id="newton-isaacsim_rtx-albedo",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "depth"),
        id="newton-isaacsim_rtx-depth",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "simple_shading_constant_diffuse"),
        id="newton-isaacsim_rtx-simple_shading_constant_diffuse",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl"),
        id="newton-isaacsim_rtx-simple_shading_diffuse_mdl",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "simple_shading_full_mdl"),
        id="newton-isaacsim_rtx-simple_shading_full_mdl",
    ),
    pytest.param(
        ("newton", "isaacsim_rtx_renderer", "semantic_segmentation"),
        id="newton-isaacsim_rtx-semantic_segmentation",
    ),
    # newton + newton_renderer (warp)
    pytest.param(
        ("newton", "newton_renderer", "rgb"),
        id="newton-newton_warp-rgb",
    ),
    pytest.param(
        ("newton", "newton_renderer", "depth"),
        id="newton-newton_warp-depth",
    ),
    # newton + ovrtx_renderer
    pytest.param(
        ("newton", "ovrtx_renderer", "rgb"),
        id="newton-ovrtx-rgb",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "albedo"),
        id="newton-ovrtx-albedo",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "depth"),
        id="newton-ovrtx-depth",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_constant_diffuse"),
        id="newton-ovrtx-simple_shading_constant_diffuse",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_diffuse_mdl"),
        id="newton-ovrtx-simple_shading_diffuse_mdl",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "simple_shading_full_mdl"),
        id="newton-ovrtx-simple_shading_full_mdl",
        marks=_OVRTX_DISABLED,
    ),
    pytest.param(
        ("newton", "ovrtx_renderer", "semantic_segmentation"),
        id="newton-ovrtx-semantic_segmentation",
        marks=_OVRTX_DISABLED,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_overrides_to_env_cfg(env_cfg: Any, override_args: list[str]) -> Any:
    """Apply override args to env_cfg using parse_overrides and apply_overrides.

    Args:
        env_cfg: Environment config to mutate (supports :meth:`to_dict`).
        override_args: List of override strings (e.g. ``["presets=physx,isaacsim_rtx_renderer,rgb"]``).

    Returns:
        The resolved env_cfg (possibly a different object if root preset was applied).
    """
    presets = {"env": collect_presets(env_cfg)}
    global_presets, preset_sel, preset_scalar, _ = parse_overrides(override_args, presets)
    hydra_cfg = {"env": env_cfg.to_dict()}
    env_cfg, _ = apply_overrides(env_cfg, None, hydra_cfg, global_presets, preset_sel, preset_scalar, presets)
    return env_cfg


def _normalize_tensor(tensor: torch.Tensor, data_type: str) -> torch.Tensor:
    """Convert camera output tensor to [0, 1] float32 for conversion to image.

    Args:
        tensor: Camera output tensor.
        data_type: Data type of the camera output.

    Returns:
        Normalized tensor.
    """
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
    """Save a PIL image to the comparison-images/images directory.

    Args:
        img: PIL Image to save.
        filename: File name (e.g. ``"test-backend-renderer-aov-result.png"``).

    Returns:
        Absolute path to the saved file.
    """
    path = os.path.join(_COMPARISON_IMAGES_DIR, "images", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="PNG")
    return path


def _generate_html_report() -> None:
    """Generate an HTML report of all comparison scores and save it alongside the comparison images.

    The report is written to ``<_COMPARISON_IMAGES_DIR>/_report_.html`` and includes a table of
    all image comparison results sorted by PixelDiff % descending, with thumbnail links to actual
    and golden images where available.
    """
    if not _COMPARISON_SCORES:
        return

    os.makedirs(_COMPARISON_IMAGES_DIR, exist_ok=True)
    report_path = os.path.join(_COMPARISON_IMAGES_DIR, "_report_.html")

    sorted_scores = sorted(_COMPARISON_SCORES, key=lambda e: -e["diff_pct"])

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
            f"<td>{entry['renderer']}</td>"
            f"<td>{entry['aov']}</td>"
            f"<td>{entry['diff_pct']:.2f}</td>"
            f"<td>{entry['threshold']:.1f}</td>"
            f"<td>{entry['ssim']:.4f}</td>"
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
        "<title>Rendering Correctness — Image Comparison Report</title>\n"
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
        "<h1>Rendering Correctness — Image Comparison Report</h1>\n"
        f"<p>Sorted by PixelDiff&nbsp;% (desc) &mdash; {len(sorted_scores)}&nbsp; total.</p>\n"
        "<table>\n"
        "<thead><tr>"
        "<th>Test</th>"
        "<th>Backend</th>"
        "<th>Renderer</th>"
        "<th>AOV</th>"
        "<th>PixelDiff&nbsp;%</th>"
        "<th>Threshold&nbsp;%</th>"
        "<th>SSIM</th>"
        "<th>Status</th>"
        "<th>ACTUAL</th>"
        "<th>GOLDEN</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n</body>\n</html>\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def _make_grid(images: torch.Tensor) -> torch.Tensor:
    """Make a grid of images from a tensor of shape (B, H, W, C).

    Args:
        images: A tensor of shape (B, H, W, C) containing the images.

    Returns:
        A tensor of shape (H, W, C) containing the grid of images.
    """
    from torchvision.utils import make_grid

    return make_grid(torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1), nrow=round(images.shape[0] ** 0.5))


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM between two (1, C, H, W) float tensors in [0, 1].

    https://en.wikipedia.org/wiki/Structural_similarity_index_measure

    Uses a uniform averaging window and the standard SSIM constants (K1=0.01, K2=0.03,
    data_range=1.0).

    Args:
        img1: First image tensor of shape (1, C, H, W) in [0, 1].
        img2: Second image tensor of shape (1, C, H, W) in [0, 1].
        window_size: Side length of the square averaging window.

    Returns:
        Mean SSIM score (1.0 = identical).
    """
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
    """Compute the percentage of pixels whose L2 norm difference exceeds a threshold.

    Args:
        result_image: Result image as PIL Image.
        golden_image: Golden image as PIL Image (must be same size/mode).
        pixel_diff_threshold: Pixel L2 norm difference threshold.

    Returns:
        Percentage of pixels that differ beyond the threshold.
    """
    diff_array = np.array(ImageChops.difference(result_image, golden_image))
    l2_norm_array = np.linalg.norm(diff_array, axis=2)
    num_different_pixels = np.sum(l2_norm_array > pixel_diff_threshold)
    return 100.0 * num_different_pixels / l2_norm_array.size


def _compare_images(
    result_image: Image.Image,
    golden_image: Image.Image,
    max_different_pixels_percentage: float,
) -> tuple[bool, str | None, float, float]:
    """Compare result and golden images using pixel L2 norm (pass/fail) and SSIM (reference).

    Args:
        result_image: Result image as PIL Image to compare with golden image.
        golden_image: Golden image as PIL Image to compare with result image.
        max_different_pixels_percentage: Maximum percentage of pixels allowed to exceed pixel_diff_threshold.

    Returns:
        (passed, error_message_or_None, diff_percentage, ssim_score).
        Scores are 0.0 / 0.0 when comparison cannot be performed (size/mode mismatch).
    """
    if result_image.size != golden_image.size:
        return False, f"Size mismatch: expected {golden_image.size}, got {result_image.size}.", 0.0, 0.0

    if result_image.mode != golden_image.mode:
        return False, f"Mode mismatch: expected {golden_image.mode}, got {result_image.mode}.", 0.0, 0.0

    diff_pct = _pixel_diff_percentage(result_image, golden_image)

    # SSIM (reference only, not used for pass/fail).
    result_tensor = torch.from_numpy(np.array(result_image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    golden_tensor = torch.from_numpy(np.array(golden_image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    ssim_score = _ssim(result_tensor, golden_tensor)

    if diff_pct > max_different_pixels_percentage:
        return (
            False,
            f"The percentage of different pixels ({diff_pct:.2f}%) exceeds the threshold of"
            f" {max_different_pixels_percentage:.2f}%. SSIM={ssim_score:.4f} (reference).",
            diff_pct,
            ssim_score,
        )

    return True, None, diff_pct, ssim_score


def _validate_camera_outputs(
    test_name: str,
    physics_backend: str,
    renderer: str,
    camera_outputs: dict[str, torch.Tensor],
    max_different_pixels_percentage: float,
) -> None:
    """Validate correctness and consistency of camera outputs.

    Args:
        test_name: Test name.
        physics_backend: Physics backend.
        renderer: Renderer.
        camera_outputs: {data_type -> tensor}.
        max_different_pixels_percentage: Maximum percentage of pixels allowed to exceed pixel_diff_threshold.
    """
    assert len(camera_outputs) > 0, f"[{test_name}] No camera outputs produced by {physics_backend} + {renderer}."

    golden_image_dir = os.path.join(_GOLDEN_IMAGES_DIRECTORY, test_name)
    os.makedirs(golden_image_dir, exist_ok=True)

    for data_type, tensor in camera_outputs.items():
        # Replace inf/nan with zero so they do not break comparison; ensure the tensor has at least one non-zero value.
        condition = torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))
        corrected = torch.where(condition, torch.zeros_like(tensor), tensor)
        max_val = corrected.max()
        assert max_val > 0, (
            f"[{test_name}] Camera output '{data_type}' has no non-zero pixels. "
            f"Shape: {corrected.shape}, dtype: {corrected.dtype}."
        )

        # convert tensors to a tiled image.
        normalized = _normalize_tensor(corrected, data_type)
        grid = _make_grid(normalized)

        # permute(1, 2, 0) is there to change the tensor layout from channel-first to channel-last so it matches what
        # PIL expects.
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        result_image = Image.fromarray(ndarr)

        # first run creates baseline and fails; second run validates.
        golden_path = os.path.join(golden_image_dir, f"{physics_backend}-{renderer}-{data_type}.png")
        if not os.path.exists(golden_path):
            result_image.save(golden_path)
            pytest.fail(
                f"[{test_name}] Golden image not found at {golden_path}. Saved result image to {golden_path}. "
                "Please run the test again to validate the consistency of rendering outputs."
            )

        try:
            golden_image = Image.open(golden_path)
        except Exception as e:
            pytest.fail(f"Error opening golden image: {e}")

        # validate the consistency of rendering outputs.
        succeeded, error_message, diff_pct, ssim_score = _compare_images(
            result_image, golden_image, max_different_pixels_percentage
        )

        entry = {
            "test": test_name,
            "backend": physics_backend,
            "renderer": renderer,
            "aov": data_type,
            "diff_pct": diff_pct,
            "ssim": ssim_score,
            "threshold": max_different_pixels_percentage,
            "passed": succeeded,
            "img_result_path": None,
            "img_golden_path": None,
        }

        if diff_pct > 0:
            prefix = f"{test_name}-{physics_backend}-{renderer}-{data_type}"
            entry["img_result_path"] = _save_comparison_image(result_image, f"{prefix}-actual.png")
            entry["img_golden_path"] = _save_comparison_image(golden_image, f"{prefix}-golden.png")

        _COMPARISON_SCORES.append(entry)

        if not succeeded:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            result_path = os.path.join(golden_image_dir, f"{physics_backend}-{renderer}-{data_type}-{timestamp}.png")
            result_image.save(result_path)
            pytest.fail(
                f"[{test_name}] Inconsistency detected for camera output '{data_type}': {error_message}. "
                f"Saved result image to {result_path} for further investigation. "
                f"If result image is correct, please replace the golden image at {golden_path} with the result image."
            )


def _collect_camera_outputs(env: object) -> dict[str, dict[str, torch.Tensor]]:
    """Collect camera outputs from env.scene.sensors.

    Args:
        env: Gymnasium env (or any object with optional unwrapped.scene.sensors).

    Returns:
        Nested dict: sensor name -> {data_type -> tensor} for non-empty tensor outputs.
    """
    base = getattr(env, "unwrapped", env)
    out = {}

    scene = getattr(base, "scene", None)
    if scene is not None:
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            for name, sensor in sensors.items():
                data = getattr(sensor, "data", None)
                output = getattr(data, "output", None) if data is not None else None
                if not isinstance(output, dict):
                    continue

                # Collect only tensor entries (ignore empty or lazy-unfilled)
                tensor_output = {k: v for k, v in output.items() if isinstance(v, torch.Tensor) and v.numel() > 0}
                if tensor_output:
                    out[name] = tensor_output

    return out


# ---------------------------------------------------------------------------
# Shadow Hand vision env
# ---------------------------------------------------------------------------


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def shadow_hand_env(request):
    """Build Shadow Hand vision env for (physics_backend, renderer, data_type); step once, yield, close."""
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env import ShadowHandVisionEnv
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_vision_env_cfg import ShadowHandVisionEnvCfg

    physics_backend, renderer, data_type = request.param

    override_args = [f"presets={physics_backend},{renderer},{data_type}"]

    env_cfg = ShadowHandVisionEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4
    env_cfg.seed = 42

    if data_type == "depth":
        # Disable CNN forward pass as it cannot be meaningfully trained from depth alone and will raise a ValueError.
        env_cfg.feature_extractor.enabled = False

    env = None
    try:
        env = ShadowHandVisionEnv(env_cfg)
        env.reset()
        actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
        env.step(actions)
        yield physics_backend, renderer, data_type, env
    finally:
        if env is not None:
            env.close()


def test_shadow_hand(shadow_hand_env):
    """Camera output must contain at least one non-zero pixel (Shadow Hand vision env)."""
    physics_backend, renderer, _, env = shadow_hand_env

    _validate_camera_outputs(
        "shadow_hand",
        physics_backend,
        renderer,
        env._tiled_camera.data.output,
        max_different_pixels_percentage=5.0,
    )


# ---------------------------------------------------------------------------
# Cartpole camera env
# ---------------------------------------------------------------------------


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def cartpole_env(request):
    """Build Cartpole camera env for (physics_backend, renderer, data_type); step once, yield, close."""
    from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg

    physics_backend, renderer, data_type = request.param

    override_args = [f"presets={physics_backend},{renderer},{data_type}"]

    env_cfg = CartpoleCameraPresetsEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4
    env_cfg.seed = 42

    env = None
    try:
        env = CartpoleCameraEnv(env_cfg)
        env.reset()
        actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
        env.step(actions)
        yield physics_backend, renderer, data_type, env
    finally:
        if env is not None:
            env.close()


def test_cartpole(cartpole_env):
    """Camera output must contain at least one non-zero pixel (Cartpole camera env)."""
    physics_backend, renderer, _, env = cartpole_env

    _validate_camera_outputs(
        "cartpole",
        physics_backend,
        renderer,
        env._tiled_camera.data.output,
        max_different_pixels_percentage=5.0,
    )


# ---------------------------------------------------------------------------
# Dexsuite Kuka-Allegro Lift (single camera)
# ---------------------------------------------------------------------------


@pytest.fixture(params=_PHYSICS_RENDERER_AOV_COMBINATIONS)
def dexsuite_kuka_allegro_lift_env(request):
    """Build Dexsuite Kuka-Allegro Lift (single camera) for backend/renderer/data_type; step once, yield, close."""
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
        DexsuiteKukaAllegroLiftEnvCfg,
    )

    physics_backend, renderer, data_type = request.param

    # Dexsuite data type has explicit resolution suffix (64, 128, 256). We only test 64x64.
    override_args = [f"presets={physics_backend},{renderer},{data_type}64,single_camera,cube"]

    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4
    env_cfg.seed = 42

    env = None
    try:
        env = ManagerBasedRLEnv(env_cfg)
        env.reset()
        actions = torch.zeros(env_cfg.scene.num_envs, env.action_space.shape[-1], device=env.device)
        env.step(actions)
        yield physics_backend, renderer, data_type, env
    finally:
        if env is not None:
            env.close()


def test_dexsuite_kuka_allegro_lift(dexsuite_kuka_allegro_lift_env):
    """Camera output must contain at least one non-zero pixel (Dexsuite Kuka-Allegro Lift, single camera)."""
    physics_backend, renderer, _, env = dexsuite_kuka_allegro_lift_env

    _validate_camera_outputs(
        "dexsuite_kuka",
        physics_backend,
        renderer,
        env.scene.sensors["base_camera"].data.output,
        max_different_pixels_percentage=10.0,
    )


# ---------------------------------------------------------------------------
# Registered tasks (camera-based observations)
# ---------------------------------------------------------------------------

# Task IDs that expose camera/tiled_camera image observations; each is validated for non-blank rendering.
_RENDER_CORRECTNESS_TASK_IDS = [
    "Isaac-Cartpole-Albedo-Camera-Direct-v0",
    "Isaac-Cartpole-Camera-Presets-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0",
    "Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0",
    "Isaac-Cartpole-Depth-Camera-Direct-v0",
    "Isaac-Cartpole-Depth-v0",
    "Isaac-Cartpole-RGB-Camera-Direct-v0",
    "Isaac-Cartpole-RGB-ResNet18-v0",
    "Isaac-Cartpole-RGB-v0",
    "Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
    "Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
    "Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
]


@pytest.mark.parametrize("task_id", _RENDER_CORRECTNESS_TASK_IDS)
def test_registered_tasks(task_id):
    """Camera output must be non-empty for each registered task with camera-based observations."""
    env = None
    try:
        env_cfg = parse_env_cfg(task_id, num_envs=4)
        env_cfg.seed = 42

        env = gym.make(task_id, cfg=env_cfg)
        unwrapped: Any = env.unwrapped
        sim = getattr(unwrapped, "sim", None)
        if sim is not None:
            sim._app_control_on_stop_handle = None

        env.reset()

        num_envs = getattr(unwrapped, "num_envs", 4)
        device = getattr(unwrapped, "device", None)

        if getattr(unwrapped, "possible_agents", None):
            action_spaces = getattr(unwrapped, "action_spaces", {})
            actions = {
                agent: sample_space(
                    action_spaces[agent],
                    device=device,
                    batch_size=num_envs,
                    fill_value=0,
                )
                for agent in unwrapped.possible_agents
            }
        else:
            actions = sample_space(
                getattr(unwrapped, "single_action_space", None),
                device=device,
                batch_size=num_envs,
                fill_value=0,
            )

        env.step(actions)

        camera_outputs_nested_dict = _collect_camera_outputs(env)
        num_camera_outputs = len(camera_outputs_nested_dict)
        assert num_camera_outputs == 1, f"[{task_id}] Expected 1 camera output, got {num_camera_outputs}."

        camera_outputs = next(iter(camera_outputs_nested_dict.values()))

        _validate_camera_outputs(
            f"registered_tasks/{task_id}",
            "default_physics",
            "default_renderer",
            camera_outputs,
            max_different_pixels_percentage=5.0,
        )
    finally:
        if env is not None:
            env.close()
