# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden-image comparison shared by renderer and visualizer tests."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
from PIL import Image, ImageChops

from isaaclab.utils.images import make_camera_output_grid, normalize_camera_output_for_display


@dataclass(frozen=True)
class ImageComparison:
    """Result of one pixel-difference and SSIM comparison."""

    label: str
    passed: bool
    diff_pct: float
    ssim: float
    max_diff_pct: float
    min_ssim: float | None
    error: str | None = None
    actual_path: Path | None = None
    golden_path: Path | None = None

    def record(self, request: Any) -> None:
        """Attach compact diagnostics to a pytest JUnit node."""
        request.node.user_properties.extend(
            [
                (f"diff_pct:{self.label}", f"{self.diff_pct:.2f}"),
                (f"ssim:{self.label}", f"{self.ssim:.4f}"),
                (f"threshold:{self.label}", f"{self.max_diff_pct:.2f}"),
            ]
        )
        if self.actual_path is not None:
            request.node.user_properties.extend(
                [
                    (f"img_result:{self.label}", str(self.actual_path)),
                    (f"img_golden:{self.label}", str(self.golden_path)),
                ]
            )

    def assert_passed(self) -> None:
        """Raise a focused assertion after diagnostics have been recorded."""
        assert self.passed, self.error


def frame_image(frame: Image.Image | np.ndarray | torch.Tensor | wp.array) -> Image.Image:
    """Convert a visualizer frame to an RGB PIL image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, wp.array):
        frame = wp.to_torch(frame)
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    array = array[..., :3]
    if array.size and np.nanmax(array) <= 1.0 + 1.0e-6:
        array = array * 255.0
    return Image.fromarray(np.nan_to_num(array).clip(0, 255).astype(np.uint8), mode="RGB")


def camera_output_image(output: torch.Tensor | wp.array | Any, data_type: str) -> Image.Image:
    """Normalize a batched camera AOV and compose its views into a PIL image."""
    tensor = output if torch.is_tensor(output) else output.torch
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if tensor.numel() == 0 or tensor.abs().max().item() == 0:
        raise AssertionError(f"Camera output '{data_type}' is empty or contains only zeroes.")
    grid = make_camera_output_grid(normalize_camera_output_for_display(tensor, data_type))
    array = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(array)


def compare_to_golden(
    image: Image.Image,
    golden_path: str | Path,
    *,
    label: str,
    artifact_dir: str | Path,
    max_diff_pct: float,
    min_ssim: float | None = 0.985,
    pixel_l2_threshold: float = 10.0,
    alpha_only: bool = False,
) -> ImageComparison:
    """Compare an image to a baseline, bootstrapping a missing baseline for review."""
    golden_path = Path(golden_path)
    artifact_dir = Path(artifact_dir)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ISAACLAB_UPDATE_GOLDENS") == "1":
        image.save(golden_path)
        return ImageComparison(label, True, 0.0, 1.0, max_diff_pct, min_ssim, golden_path=golden_path)
    if not golden_path.exists():
        image.save(golden_path)
        return ImageComparison(
            label,
            False,
            0.0,
            0.0,
            max_diff_pct,
            min_ssim,
            f"Golden image was missing. Review the generated baseline at {golden_path}.",
            golden_path=golden_path,
        )

    try:
        golden = Image.open(golden_path)
        golden.load()
    except Exception as exc:
        return ImageComparison(
            label,
            False,
            0.0,
            0.0,
            max_diff_pct,
            min_ssim,
            f"Could not read golden image {golden_path}: {exc}",
            golden_path=golden_path,
        )

    actual_for_comparison = _alpha_only(image) if alpha_only else image
    golden_for_comparison = _alpha_only(golden) if alpha_only else golden
    error: str | None = None
    diff_pct = 0.0
    ssim = 0.0
    if actual_for_comparison.size != golden_for_comparison.size:
        error = f"Size mismatch: expected {golden.size}, got {image.size}."
    elif actual_for_comparison.mode != golden_for_comparison.mode:
        error = f"Mode mismatch: expected {golden.mode}, got {image.mode}."
    else:
        diff_pct = _pixel_diff_percentage(actual_for_comparison, golden_for_comparison, pixel_l2_threshold)
        ssim = _ssim(actual_for_comparison, golden_for_comparison)
        if diff_pct > max_diff_pct:
            error = f"Pixel difference {diff_pct:.2f}% exceeds {max_diff_pct:.2f}% (SSIM {ssim:.4f})."
        elif min_ssim is not None and ssim < min_ssim:
            error = f"SSIM {ssim:.4f} is below {min_ssim:.4f} (pixel difference {diff_pct:.2f}%)."

    actual_path = None
    if error is not None or diff_pct > 0:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        actual_path = artifact_dir / f"{safe_label}-actual.png"
        saved_golden_path = artifact_dir / f"{safe_label}-golden.png"
        image.save(actual_path)
        golden.save(saved_golden_path)
    else:
        saved_golden_path = None

    return ImageComparison(
        label,
        error is None,
        diff_pct,
        ssim,
        max_diff_pct,
        min_ssim,
        error,
        actual_path,
        saved_golden_path,
    )


def _alpha_only(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    zero = alpha.point(lambda _: 0)
    return Image.merge("RGBA", (zero, zero, zero, alpha))


def _pixel_diff_percentage(actual: Image.Image, golden: Image.Image, threshold: float) -> float:
    difference = np.asarray(ImageChops.difference(actual, golden), dtype=np.float32)
    if difference.ndim == 2:
        difference = difference[..., None]
    different = np.linalg.norm(difference, axis=-1) > threshold
    return 100.0 * float(np.count_nonzero(different)) / different.size


def _ssim(actual: Image.Image, golden: Image.Image, window_size: int = 11) -> float:
    actual_tensor = _image_tensor(actual)
    golden_tensor = _image_tensor(golden)
    channels = actual_tensor.shape[1]
    kernel = torch.full(
        (channels, 1, window_size, window_size),
        1.0 / (window_size * window_size),
        dtype=torch.float32,
    )
    padding = window_size // 2
    mu_actual = torch.nn.functional.conv2d(actual_tensor, kernel, padding=padding, groups=channels)
    mu_golden = torch.nn.functional.conv2d(golden_tensor, kernel, padding=padding, groups=channels)
    actual_sq = mu_actual.square()
    golden_sq = mu_golden.square()
    actual_golden = mu_actual * mu_golden
    sigma_actual = (
        torch.nn.functional.conv2d(actual_tensor.square(), kernel, padding=padding, groups=channels) - actual_sq
    )
    sigma_golden = (
        torch.nn.functional.conv2d(golden_tensor.square(), kernel, padding=padding, groups=channels) - golden_sq
    )
    sigma_cross = (
        torch.nn.functional.conv2d(actual_tensor * golden_tensor, kernel, padding=padding, groups=channels)
        - actual_golden
    )
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * actual_golden + c1) * (2 * sigma_cross + c2)) / (
        (actual_sq + golden_sq + c1) * (sigma_actual + sigma_golden + c2)
    )
    return float(score.mean().item())


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
