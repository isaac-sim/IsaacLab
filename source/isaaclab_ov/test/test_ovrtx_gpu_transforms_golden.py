# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden regression for multi-camera corruption when OVRTX reads GPU transforms.

Run this file in a separate process: OVRTX owns process-global renderer state.
The test exercises the native renderer without Isaac Sim or external scene assets.
See ``golden_images/gpu_transforms/README.md`` for reference provenance.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ovrtx = pytest.importorskip("ovrtx", reason="requires the ovrtx extra")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.kitless,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an RTX-capable CUDA device"),
]

_GOLDEN_DIR = Path(__file__).parent / "golden_images" / "gpu_transforms"
_NUM_FRAMES = 100
_NUM_COMPARISON_FRAMES = 10
# Match the pixel tolerance used by the task-level OVRTX golden tests.
_PIXEL_L2_THRESHOLD = 10.0
_MAX_DIFFERENT_PIXELS_PERCENTAGE = 3.0


def _capture_frames(output_dir: Path, *, read_gpu_transforms: bool) -> tuple[np.ndarray, str]:
    """Render moving objects, then capture both cameras after their poses settle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scene.usda"
    scene_path.write_text((_GOLDEN_DIR / "scene.usda").read_text())
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    checker = (((x // 16 + y // 16) % 2) * 190 + 40).astype(np.uint8)
    Image.fromarray(np.repeat(checker[..., None], 3, axis=-1)).save(output_dir / "checkerboard.png")

    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            read_gpu_transforms=read_gpu_transforms,
            texture_streaming_mode=ovrtx.TextureStreamingMode.SYNCHRONOUS,
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            log_level="warn",
            log_file_path=str(output_dir / "renderer.log"),
        )
    )
    try:
        native_version = ".".join(str(part) for part in renderer.version)
        renderer.open_usd(str(scene_path))
        product_paths = [f"/SmokeRender/Camera{i}" for i in range(2)]
        for _ in range(40):
            renderer.step(set(product_paths), 1 / 60)

        transforms = torch.eye(4, dtype=torch.float64, device="cuda:0").repeat(2, 1, 1)
        frames = []
        for frame_index in range(_NUM_FRAMES):
            # Hold the final pose for 40 frames so denoising can converge before comparison.
            phase = min(frame_index, 59) / 8.0
            transforms[0, 3, 0] = float(70 * np.sin(phase))
            transforms[0, 3, 1] = 50
            transforms[1, 3, 0] = float(100 - 35 * np.sin(phase))
            transforms[1, 3, 1] = 25
            transforms[1, 3, 2] = -50
            torch.cuda.synchronize()
            renderer.write_attribute_async(
                ["/World/Sphere", "/World/Cube"],
                "omni:xform",
                transforms,
                data_access=ovrtx.DataAccess.ASYNC,
            ).wait()
            products = renderer.step(set(product_paths), 1 / 60)
            camera_frames = [_read_rgb(products[path].frames[-1]) for path in product_paths]
            del products
            if frame_index >= _NUM_FRAMES - _NUM_COMPARISON_FRAMES:
                frames.append(camera_frames)
        return np.asarray(frames), native_version
    finally:
        renderer.destroy()


def _read_rgb(frame: ovrtx.FrameOutput) -> np.ndarray:
    """Copy a frame before the renderer reuses its output buffers."""
    # OVRTX 0.4 keys outputs by source name; 0.5 keys them by RenderVar prim path.
    color_key = "LdrColor" if "LdrColor" in frame.render_vars else "/SmokeRender/Color"
    mapped = frame.render_vars[color_key].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(mapped)[..., :3].copy()
    finally:
        mapped.unmap()


def test_gpu_transform_frames_match_golden(tmp_path: Path, record_property):
    """GPU transform reads must preserve each camera's settled appearance."""
    goldens = []
    for camera_index in range(2):
        with Image.open(_GOLDEN_DIR / f"camera{camera_index}.png") as image:
            goldens.append(np.asarray(image.convert("RGB")))
    frames, native_version = _capture_frames(tmp_path, read_gpu_transforms=True)
    record_property("ovrtx_native_version", native_version)
    assert frames.shape == (_NUM_COMPARISON_FRAMES, 2, 360, 640, 3)

    failures = []
    for camera_index, golden in enumerate(goldens):
        actual = frames[:, camera_index]
        assert actual.shape[1:] == golden.shape
        difference = actual.astype(np.float32) - golden.astype(np.float32)
        different_pixels = np.linalg.norm(difference, axis=-1) > _PIXEL_L2_THRESHOLD
        percentages = 100 * different_pixels.mean(axis=(1, 2))
        worst_frame = int(np.argmax(percentages))
        worst_percentage = float(percentages[worst_frame])
        record_property(f"camera{camera_index}_different_pixels_percent", worst_percentage)
        if worst_percentage > _MAX_DIFFERENT_PIXELS_PERCENTAGE:
            frame_index = _NUM_FRAMES - _NUM_COMPARISON_FRAMES + worst_frame
            prefix = tmp_path / f"camera{camera_index}_frame{frame_index}"
            Image.fromarray(actual[worst_frame]).save(f"{prefix}_actual.png")
            Image.fromarray(golden).save(f"{prefix}_golden.png")
            Image.fromarray((different_pixels[worst_frame] * 255).astype(np.uint8)).save(f"{prefix}_diff.png")
            failures.append(f"camera {camera_index}, frame {frame_index}: {worst_percentage:.2f}% differing pixels")

    assert not failures, (
        f"OVRTX {native_version}: GPU-transform rendering differs from the golden by more than"
        f" {_MAX_DIFFERENT_PIXELS_PERCENTAGE}% of pixels (RGB L2 > {_PIXEL_L2_THRESHOLD})."
        f" {'; '.join(failures)}. Images and renderer log: {tmp_path}"
    )
