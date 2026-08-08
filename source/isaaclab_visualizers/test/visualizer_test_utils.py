# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-free visualizer rendering and lifecycle checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from PIL import Image

from isaaclab.test.integration_scene_cfgs import RenderingTestSceneCfg
from isaaclab.test.utils.golden_image import camera_output_image, compare_to_golden, frame_image
from isaaclab.test.utils.rendering import CAMERA_EYE, CAMERA_TARGET, RenderingScene, build_rendering_scene

_GOLDEN_DIR = Path(__file__).parent / "golden_images" / "rendering_scene"
_ARTIFACT_DIR = Path.cwd() / "tests" / "comparison-images" / "images"
_WINDOW_SIZE = (320, 240)


def make_visualizer_cfg(kind: str, *, tiled: bool):
    """Create one deterministic viewport or generated-camera configuration."""
    common = dict(
        eye=CAMERA_EYE,
        lookat=CAMERA_TARGET,
        randomly_sample_visible_envs=False,
        enable_live_plots=False,
        enable_markers=False,
    )
    streaming = dict(
        streaming_view=tiled,
        streaming_envs=[0, 1, 2, 3] if tiled else 1,
        streaming_cam_eye=CAMERA_EYE,
        streaming_cam_target_prim_path="/World/envs/*/Robot",
    )
    width, height = _WINDOW_SIZE
    if kind == "kit":
        return KitVisualizerCfg(
            headless=True,
            window_width=width,
            window_height=height,
            **streaming,
            **common,
        )
    if kind == "newton":
        return NewtonGLVisualizerCfg(
            headless=True,
            window_width=width,
            window_height=height,
            **streaming,
            **common,
        )
    raise ValueError(f"Unknown visualizer: {kind!r}")


def run_visualizer_case(physics: str, kind: str, tiled: bool, request: Any) -> None:
    """Validate a reset-pose golden, then confirm scene state and pixels move."""
    mode = "tiled" if tiled else "viewport"
    with build_rendering_scene(
        RenderingTestSceneCfg(num_envs=4 if tiled else 1, env_spacing=5.0, lazy_sensor_update=True),
        physics,
        visualizer_cfgs=make_visualizer_cfg(kind, tiled=tiled),
    ) as runtime:
        assert runtime.sim.get_physics_step_count() == 0
        visualizer = _active_visualizer(runtime, kind)
        reset_image = _capture(runtime, visualizer, tiled=tiled)
        _assert_useful_image(reset_image, f"{kind} {mode}")

        label = f"{physics}-{kind}-{mode}"
        comparison = compare_to_golden(
            reset_image,
            _GOLDEN_DIR / f"{label}.png",
            label=f"visualizer-{label}",
            artifact_dir=_ARTIFACT_DIR,
            max_diff_pct=8.0 if kind == "kit" and tiled else 5.0 if kind == "kit" else 0.75,
            min_ssim=0.97 if kind == "kit" else 0.99,
        )
        comparison.record(request)

        cube = runtime.scene["moving_cube"]
        start_position = cube.data.root_pos_w.torch.clone()
        for _ in range(4):
            runtime.step()
        assert runtime.sim.get_physics_step_count() == 4
        assert torch.max(torch.abs(cube.data.root_pos_w.torch - start_position)).item() > 1.0e-3

        moved_image = _capture(runtime, visualizer, tiled=tiled)
        _assert_useful_image(moved_image, f"moving {kind} {mode}")
        changed = np.linalg.norm(
            np.asarray(reset_image, dtype=np.float32) - np.asarray(moved_image, dtype=np.float32), axis=-1
        )
        assert np.count_nonzero(changed > 10.0) > 20, f"{kind} {mode} did not render the moving scene."
        comparison.assert_passed()


def _active_visualizer(runtime: RenderingScene, kind: str):
    expected = "newton_gl" if kind == "newton" else kind
    matches = [visualizer for visualizer in runtime.sim.visualizers if visualizer.cfg.visualizer_type == expected]
    assert len(matches) == 1, f"Expected one {expected} visualizer, got {len(matches)}."
    return matches[0]


def _capture(runtime: RenderingScene, visualizer: Any, *, tiled: bool) -> Image.Image:
    if not tiled:
        frame = None
        for _ in range(6):
            runtime.sim.render()
            frame = visualizer.render_rgb_array()
        assert frame is not None
        return frame_image(frame)

    camera = visualizer._camera_sensor
    assert camera is not None, "Visualizer did not create its streaming camera."
    for _ in range(6):
        if getattr(visualizer, "_camera_is_owned", False):
            visualizer._update_owned_camera_poses()
            sync = getattr(visualizer, "_sync_camera_pose_updates_to_kit", None)
            if callable(sync):
                sync()
        runtime.sim.render()
        camera.update(0.0, force_recompute=True)
    output = camera.data.output["rgb"]
    tensor = output if torch.is_tensor(output) else output.torch
    indices = [int(index) for index in (visualizer._camera_sensor_indices or range(tensor.shape[0]))]
    return camera_output_image(tensor[indices], "rgb")


def _assert_useful_image(image: Image.Image, label: str) -> None:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    assert pixels.std() > 3.0 and np.ptp(pixels) > 20.0, f"{label} produced a flat frame."
