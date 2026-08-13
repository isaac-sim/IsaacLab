# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test: visualization markers must appear in Newton GL render_rgb_array() output.

NewtonGLVisualizer.render_rgb_array() is used for headless video recording (``--video``).
The interactive step() path logs visualization markers (goal poses, command arrows, etc.)
on every frame, but the capture path previously skipped them, producing videos where
debug geometry visible in the live viewer was absent from recordings.

This test verifies the fix by:
1. Creating an AnymalD env with command-velocity arrows enabled (``debug_vis=True``).
2. Running enough physics steps for markers to be logged.
3. Capturing two video sequences via ``render_rgb_array()``:
   - Without markers: ``enable_markers=False`` — simulates the broken capture path.
   - With markers: ``enable_markers=True`` — the fixed capture path.
4. Saving both videos to ``tests/comparison-images/`` for visual inspection.
5. Asserting the frames differ by at least ``_MIN_DIFF_PCT`` of pixels — markers add
   visible colored geometry that exceeds this threshold when present.

When the marker logging call in ``render_rgb_array()`` is absent, both frames are
identical (no markers on either path) and the test fails.
"""

import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_integration_utils as _viz_utils  # noqa: E402

_viz_utils.set_visualizer_integration_simulation_app(simulation_app)

pytestmark = pytest.mark.isaacsim_ci

# Fraction of pixels that must differ between the with-markers and without-markers
# frames for the test to confirm markers are visible.  Command-velocity arrows on
# AnymalD are large colored overlays; even a loose 0.5% threshold is well exceeded
# when they are present.
_MIN_DIFF_PCT = 0.5

# Number of frames captured per video.
_VIDEO_FRAMES = 30

# FPS for saved videos.
_VIDEO_FPS = 10

_OUTPUT_DIR = os.path.join(os.getcwd(), "tests", "comparison-images")


def _pixel_diff_percentage(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Return percentage of pixels that differ by more than 10 in L2 norm."""
    diff = np.linalg.norm(frame_a.astype(float) - frame_b.astype(float), axis=2)
    return 100.0 * float(np.sum(diff > 10.0)) / diff.size


def _save_video(frames: list[np.ndarray], path: str, fps: int = _VIDEO_FPS) -> None:
    """Save a list of RGB frames as an MP4 video."""
    import imageio.v3 as iio

    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", pixelformat="yuv420p")


@pytest.fixture(autouse=True)
def _cleanup():
    from isaaclab.utils.seed import configure_seed

    configure_seed(42, torch_deterministic=True)
    yield
    from isaaclab.sim import SimulationContext

    SimulationContext.clear_instance()


def test_newton_gl_render_rgb_array_includes_markers():
    """render_rgb_array() with enable_markers=True must differ from enable_markers=False.

    Saves two videos to ``tests/comparison-images/`` for visual inspection:

    * ``newton_markers_video_capture-without_markers.mp4`` — broken path (markers absent).
    * ``newton_markers_video_capture-with_markers.mp4`` — fixed path (markers visible).
    """
    import torch

    import isaaclab.sim as sim_utils

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()

        # Build AnymalD flat env with Newton GL (headless) and markers enabled.
        # Eye/lookat are set to the same angle used in the integration golden tests
        # so the command-velocity arrows are in the camera FOV.
        env = _viz_utils._make_anymal_d_env("newton", "physx")
        _viz_utils._configure_sim_for_visualizer_test(env)

        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)

        # Warm up physics and let command markers (velocity arrows) be logged.
        for _ in range(_viz_utils._START_BUFFER_STEPS):
            env.step(action=actions)

        newton_viz = next(
            v
            for v in env.sim.visualizers
            if hasattr(v, "render_rgb_array")
            and hasattr(v.cfg, "enable_markers")
            and v.cfg.visualizer_type in ("newton_gl", "newton")
        )
        _viz_utils._warm_newton_viewer(newton_viz)

        frames_with: list[np.ndarray] = []
        frames_without: list[np.ndarray] = []

        for _ in range(_VIDEO_FRAMES):
            env.step(action=actions)

            # Capture without markers (simulates the broken capture path).
            newton_viz.cfg.enable_markers = False
            frames_without.append(newton_viz.render_rgb_array()[..., :3])

            # Capture with markers (the fixed capture path).
            newton_viz.cfg.enable_markers = True
            frames_with.append(newton_viz.render_rgb_array()[..., :3])

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        path_without = os.path.join(_OUTPUT_DIR, "newton_markers_video_capture-without_markers.mp4")
        path_with = os.path.join(_OUTPUT_DIR, "newton_markers_video_capture-with_markers.mp4")
        _save_video(frames_without, path_without)
        _save_video(frames_with, path_with)

        # Also save the first frame of each as a PNG for quick inspection.
        Image.fromarray(frames_without[0]).save(path_without.replace(".mp4", "-frame0.png"))
        Image.fromarray(frames_with[0]).save(path_with.replace(".mp4", "-frame0.png"))

        # Assert that markers are visibly present in the fixed frames.
        diff_pct = _pixel_diff_percentage(frames_with[-1], frames_without[-1])
        assert diff_pct >= _MIN_DIFF_PCT, (
            f"render_rgb_array() with enable_markers=True produced a frame that is only "
            f"{diff_pct:.3f}% different from enable_markers=False (threshold: {_MIN_DIFF_PCT}%). "
            "Visualization markers (command-velocity arrows) are not being rendered on the "
            f"headless capture path.\nVideos saved to {_OUTPUT_DIR}/ for inspection."
        )
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)
