# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test: visualization markers must appear in Newton GL render_rgb_array() output.

**Original bug**: ``NewtonGLVisualizer.render_rgb_array()`` (used for ``--video`` recording)
only logged the physics body state between ``begin_frame`` and ``end_frame``.  The
interactive ``step()`` path also calls ``render_newton_visualization_markers()`` each frame,
which logs velocity arrows, goal poses, contact indicators, and all other debug geometry.
Recorded videos therefore showed the raw scene without any of the overlays visible in the
live viewer — most noticeably in reorientation tasks where the goal-cube marker is the only
cue for what the policy is tracking.

**Fix**: add the ``render_newton_visualization_markers()`` call inside ``render_rgb_array()``,
gated on ``cfg.enable_markers`` so callers that deliberately disable markers still get a
clean frame.

This test verifies the fix using the AnymalD flat environment, which has command-velocity
arrow markers enabled by default (``debug_vis=True``).  The arrows are logged as Newton
visualization markers on every physics step via the command manager debug-vis callback.

The assertion uses ``unittest.mock.patch`` to spy on ``render_newton_visualization_markers``
because Newton's viewer persists mesh state between frames: once arrows are logged during
warmup they remain visible in subsequent frames even if the logging call is skipped.
Spying on the call directly verifies that the fix invokes the function when
``enable_markers=True`` and skips it when ``enable_markers=False``.

Two MP4 videos are also saved to ``tests/comparison-images/`` on every run for visual
inspection using ``ffmpeg`` (independent of Isaac Sim's bundled imageio):

* ``newton_markers_video_capture-without_markers.mp4`` — ``enable_markers=False`` path.
* ``newton_markers_video_capture-with_markers.mp4``    — ``enable_markers=True`` path.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_integration_utils as _viz_utils  # noqa: E402

_viz_utils.set_visualizer_integration_simulation_app(simulation_app)

pytestmark = pytest.mark.isaacsim_ci

# Number of frames captured per video.
_VIDEO_FRAMES = 30

# FPS for saved MP4 videos.
_VIDEO_FPS = 10

_OUTPUT_DIR = os.path.join(os.getcwd(), "tests", "comparison-images")

_MARKER_MODULE = "isaaclab_visualizers.newton.newton_visualizer.render_newton_visualization_markers"


def _save_mp4(frames: list[np.ndarray], path: str, fps: int = _VIDEO_FPS) -> None:
    """Save RGB frames as an MP4 using ffmpeg via subprocess (avoids bundled imageio)."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmp, f"frame_{i:04d}.png"))
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmp, "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={w}:{h}",
            path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _cleanup():
    from isaaclab.utils.seed import configure_seed

    configure_seed(42, torch_deterministic=True)
    yield
    from isaaclab.sim import SimulationContext

    SimulationContext.clear_instance()


def test_newton_gl_render_rgb_array_includes_markers():
    """render_rgb_array() must call render_newton_visualization_markers when enable_markers=True.

    Newton's viewer persists mesh state between frames, so pixel-diff comparison is
    unreliable (once markers are logged during warmup they remain visible even without
    the fix).  Instead this test spies on the function call directly.

    Also saves two MP4 videos to ``tests/comparison-images/`` for visual inspection.
    """
    import torch

    import isaaclab.sim as sim_utils

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()

        env = _viz_utils._make_anymal_d_env("newton", "physx")
        _viz_utils._configure_sim_for_visualizer_test(env)

        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
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

        # ---- Spy: verify the call contract -----------------------------------
        with patch(_MARKER_MODULE) as mock_markers:
            newton_viz.cfg.enable_markers = True
            newton_viz.render_rgb_array()
            assert mock_markers.call_count == 1, (
                f"render_newton_visualization_markers was called {mock_markers.call_count} times "
                "with enable_markers=True; expected exactly 1 call. "
                "The fix that adds markers to the headless capture path may be missing."
            )

        with patch(_MARKER_MODULE) as mock_markers:
            newton_viz.cfg.enable_markers = False
            newton_viz.render_rgb_array()
            assert mock_markers.call_count == 0, (
                f"render_newton_visualization_markers was called {mock_markers.call_count} times "
                "with enable_markers=False; expected 0 calls."
            )

        newton_viz.cfg.enable_markers = True  # restore default

        # ---- Visual output: save MP4s for inspection -------------------------
        frames_with: list[np.ndarray] = []
        frames_without: list[np.ndarray] = []

        for _ in range(_VIDEO_FRAMES):
            newton_viz.cfg.enable_markers = False
            frames_without.append(newton_viz.render_rgb_array()[..., :3])
            newton_viz.cfg.enable_markers = True
            frames_with.append(newton_viz.render_rgb_array()[..., :3])

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        path_without = os.path.join(_OUTPUT_DIR, "newton_markers_video_capture-without_markers.mp4")
        path_with = os.path.join(_OUTPUT_DIR, "newton_markers_video_capture-with_markers.mp4")
        _save_mp4(frames_without, path_without)
        _save_mp4(frames_with, path_with)

        # Save frame-0 PNGs alongside each video for quick comparison.
        Image.fromarray(frames_without[0]).save(path_without.replace(".mp4", "-frame0.png"))
        Image.fromarray(frames_with[0]).save(path_with.replace(".mp4", "-frame0.png"))

    finally:
        _viz_utils._cleanup_visualizer_test_process(env)
