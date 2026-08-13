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
3. Capturing two frames via ``render_rgb_array()``:
   - ``enable_markers=True`` (default, should show command arrows)
   - ``enable_markers=False`` (no markers, clean frame)
4. Asserting the two frames are meaningfully different — markers add visible colored
   geometry that shifts enough pixels to exceed the difference threshold.

When the marker logging call in ``render_rgb_array()`` is absent, both frames are
identical (no markers on either path) and the test fails.
"""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import pytest  # noqa: E402

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


def _pixel_diff_percentage(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Return percentage of pixels that differ by more than 10 in L2 norm."""
    diff = np.linalg.norm(frame_a.astype(float) - frame_b.astype(float), axis=2)
    return 100.0 * float(np.sum(diff > 10.0)) / diff.size


@pytest.fixture(autouse=True)
def _cleanup():
    from isaaclab.utils.seed import configure_seed

    configure_seed(42, torch_deterministic=True)
    yield
    from isaaclab.sim import SimulationContext

    SimulationContext.clear_instance()


def test_newton_gl_render_rgb_array_includes_markers():
    """render_rgb_array() with enable_markers=True must differ from enable_markers=False."""
    import copy

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        sim_utils.create_new_stage()

        # Build AnymalD flat env with Newton GL (headless) and markers enabled.
        env = _viz_utils._make_anymal_d_env("newton", "physx")
        _viz_utils._configure_sim_for_visualizer_test(env)

        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)
        env.reset()

        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        # Step enough times for command markers (velocity arrows) to be logged
        # and visible in the rendered frame.
        for _ in range(_viz_utils._START_BUFFER_STEPS):
            env.step(action=actions)

        newton_viz = next(
            v for v in env.sim.visualizers
            if hasattr(v, "render_rgb_array") and hasattr(v.cfg, "enable_markers")
            and v.cfg.visualizer_type in ("newton_gl", "newton")
        )
        _viz_utils._warm_newton_viewer(newton_viz)

        # Capture with markers enabled (the default, should include command arrows).
        frame_with_markers = newton_viz.render_rgb_array()[..., :3]

        # Temporarily disable markers and capture again.
        newton_viz.cfg.enable_markers = False
        frame_without_markers = newton_viz.render_rgb_array()[..., :3]
        newton_viz.cfg.enable_markers = True

        diff_pct = _pixel_diff_percentage(frame_with_markers, frame_without_markers)

        assert diff_pct >= _MIN_DIFF_PCT, (
            f"render_rgb_array() with enable_markers=True produced a frame that is only "
            f"{diff_pct:.3f}% different from enable_markers=False (threshold: {_MIN_DIFF_PCT}%). "
            "Visualization markers (command-velocity arrows) are not being rendered on the "
            "headless capture path."
        )
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)
