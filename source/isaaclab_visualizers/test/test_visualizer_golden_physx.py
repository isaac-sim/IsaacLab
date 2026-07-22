# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer golden image tests on the PhysX physics backend.

Covers KitVisualizer (RTX viewport) and NewtonVisualizer (OpenGL) in both
viewport and tiled-camera capture modes for the cartpole, shadow hand, and
AnymalD environments.  Golden images are stored under
``golden_images/{scene}/{physics_backend}-{visualizer_type}-{mode}.png``.

On the first run for a new combination the current frame is saved as the
golden; the test fails so the file can be reviewed before committing.
"""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_golden_utils as _golden  # noqa: E402
import visualizer_integration_utils as _viz_utils  # noqa: E402

_viz_utils.set_visualizer_integration_simulation_app(simulation_app)

pytestmark = [pytest.mark.isaacsim_ci, _golden.FLAKY_MARK, pytest.mark.timeout(300)]

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = _golden.make_determinism_fixture()
_attach_comparison_properties_fixture = _golden.make_attach_comparison_properties_fixture(_COMPARISON_SCORES)

_PHYSICS_BACKEND = "physx"


def _run_anymal_d_physx(
    physics_backend: str,
    visualizer_type: str,
    mode: str,
    comparison_scores: list[dict],
) -> None:
    # PhysX 4-env AnymalD contact dynamics are not bit-reproducible under Kit RTX
    # tiled rendering.  Capture the reset pose (0 steps) for a stable golden.
    # At 0 buffer steps the 4 sub-viewports each need extra TAA samples; 40 extra
    # render-only passes compensate (same approach used for shadow_hand kit-tiled).
    kit_tiled = visualizer_type == "kit" and mode == "tiled"
    steps = 0 if kit_tiled else None
    extra = 40 if kit_tiled else 0
    _golden.run_visualizer_golden_anymal_d(
        physics_backend, visualizer_type, mode, comparison_scores, buffer_steps=steps, extra_render_passes=extra
    )


_SCENE_RUNNERS = {
    "cartpole": _golden.run_visualizer_golden_cartpole,
    "shadow_hand": _golden.run_visualizer_golden_shadow_hand,
    "anymal_d": _run_anymal_d_physx,
}

SCENE_VISUALIZER_MODE_COMBINATIONS = [
    pytest.param("cartpole", "kit", "tiled", id="physx-cartpole-kit-tiled", marks=_golden.FLAKY_MARK),
    pytest.param("cartpole", "newton", "tiled", id="physx-cartpole-newton-tiled", marks=_golden.FLAKY_MARK),
    pytest.param("cartpole", "kit", "viewport", id="physx-cartpole-kit-viewport", marks=_golden.FLAKY_MARK),
    pytest.param("cartpole", "newton", "viewport", id="physx-cartpole-newton-viewport", marks=_golden.FLAKY_MARK),
    # anymal_d runs before shadow_hand: shadow_hand leaves PhysX GPU tensor state that
    # collapses the anymal_d robot on the next env init.  Running anymal_d first avoids
    # this in-process contamination without needing a hard-to-replicate cleanup step.
    pytest.param("anymal_d", "kit", "tiled", id="physx-anymal_d-kit-tiled", marks=_golden.FLAKY_MARK),
    pytest.param("anymal_d", "newton", "tiled", id="physx-anymal_d-newton-tiled", marks=_golden.FLAKY_MARK),
    pytest.param("anymal_d", "kit", "viewport", id="physx-anymal_d-kit-viewport", marks=_golden.FLAKY_MARK),
    pytest.param("anymal_d", "newton", "viewport", id="physx-anymal_d-newton-viewport", marks=_golden.FLAKY_MARK),
    pytest.param("shadow_hand", "kit", "tiled", id="physx-shadow_hand-kit-tiled", marks=_golden.FLAKY_MARK),
    pytest.param("shadow_hand", "newton", "tiled", id="physx-shadow_hand-newton-tiled", marks=_golden.FLAKY_MARK),
    pytest.param("shadow_hand", "kit", "viewport", id="physx-shadow_hand-kit-viewport", marks=_golden.FLAKY_MARK),
    pytest.param("shadow_hand", "newton", "viewport", id="physx-shadow_hand-newton-viewport", marks=_golden.FLAKY_MARK),
]


@pytest.mark.parametrize("scene,visualizer_type,mode", SCENE_VISUALIZER_MODE_COMBINATIONS)
def test_visualizer_golden(scene: str, visualizer_type: str, mode: str) -> None:
    """Golden image correctness for PhysX across scenes, visualizer types, and modes."""
    _SCENE_RUNNERS[scene](_PHYSICS_BACKEND, visualizer_type, mode, _COMPARISON_SCORES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
