# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton video recording."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab_newton.video_recording.newton_gl_perspective_video import NewtonGlPerspectiveVideo


def test_standalone_viewer_shows_only_configured_environment() -> None:
    """The standalone Newton recorder restricts rendering to its configured world."""
    viewer = MagicMock()
    viewer_type = MagicMock(return_value=viewer)
    cfg = SimpleNamespace(
        window_width=320,
        window_height=180,
        horiz_fov_deg=60.0,
        eye=(2.0, 1.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        env_index=7,
    )
    physics_module = SimpleNamespace(NewtonManager=SimpleNamespace(get_model=MagicMock(return_value=object())))
    pyglet_module = SimpleNamespace(options={})
    viewer_module = SimpleNamespace(ViewerGL=viewer_type)
    warp_module = SimpleNamespace(vec3=lambda x, y, z: (x, y, z))

    with patch.dict(
        sys.modules,
        {
            "isaaclab_newton.physics": physics_module,
            "pyglet": pyglet_module,
            "newton.viewer": viewer_module,
            "warp": warp_module,
        },
    ):
        recorder = NewtonGlPerspectiveVideo(cfg)
        recorder._ensure_viewer()

    viewer.set_model.assert_called_once()
    viewer.set_visible_worlds.assert_called_once_with([7])
    viewer.set_world_offsets.assert_called_once_with((0.0, 0.0, 0.0))
