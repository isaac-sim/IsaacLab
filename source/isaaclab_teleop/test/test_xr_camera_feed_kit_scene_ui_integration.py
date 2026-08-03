# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the XR Kit experience before importing Scene UI."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, xr=True, device="cpu").app

"""Rest everything follows."""

import pytest
from isaaclab_teleop.camera_feed import _PanelDescriptor
from isaaclab_teleop.camera_feed_kit_scene_ui import _KitSceneUiCameraFeedPresenter

import isaaclab.sim as sim_utils

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


def test_real_scene_ui_imports_and_constructs_world_panel():
    """The XR experience provides the real Scene UI signatures used by PiP."""
    sim_utils.create_new_stage()
    presenter = _KitSceneUiCameraFeedPresenter()
    descriptor = _PanelDescriptor(
        label="Camera",
        width_m=0.48,
        offset_m=(0.0, 0.0),
        distance_m=0.8,
        placement="world",
        world_position_m=(0.0, 0.8, 1.6),
        world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    panel = presenter.create_panel(descriptor, width=720, height=450)
    try:
        assert panel._container is not None
        assert panel._component is not None
    finally:
        panel.close()

    assert panel._closed
