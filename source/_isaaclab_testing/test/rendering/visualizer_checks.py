# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image and pause-control assertions shared by the two native viewer processes."""

from unittest.mock import Mock

import numpy as np
import torch


def changed_pixels(first: np.ndarray, second: np.ndarray, threshold: float = 50.0) -> int:
    """Count RGB pixels whose color changes beyond renderer noise."""
    return int((np.abs(first.astype(np.float32) - second.astype(np.float32)).max(axis=-1) >= threshold).sum())


def press_button(viewer, label: str) -> None:
    """Exercise the real training-control callback with one ImGui button press."""
    viewer._render_training_controls(
        Mock(
            button=lambda text: text == label,
            slider_float=lambda _label, value, *_args: (False, value),
            is_item_hovered=lambda: False,
        )
    )


def check_motion_and_pause(runtime, visualizer) -> None:
    """Playing moves pixels; render pause retains pixels while physics continues; simulation pause retains both."""
    first = visualizer.render_rgb_array().copy()
    for _ in range(12):
        runtime.step()
    playing = visualizer.render_rgb_array().copy()
    assert changed_pixels(first, playing) > 100
    viewer = visualizer._viewer
    press_button(viewer, "Pause Rendering")
    assert visualizer.is_rendering_paused()
    state = runtime.scene["robot"].data.joint_pos.torch.clone()
    step = runtime.sim.get_physics_step_count()
    for _ in range(12):
        runtime.step()
    assert runtime.sim.get_physics_step_count() == step + 12
    assert not torch.allclose(state, runtime.scene["robot"].data.joint_pos.torch)
    assert changed_pixels(playing, visualizer.render_rgb_array()) <= 100
    press_button(viewer, "Resume Rendering")
    assert not visualizer.is_rendering_paused()
    for _ in range(3):
        visualizer.step(0.0)
    resumed = visualizer.render_rgb_array().copy()
    assert changed_pixels(playing, resumed) > 100
    press_button(viewer, "Pause Simulation")
    assert visualizer.is_training_paused()
    state = runtime.scene["robot"].data.joint_pos.torch.clone()
    step = runtime.sim.get_physics_step_count()
    for _ in range(3):
        visualizer.step(0.0)
    assert runtime.sim.get_physics_step_count() == step
    torch.testing.assert_close(state, runtime.scene["robot"].data.joint_pos.torch)
    assert changed_pixels(resumed, visualizer.render_rgb_array()) <= 100
    press_button(viewer, "Resume Simulation")
    assert not visualizer.is_training_paused()
    for _ in range(12):
        runtime.step()
    assert changed_pixels(resumed, visualizer.render_rgb_array()) > 100
