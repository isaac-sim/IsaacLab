# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect the kinematic teapot demo when adding robot-driven pouring."""

import importlib
import math
import runpy
import sys

import numpy as np


def test_teapot_fill_preserves_kinematic_defaults(source_checkout_root, monkeypatch):
    """The additional robot demo must preserve the existing solver and motion."""
    script = source_checkout_root / "examples/demos/teapot_fill.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--device", "cpu", "--visualizer", "none"])
    demo = runpy.run_path(str(script))
    cfg = demo["create_sim_cfg"]()
    assert cfg.dt == 1.0 / 800
    assert demo["args_cli"].max_steps == 6000
    assert cfg.physics.solver_cfg.air_drag == 1.0e-3
    assert not cfg.physics.solver_cfg.project_outside_colliders

    # The established cubic tilt is 65 * 0.15625 degrees one quarter into its two-second tilt.
    position, orientation, twist = demo["container_pose_at_time"](1.05)
    angle = math.radians(65.0 * 0.15625)
    np.testing.assert_allclose(position, (-0.105, 0.0, 1.058979469))
    np.testing.assert_allclose(orientation, (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)))
    np.testing.assert_allclose(twist, (0.0, 0.0, 0.0, 0.0, math.radians(65.0) * 0.5625, 0.0))

    # The original pot rises linearly and retains its pouring tilt after the lift.
    position, _, twist = demo["container_pose_at_time"](3.30)
    np.testing.assert_allclose(position, (-0.105, 0.0, 1.118979469))
    np.testing.assert_allclose(twist, (0.0, 0.0, 0.08, 0.0, 0.0, 0.0))
    position, orientation, twist = demo["container_pose_at_time"](7.5)
    np.testing.assert_allclose(position, (-0.105, 0.0, 1.298979469))
    np.testing.assert_allclose(orientation, (0.0, math.sin(math.radians(32.5)), 0.0, math.cos(math.radians(32.5))))
    np.testing.assert_array_equal(twist, np.zeros(6))


def test_teapot_collider_sync_uses_measured_com_motion(source_checkout_root, monkeypatch):
    """Fluid boundaries use the interval's beginning pose and measured COM twist."""
    import warp as wp

    monkeypatch.syspath_prepend(str(source_checkout_root / "examples/demos"))
    helper = importlib.import_module("rizon_sharpa_teapot")
    # A quarter-turn moves this offset COM from (2.2, -0.9, 0.8) to (2.2, -1.0, 0.9).
    start = (2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    end = (2.3, -1.2, 0.6, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    source_start = wp.array([wp.transform_identity(), start], dtype=wp.transform, device="cpu")
    source_end = wp.array([wp.transform_identity(), end], dtype=wp.transform, device="cpu")
    source_com = wp.array([(0.0, 0.0, 0.0), (0.2, 0.1, 0.3)], dtype=wp.vec3, device="cpu")
    fluid_pose = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device="cpu")
    fluid_velocity = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")

    wp.launch(
        helper._sync_teapot_collider,
        dim=1,
        inputs=[source_start, source_end, source_com, 1, 0.25, fluid_pose, fluid_velocity, 0],
        device="cpu",
    )

    np.testing.assert_allclose(fluid_pose.numpy()[0], start)
    np.testing.assert_allclose(fluid_velocity.numpy()[0], (0.0, -0.4, 0.4, 0.0, 0.0, 2.0 * math.pi), atol=2.0e-6)
    np.testing.assert_array_equal(fluid_pose.numpy()[1], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(fluid_velocity.numpy()[1], np.zeros(6))


def test_teapot_arm_interpolation_preserves_unmapped_hand_targets(source_checkout_root, monkeypatch):
    """Mapped arm targets follow their velocities while finger targets stay fixed."""
    import warp as wp

    monkeypatch.syspath_prepend(str(source_checkout_root / "examples/demos"))
    helper = importlib.import_module("rizon_sharpa_teapot")
    indices = wp.array([(3, 0), (0, 2)], dtype=wp.vec2i, device="cpu")
    position = wp.array([0.3, 0.2, 0.1, 0.4], dtype=wp.float32, device="cpu")
    velocity = wp.array([0.8, 0.4, -0.4], dtype=wp.float32, device="cpu")

    wp.launch(helper._interpolate_arm_control, dim=2, inputs=[indices, position, velocity, 0.125], device="cpu")

    np.testing.assert_allclose(position.numpy(), (0.25, 0.2, 0.1, 0.5), atol=1.0e-7)
    np.testing.assert_allclose(velocity.numpy(), (0.8, 0.4, -0.4), atol=1.0e-7)
