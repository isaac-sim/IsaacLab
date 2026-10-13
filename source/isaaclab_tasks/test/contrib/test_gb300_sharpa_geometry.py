# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured one-way cable endpoint state across CUDA graph replay and reset."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import VBDSolverCfg
from newton.solvers.experimental.coupled import ModelView

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.gb300_sharpa_insertion.cable import _pin_seated_tail
from isaaclab_tasks.contrib.gb300_sharpa_insertion.endpoint import NewtonEndpointVBDManager


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_cable_endpoint_follows_measured_plug_twist_through_captured_step_and_reset(device):
    """Prescribed plug motion survives VBD gravity and drives the cable boundary after reset."""
    quarter_turn = np.sqrt(0.5)
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        plug = builder.add_body(
            com=(0.01, 0, 0),
            mass=0.02,
            inertia=wp.mat33(np.eye(3) * 1e-5),
            label="/World/envs/env_0/Cable/head",
        )
        endpoint = builder.add_body(
            xform=wp.transform((0.04, 0, 0), (quarter_turn, 0, 0, quarter_turn)),
            com=(0, 0.02, 0),
            is_kinematic=True,
            label="/World/envs/env_0/Cable/geometry/plug_endpoint",
        )
        tail = builder.add_body(
            xform=wp.transform((0.0, 0.25, 0.5), wp.quat_identity()),
            label="/World/envs/env_0/Cable/tail",
        )
        tail_shape = builder.add_shape_capsule(tail, radius=0.00325, half_height=0.02)
        _pin_seated_tail(builder, tail)
        builder.color()
        model = builder.finalize()
        original_flags = model.body_flags.numpy()
        proxy_flags = original_flags.copy()
        proxy_flags[plug] |= int(newton.BodyFlags.PROXY)
        view = ModelView(model, "cable")
        view.body_flags = wp.array(proxy_flags, dtype=model.body_flags.dtype)
        solver = NewtonEndpointVBDManager._create_solver(view, VBDSolverCfg(iterations=2, rigid_compliant_alm=True))
        state_in, state_out = model.state(), model.state()
        control = model.control()
        dt = 1 / 240

        def advance():
            solver.coupling_notify_input_state_update(
                state_in, newton.StateFlags.BODY_Q | newton.StateFlags.BODY_QD, dt=dt
            )
            solver.step(state_in, state_out, control, None, dt)

        advance()
        with wp.ScopedCapture(device=device) as capture:
            advance()

        assert not model.shape_flags.numpy()[tail_shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        poses, velocities = model.body_q.numpy(), model.body_qd.numpy()
        cases = (
            (
                (1, 2, 3, 0, 0, quarter_turn, quarter_turn),
                (1, 2, 3, 0, 0, 4),
                (1, 2.04, 3, 0.5, 0.5, 0.5, 0.5),
                (0.88, 2, 3, 0, 0, 4),
            ),
            ((-2, 0.5, 1, 0, 0, 0, 1), (0,) * 6, (-1.96, 0.5, 1, quarter_turn, 0, 0, quarter_turn), (0,) * 6),
        )
        # The first COM offset is (0, 0.03, 0.02); omega cross offset is (-0.12, 0, 0).
        for index, case in enumerate(cases):
            plug_pose, plug_twist, endpoint_pose, endpoint_twist = case
            poses[plug], velocities[plug] = plug_pose, plug_twist
            state_in.body_q.assign(poses)
            state_in.body_qd.assign(velocities)
            if index:
                solver.reset(state_in, flags=0)
            wp.capture_launch(capture.graph)
            for state in (state_in, state_out):
                np.testing.assert_allclose(
                    state.body_q.numpy()[[plug, endpoint, tail]],
                    (plug_pose, endpoint_pose, (0, 0.25, 0.5, 0, 0, 0, 1)),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    state.body_qd.numpy()[[plug, endpoint, tail]], (plug_twist, endpoint_twist, (0,) * 6), atol=1e-6
                )
        np.testing.assert_array_equal(model.body_flags.numpy(), original_flags)
