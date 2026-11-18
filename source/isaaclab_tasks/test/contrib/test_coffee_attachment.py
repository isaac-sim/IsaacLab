# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native mounted-spoon updates in coupled Newton worlds."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
from newton.selection import ArticulationView

from isaaclab.sim import SimulationContext

from isaaclab_tasks.contrib.rizon_sharpa_coffee.attachment import MountedSpoonAttachment

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_mounted_spoon_updates_both_states_in_each_world(monkeypatch):
    """A moving palm drives its own free spoon even with different body and joint offsets."""
    builder = newton.ModelBuilder()
    palms, spoons, joints = [], [], []
    for world in range(2):
        builder.begin_world()
        if world == 1:
            builder.add_body(label="unrelated")
        hand = builder.add_link(label=f"hand_{world}")
        palm = builder.add_link(label=f"palm_{world}")
        root = builder.add_joint_fixed(-1, hand)
        wrist = builder.add_joint_revolute(hand, palm)
        builder.add_articulation([root, wrist], label=f"robot_{world}")
        spoon = builder.add_link(label=f"tool_{world}")
        joint = builder.add_joint_free(spoon)
        builder.add_articulation([joint], label=f"spoon_{world}")
        palms.append(palm)
        spoons.append(spoon)
        joints.append(joint)
        builder.end_world()

    model = builder.finalize(device="cpu")
    states = [model.state(), model.state()]
    poses = states[0].body_q.numpy()
    poses[palms, 0] = [2, 4]
    states[0].body_q.assign(poses)
    velocities = states[0].body_qd.numpy()
    velocities[palms, 5] = [1, 2]
    states[0].body_qd.assign(velocities)
    callbacks = []
    backend = SimpleNamespace(model=model, state_0=states[0], state_1=states[1])
    manager = SimpleNamespace(backend=backend, register_post_actuator_callback=callbacks.append)
    monkeypatch.setattr(SimulationContext, "instance", lambda: SimpleNamespace(backend_manager=manager))
    robot = SimpleNamespace(
        root_view=ArticulationView(model, "robot_*", homogeneous=False), find_bodies=lambda _: ([1], [])
    )
    spoon = SimpleNamespace(root_view=ArticulationView(model, "spoon_*", homogeneous=False))
    MountedSpoonAttachment(robot, spoon, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    (sync,) = callbacks
    sync()
    for state in states:
        np.testing.assert_allclose(state.body_q.numpy()[spoons, 0], [2.5, 4.5])
        np.testing.assert_allclose(state.body_qd.numpy()[spoons, 1], [0.5, 1.0])
        np.testing.assert_allclose(state.joint_q.numpy()[model.joint_q_start.numpy()[joints]], [2.5, 4.5])
        np.testing.assert_allclose(state.joint_qd.numpy()[model.joint_qd_start.numpy()[joints] + 1], [0.5, 1.0])
