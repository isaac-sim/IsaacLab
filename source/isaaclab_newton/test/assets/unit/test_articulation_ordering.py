# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused public-order state publication tests for Newton articulations."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.assets.articulation.base_articulation import BaseArticulation

pytestmark = pytest.mark.unit


class _LaunchCache:
    def launch(self, _name, kernel, *, dim, inputs, outputs) -> None:
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device="cpu")


def test_post_step_publish_refreshes_joint_and_body_shadows_in_public_order() -> None:
    """The Newton post-step hook must publish every Tier-1 state shadow in public order."""
    data = object.__new__(ArticulationData)
    data._device = "cpu"
    data._num_instances = 2
    data._num_joints = 2
    data._num_bodies = 2
    user_to_backend = wp.array([1, 0], dtype=wp.int32, device="cpu")
    data.joint_ordering = SimpleNamespace(user_to_backend=user_to_backend)
    data.body_ordering = SimpleNamespace(user_to_backend=user_to_backend)
    data._read_launch_cache = _LaunchCache()
    data._sim_bind_joint_pos = wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.float32, device="cpu")
    data._sim_bind_joint_vel = wp.array([[5.0, 6.0], [7.0, 8.0]], dtype=wp.float32, device="cpu")
    data._joint_pos_user = wp.zeros((2, 2), dtype=wp.float32, device="cpu")
    data._joint_vel_user = wp.zeros((2, 2), dtype=wp.float32, device="cpu")
    backend_pose = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    data._sim_bind_body_link_pose_w = wp.array(backend_pose, dtype=wp.transformf, device="cpu")
    data._sim_bind_body_com_vel_w = wp.array(
        np.arange(24, dtype=np.float32).reshape(2, 2, 6), dtype=wp.spatial_vectorf, device="cpu"
    )
    data._body_link_pose_w_user = wp.zeros((2, 2), dtype=wp.transformf, device="cpu")
    data._body_com_vel_w_user = wp.zeros((2, 2), dtype=wp.spatial_vectorf, device="cpu")

    data._refresh_user_order_state()

    np.testing.assert_allclose(data._joint_pos_user.numpy(), [[2.0, 1.0], [4.0, 3.0]])
    np.testing.assert_allclose(data._joint_vel_user.numpy(), [[6.0, 5.0], [8.0, 7.0]])
    np.testing.assert_allclose(data._body_link_pose_w_user.numpy(), backend_pose[:, [1, 0]])
    np.testing.assert_allclose(
        data._body_com_vel_w_user.numpy(), np.arange(24, dtype=np.float32).reshape(2, 2, 6)[:, [1, 0]]
    )


def test_clear_callbacks_unregisters_only_the_articulation_post_step_hook(monkeypatch) -> None:
    """Clearing one articulation must remove its ordered publish hook without leaking it globally."""
    articulation = object.__new__(Articulation)

    def callback() -> None:
        pass

    articulation._model_init_handle = None
    articulation._physics_ready_handle = None
    articulation._post_step_callback = callback
    unregistered = []
    monkeypatch.setattr(BaseArticulation, "_clear_callbacks", lambda self: None)
    monkeypatch.setattr(
        SimulationManager,
        "unregister_post_step_callback",
        classmethod(lambda cls, value: unregistered.append(value)),
    )

    articulation._clear_callbacks()

    assert unregistered == [callback]
    assert articulation._post_step_callback is None
