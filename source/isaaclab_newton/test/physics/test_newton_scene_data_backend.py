# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native-array freshness after Newton swaps its authoritative state buffers."""

from types import SimpleNamespace

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

from isaaclab.scene_data import SceneDataFormat


def test_native_geometry_and_transforms_follow_current_state():
    state = SimpleNamespace(
        particle_q=wp.zeros(5, dtype=wp.vec3f, device="cpu"),
        body_q=wp.zeros(2, dtype=wp.transformf, device="cpu"),
    )
    model = SimpleNamespace(body_count=2, body_label=["/World/A", "/World/B"])
    manager = SimpleNamespace(
        get_model=lambda: model,
        get_state_0=lambda: state,
        forward=lambda: None,
        transforms_may_change_on_graph_replay=False,
    )
    backend = NewtonSceneDataBackend(manager)
    source = SceneDataFormat.Points()
    source.points = state.particle_q
    ranges = {"/World/mpm": (0, 3), "/World/deformable": (3, 2)}
    backend._geometry_batches = [(source, ranges)]

    assert backend.get_geometry_batches()[0] == (source, ranges)
    assert backend.transforms.transforms is state.body_q
    assert backend.transform_paths == ["/World/A", "/World/B"]
    transform_version = backend.transforms_timestamp
    geometry_version = backend.geometry_timestamp
    old_points, old_transforms = state.particle_q, state.body_q
    state.particle_q = wp.ones(5, dtype=wp.vec3f, device="cpu")
    state.body_q = wp.zeros(2, dtype=wp.transformf, device="cpu")

    assert backend.get_geometry_batches()[0][0].points is state.particle_q
    assert backend.transforms.transforms is state.body_q
    assert source.points is not old_points
    assert backend.transforms.transforms is not old_transforms
    assert backend.geometry_timestamp == geometry_version + 1
    assert backend.transforms_timestamp == transform_version + 1
