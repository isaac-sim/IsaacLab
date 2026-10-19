# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backends driven through the functional core without the Newton manager.

Covers several backends with different solvers stepped side by side, eagerly or from their own CUDA graphs, and
independent managers that build, step, and close their own models.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    NewtonBackend,
    NewtonCfg,
    NewtonManager,
)
from isaaclab_newton.physics import newton_backend as nb
from isaaclab_newton.physics.newton_solver import create_builder, create_solver_adapter

DEVICE = "cuda:0"
DT = 0.01


def _pendulums(physics_cfg: NewtonCfg, steps: int, *, captured=False) -> NewtonBackend:
    """A backend of one pendulum per world, stepping ``steps`` physics steps."""
    builder = _pendulum_builder(physics_cfg)
    create_solver_adapter(physics_cfg.solver_cfg).prepare_solver_builder(builder)
    model = builder.finalize(device=DEVICE)
    backend = NewtonBackend(model, physics_cfg, dt=DT)
    nb.init_solver(backend)
    backend.steps_per_call = steps
    if not captured:
        backend.capture = None
    return backend


def _pendulum_builder(physics_cfg):
    """Four pendulum worlds, released from a different angle in each world."""
    builder = create_builder(physics_cfg)
    for world in range(4):
        builder.begin_world()
        link = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
        joint = builder.add_joint_revolute(
            -1,
            link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        )
        builder.joint_q[-1] = 0.2 + 0.2 * world
        builder.add_articulation([joint])
        builder.end_world()
    return builder


def _configs() -> tuple[tuple[NewtonCfg, int], tuple[NewtonCfg, int]]:
    """Two solver configurations and the physics steps each backend advances per call."""
    return (
        (NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True), num_substeps=2), 3),
        (NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), num_substeps=4), 2),
    )


def _angles(backend: NewtonBackend) -> np.ndarray:
    return backend.state_0.joint_q.numpy().copy()


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_backends_with_different_solvers_step_side_by_side(captured):
    """Each backend advances only its own model and matches a backend stepped alone."""
    pair = [_pendulums(cfg, steps, captured=captured) for cfg, steps in _configs()]
    alone = [_pendulums(cfg, steps) for cfg, steps in _configs()]
    start = [_angles(backend) for backend in pair]

    for _ in range(5):
        nb.step(pair[0])
    np.testing.assert_array_equal(_angles(pair[1]), start[1])
    for _ in range(5):
        nb.step(pair[1])
    for _ in range(5):
        nb.step(alone[0])
        nb.step(alone[1])

    assert type(pair[0].solver) is not type(pair[1].solver)
    for backend, reference, initial in zip(pair, alone, start, strict=True):
        assert backend.step_graph.captured is captured
        assert np.all(np.abs(_angles(backend) - initial) > 1e-3)
        np.testing.assert_allclose(_angles(backend), _angles(reference), atol=1e-5)


def test_independent_managers_isolate_construction_and_state():
    """Two managers build, step, and close their own models without touching each other."""
    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True))
    first, second = (NewtonManager(_pendulum_builder(cfg), cfg, dt=DT, device=DEVICE) for _ in range(2))
    first.register_site(None, wp.transform_identity())
    try:
        for manager in (first, second):
            manager.reset()
        assert first.backend.model.shape_count == 1 and second.backend.model.shape_count == 0
        initial = _angles(second.backend)
        for _ in range(3):
            first.step()
        np.testing.assert_array_equal(_angles(second.backend), initial)
        assert np.max(np.abs(_angles(first.backend) - initial)) > 1e-4
        first.close()
        second.step()
        assert np.max(np.abs(_angles(second.backend) - initial)) > 1e-5
    finally:
        first.close()
        second.close()
