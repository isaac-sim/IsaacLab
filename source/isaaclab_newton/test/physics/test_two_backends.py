# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two Newton backends with different solvers, built and stepped side by side without the Newton manager."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonBackend, NewtonCfg
from isaaclab_newton.physics import newton_backend as nb

DEVICE = "cuda:0"
DT = 0.01


def _pendulums(physics_cfg: NewtonCfg, steps: int, *, captured: bool = False, num_worlds: int = 4) -> NewtonBackend:
    """Build a backend of one revolute pendulum per world, released from a different angle in each world.

    Each :func:`nb.step` advances ``steps`` physics steps, replayed from CUDA graphs when ``captured``.
    """
    builder = physics_cfg.class_type.create_builder(physics_cfg=physics_cfg)
    for world in range(num_worlds):
        builder.begin_world()
        link = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        )
        builder.add_articulation([joint])
        builder.joint_q[-1] = 0.2 + 0.2 * world
        builder.end_world()
    model = builder.finalize(device=DEVICE)
    backend = NewtonBackend(model, physics_cfg, dt=DT)
    nb.init_solver(backend)
    backend.steps_per_call = steps
    if not captured:
        backend.capture = None
    return backend


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


def test_backends_with_different_solvers_record_into_one_graph():
    """Both backends' steps replay from one caller-owned CUDA graph and match eager stepping."""
    pair = [_pendulums(cfg, steps) for cfg, steps in _configs()]
    eager = [_pendulums(cfg, steps) for cfg, steps in _configs()]

    # One eager step performs lazy solver allocations before anything is recorded.
    for backend in (*pair, *eager):
        nb.step(backend)
    with wp.ScopedCapture(device=DEVICE) as capture:
        for backend in pair:
            nb.record_step(backend)

    for _ in range(10):
        wp.capture_launch(capture.graph)
        for backend in eager:
            nb.step(backend)
    for backend, reference in zip(pair, eager, strict=True):
        np.testing.assert_allclose(_angles(backend), _angles(reference), atol=1e-5)


def test_record_step_requires_a_prepared_graph():
    """Recording refuses a step that was not prepared, so a caller's capture never allocates."""
    cfg, steps = _configs()[1]
    backend = _pendulums(cfg, steps)
    with pytest.raises(RuntimeError, match="Prepare the Newton step"):
        nb.record_step(backend)
