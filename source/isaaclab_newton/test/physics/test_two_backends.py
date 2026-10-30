# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backends with different solvers, built and stepped side by side without a manager."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    NewtonBackend,
    NewtonBackendCfg,
    NewtonCfg,
    create_builder,
)

from isaaclab.test.utils import DeviceScope, test_devices

DT = 0.01


def _pendulums(physics_cfg: NewtonCfg, steps: int, device: str, *, captured=False) -> NewtonBackend:
    """A backend of one pendulum per world that advances ``steps`` physics steps per call."""
    builder = _pendulum_builder(physics_cfg)
    physics_cfg.solver_cfg.class_type.prepare_builder(physics_cfg.solver_cfg, builder)
    backend = NewtonBackend(NewtonBackendCfg(physics_cfg=physics_cfg, device=device), builder, dt=DT)
    backend.init_solver()
    if not captured:
        backend.capture = None
        backend.invalidate_graph()
    # Without Newton actuators, an active actuator path lets one step run the whole decimation loop.
    backend.actuators_active = True
    backend.set_decimation(steps)
    return backend


def _pendulum_builder(physics_cfg: NewtonCfg):
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
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_backends_with_different_solvers_step_side_by_side(device, captured):
    """Each backend advances only its own model and matches a backend stepped alone."""
    pair = [_pendulums(cfg, steps, device, captured=captured) for cfg, steps in _configs()]
    alone = [_pendulums(cfg, steps, device) for cfg, steps in _configs()]
    start = [_angles(backend) for backend in pair]

    for _ in range(5):
        pair[0].step()
    np.testing.assert_array_equal(_angles(pair[1]), start[1])
    for _ in range(5):
        pair[1].step()
    for _ in range(5):
        alone[0].step()
        alone[1].step()

    assert type(pair[0].solver) is not type(pair[1].solver)
    for backend, reference, initial in zip(pair, alone, start, strict=True):
        assert (backend.graph is not None) is captured
        assert np.all(np.abs(_angles(backend) - initial) > 1e-3)
        np.testing.assert_allclose(_angles(backend), _angles(reference), atol=1e-5)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reset_masks_stay_with_their_backend(device):
    """Invalidating one world of one backend recomputes only that world's body state."""
    pair = [_pendulums(cfg, steps, device) for cfg, steps in _configs()]
    for backend in pair:
        backend.step()
    body_q = [backend.state_0.body_q.numpy().copy() for backend in pair]
    target, other = pair

    joint_q = target.state_0.joint_q.numpy()
    joint_q[:] = 0.0
    target.state_0.joint_q.assign(joint_q)
    env_mask = wp.array([True, False, False, False], dtype=wp.bool, device=device)
    articulation_ids = wp.array(np.arange(4, dtype=np.int32).reshape(4, 1), dtype=wp.int32, device=device)
    target.invalidate_fk(env_mask, articulation_ids=articulation_ids)

    assert not other.world_mask.numpy().any() and not other.fk_mask.numpy().any()
    target.forward()
    other.forward()

    moved = np.any(target.state_0.body_q.numpy() != body_q[0], axis=1)
    assert moved.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(other.state_0.body_q.numpy(), body_q[1])
    assert not target.world_mask.numpy().any() and not target.fk_mask.numpy().any()
