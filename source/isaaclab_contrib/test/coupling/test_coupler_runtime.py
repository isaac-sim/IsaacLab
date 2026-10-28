# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Kitless runtime tests for Newton's coupled-solver configurations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    NewtonBackend,
    NewtonBackendCfg,
    NewtonCfg,
    NewtonManager,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from newton import CollisionPipeline, Mesh, ModelBuilder
from newton.solvers import SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy

from isaaclab.sim import SimulationContext

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)


def _bind_coupler(builder: ModelBuilder, solver_cfg: CouplerProxyCfg | CouplerAdmmCfg) -> NewtonManager:
    """Build a manager over a backend finalized from ``builder`` whose coupler solver and contacts are constructed."""
    cfg = NewtonCfg(solver_cfg=solver_cfg)
    manager = NewtonManager(cfg)
    manager._device = "cpu"
    manager._sim = SimpleNamespace(
        get_or_create_backend=lambda _: builder,
        get_clone_plan=lambda: SimpleNamespace(env_template="/Lab/Cells/Cell{}"),
        _prepare_newton_visualizer_for_capture=lambda: None,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SimulationContext, "instance", lambda: manager._sim)
        manager.backend = NewtonBackend(NewtonBackendCfg(physics_cfg=cfg, device="cpu"), dt=1.0 / 60.0)
    manager.initialize_solver()
    return manager


def _build_overlapping_bodies(*, mesh_contact: bool = False, particle: bool = False) -> ModelBuilder:
    """Build two labeled free bodies with one rigid contact on the CPU.

    Args:
        mesh_contact: Replace the source sphere with a triangle mesh to exercise
            triangle-pair allocation and contact reduction in the capacity test.
        particle: Add a free particle above the rigid bodies.
    """
    builder = ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for x, label in ((-0.09, "/Lab/Cells/Cell0/Source/body"), (0.09, "/World/Destination/body")):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label=label,
        )
        if mesh_contact and x < 0.0:
            mesh = Mesh(
                vertices=np.array([[0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [0.1, 0.0, 0.1]], dtype=np.float32),
                indices=np.array([0, 1, 2], dtype=np.int32),
                compute_inertia=False,
            )
            builder.add_shape_mesh(body=body, mesh=mesh, label=f"{label}/shape")
        else:
            builder.add_shape_sphere(body=body, radius=0.1, label=f"{label}/shape")
    if particle:
        builder.add_particle(pos=(0.0, 0.0, 2.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    builder.color()
    return builder


def _entry_configs() -> list[CouplerEntryCfg]:
    """Return fresh entry configs so selector resolution cannot leak between cases."""
    return [
        CouplerEntryCfg(
            name="source",
            solver_cfg=XPBDSolverCfg(iterations=2),
            bodies=["{ENV_REGEX_NS}/Source/body"],
        ),
        CouplerEntryCfg(
            name="destination",
            solver_cfg=XPBDSolverCfg(iterations=2),
            bodies=[r"/World/Destination/body"],
        ),
    ]


@pytest.mark.parametrize(
    "destination_solver_cfg",
    [
        pytest.param(XPBDSolverCfg(iterations=2), id="xpbd"),
        pytest.param(
            VBDSolverCfg(rigid_contact_history=True, integrate_with_external_rigid_solver=True),
            id="vbd-external-rigid",
        ),
    ],
)
def test_proxy_destination_can_receive_only_proxy_bodies(destination_solver_cfg):
    builder = _build_overlapping_bodies()
    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="source",
                solver_cfg=XPBDSolverCfg(iterations=2),
                bodies=["{ENV_REGEX_NS}/Source/body"],
            ),
            CouplerEntryCfg(name="destination", solver_cfg=destination_solver_cfg),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="source",
                destination="destination",
                bodies=["{ENV_REGEX_NS}/Source/body"],
            )
        ],
    )

    backend = _bind_coupler(builder, solver_cfg).backend

    assert backend.solver._entries["destination"].proxy_body_local_indices.numpy().tolist() == [0]


def test_particle_only_vbd_entry_ignores_rigid_contact_history():
    """History is inactive for a particle-only view even when the parent model has bodies."""
    builder = _build_overlapping_bodies(particle=True)
    solver_cfg = CouplerAdmmCfg(
        entries=[
            CouplerEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg(), bodies=[r"/World/.*"]),
            CouplerEntryCfg(name="particles", solver_cfg=VBDSolverCfg(rigid_contact_history=True), all_particles=True),
        ],
        contact_pairs=[],
        iterations=1,
    )
    manager = _bind_coupler(builder, solver_cfg)
    model = manager.backend.model
    solver = manager.backend.solver
    assert solver.view("particles").body_count == 0

    state_0, state_1 = model.state(), model.state()
    manager.backend.collision_pipeline.collide(state_0, manager.backend.contacts)
    dt = 1.0 / 60.0
    solver.step(state_0, state_1, model.control(), manager.backend.contacts, dt)

    # A free particle starting at rest follows semi-implicit Euler under gravity.
    assert state_1.particle_q.numpy()[0, 2] == pytest.approx(2.0 - 9.81 * dt**2)


@pytest.mark.parametrize(
    ("destination_bodies", "proxies"),
    [
        pytest.param([r"/World/Destination/body"], [], id="owned-rigid"),
        pytest.param(
            [],
            [CouplerProxyMappingCfg(source="source", destination="destination", bodies=["{ENV_REGEX_NS}/Source/body"])],
            id="received-rigid",
        ),
    ],
)
def test_coupled_vbd_rigid_contact_history_is_rejected(destination_bodies, proxies):
    """Both owned and received bodies make coupled VBD history unsupported."""
    builder = _build_overlapping_bodies()
    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(name="source", solver_cfg=XPBDSolverCfg(), bodies=["{ENV_REGEX_NS}/Source/body"]),
            CouplerEntryCfg(
                name="destination", solver_cfg=VBDSolverCfg(rigid_contact_history=True), bodies=destination_bodies
            ),
        ],
        proxies=proxies,
    )
    with pytest.raises(NotImplementedError, match=r"destination.*rigid_contact_history"):
        _bind_coupler(builder, solver_cfg)


@pytest.mark.parametrize(
    ("algorithm", "expected_solver_type"),
    [
        pytest.param("proxy", SolverCoupledProxy, id="proxy"),
        pytest.param("admm", SolverCoupledADMM, id="admm"),
        pytest.param("admm_capacity", SolverCoupledADMM, id="admm_capacity"),
    ],
)
def test_real_coupler_constructs_resets_and_steps(
    algorithm: str,
    expected_solver_type: type,
):
    """Resolve cloned/global selectors and step without overwriting nested-solver joint state."""
    builder = _build_overlapping_bodies(mesh_contact=algorithm == "admm_capacity")
    entries = _entry_configs()
    if algorithm == "proxy":
        solver_cfg = CouplerProxyCfg(
            entries=entries,
            proxies=[
                CouplerProxyMappingCfg(
                    source="source",
                    destination="destination",
                    bodies=[0],
                )
            ],
            iterations=1,
        )
    else:
        solver_cfg = CouplerAdmmCfg(entries=entries, iterations=1)
        if algorithm == "admm_capacity":
            solver_cfg.contact_max_triangle_pairs = 8192
            # The table can exceed the deterministic contact-ID limit independently.
            solver_cfg.contact_reduction_hashtable_size_factor = 256.0
            solver_cfg.rigid_contact_matching = "latest"
            solver_cfg.contact_matching_pos_threshold = 0.001
            solver_cfg.contact_matching_normal_dot_threshold = 0.9

    if algorithm == "admm_capacity" and not hasattr(SolverCoupledADMM.Config, "contact_max_triangle_pairs"):
        with pytest.raises(RuntimeError, match=r"Newton.*does not support contact_max_triangle_pairs"):
            _bind_coupler(builder, solver_cfg)
        return
    manager = _bind_coupler(builder, solver_cfg)
    backend = manager.backend
    model = backend.model
    solver = backend.solver

    assert isinstance(solver, expected_solver_type)
    if algorithm == "admm_capacity":
        internal_pipeline = solver._admm_collision_pipeline
        assert internal_pipeline.narrow_phase.max_triangle_pairs == solver_cfg.contact_max_triangle_pairs
        reducer = internal_pipeline.narrow_phase.global_contact_reducer
        requested_slots = int(
            solver_cfg.contact_max_triangle_pairs * solver_cfg.contact_reduction_hashtable_size_factor
        )
        assert reducer.hashtable.capacity == 1 << (requested_slots - 1).bit_length()
        assert internal_pipeline.contact_matching == solver_cfg.rigid_contact_matching
        np.testing.assert_array_equal(internal_pipeline.shape_pairs_filtered.numpy(), [[0, 1]])
    assert solver.entry_names() == ("source", "destination")
    for name in solver.entry_names():
        nested_solver = solver.solver(name)
        assert isinstance(nested_solver, SolverXPBD)
        assert nested_solver.model is solver.view(name)

    collision_pipeline = backend.collision_pipeline
    contacts = backend.contacts
    assert isinstance(collision_pipeline, CollisionPipeline)
    assert contacts is not None
    assert set(solver._entry_contact_buffers) == {"source", "destination"}

    state_0 = model.state()
    state_1 = model.state()
    solver.reset(state_0)
    assert solver.entry_output_state_valid() is False
    state_0.joint_qd.fill_(0.125)

    collision_pipeline.collide(state_0, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) >= 1
    body_q_before = state_0.body_q.numpy().copy()

    # The native coupled solver, not global inverse kinematics, owns the published joint state.
    native_output = model.state()
    control = model.control()
    solver.step(state_0, native_output, control, contacts, 1.0 / 60.0)
    solver.reset(state_0, flags=0)
    backend.state_0 = state_0
    backend.state_1 = state_1
    backend.control = control
    manager._run_solver_substeps(contacts)
    state_1 = backend.state_0

    body_q_after = state_1.body_q.numpy()
    assert solver.entry_output_state_valid() is True
    assert np.all(np.isfinite(body_q_after))
    assert np.all(np.isfinite(state_1.body_qd.numpy()))
    assert np.any(body_q_after[:, 2] < body_q_before[:, 2])
    np.testing.assert_array_equal(state_1.joint_q.numpy(), native_output.joint_q.numpy())
    np.testing.assert_array_equal(state_1.joint_qd.numpy(), native_output.joint_qd.numpy())
    if algorithm == "admm_capacity":
        assert int(solver._admm_internal_contacts.rigid_contact_count.numpy()[0]) >= 1
        assert solver._admm_internal_contacts.rigid_contact_match_index is not None
        assert int(reducer.ht_insert_failures.numpy()[0]) == 0

    solver.reset(state_1)
    assert solver.entry_output_state_valid() is False
