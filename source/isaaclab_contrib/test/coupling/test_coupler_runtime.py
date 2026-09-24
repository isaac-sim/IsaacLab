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
from isaaclab_newton.physics import NewtonManager, XPBDSolverCfg
from newton import CollisionPipeline, Mesh, Model, ModelBuilder
from newton.solvers import SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
)


@pytest.fixture
def isolated_newton_manager(monkeypatch: pytest.MonkeyPatch):
    """Isolate every global manager slot touched by coupler construction."""
    clean_values = {
        "backend": SimpleNamespace(model=None),
        "_solver": None,
        "_use_single_state": None,
        "_contacts": None,
        "_collision_pipeline": None,
        "_collision_cfg": None,
        "_needs_collision_pipeline": False,
        "_supports_contact_sensors": True,
        "_supports_rigid_body_force_input": False,
        "_report_contacts": False,
    }
    for name, value in clean_values.items():
        monkeypatch.setattr(NewtonManager, name, value)
    yield


def _build_overlapping_body_model(*, mesh_contact: bool = False) -> Model:
    """Build two labeled free bodies with one rigid contact on the CPU.

    Args:
        mesh_contact: Replace the source sphere with a triangle mesh to exercise
            triangle-pair allocation and contact reduction in the capacity test.
    """
    builder = ModelBuilder(gravity=-9.81)
    for x, label in ((-0.09, "/World/Source/body"), (0.09, "/World/Destination/body")):
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
    builder.color()
    return builder.finalize(device="cpu")


def _entry_configs() -> list[CouplerEntryCfg]:
    """Return fresh entry configs so selector resolution cannot leak between cases."""
    return [
        CouplerEntryCfg(
            name="source",
            solver_cfg=XPBDSolverCfg(iterations=2),
            bodies=[r"/World/Source/body"],
        ),
        CouplerEntryCfg(
            name="destination",
            solver_cfg=XPBDSolverCfg(iterations=2),
            bodies=[r"/World/Destination/body"],
        ),
    ]


def test_proxy_destination_can_receive_only_proxy_bodies(isolated_newton_manager):
    model = _build_overlapping_body_model()
    solver_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="source",
                solver_cfg=XPBDSolverCfg(iterations=2),
                bodies=[r"/World/Source/body"],
            ),
            CouplerEntryCfg(name="destination", solver_cfg=XPBDSolverCfg(iterations=2)),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="source",
                destination="destination",
                bodies=[r"/World/Source/body"],
            )
        ],
    )

    NewtonManager.backend.model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)

    assert NewtonManager._solver._entries["destination"].proxy_body_local_indices.numpy().tolist() == [0]


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
    isolated_newton_manager,
):
    """Construct, prepare contacts, reset, and step the pinned Newton solver."""
    model = _build_overlapping_body_model(mesh_contact=algorithm == "admm_capacity")
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

    NewtonManager.backend.model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)
    solver = NewtonManager._solver

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

    NewtonCouplerManager._initialize_contacts()
    collision_pipeline = NewtonManager._collision_pipeline
    contacts = NewtonManager._contacts
    assert isinstance(collision_pipeline, CollisionPipeline)
    assert contacts is not None
    assert set(solver._entry_contact_buffers) == {"source", "destination"}

    state_0 = model.state()
    state_1 = model.state()
    solver.reset(state_0)
    assert solver.entry_output_state_valid() is False

    collision_pipeline.collide(state_0, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) >= 1
    body_q_before = state_0.body_q.numpy().copy()

    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 60.0)

    body_q_after = state_1.body_q.numpy()
    assert solver.entry_output_state_valid() is True
    assert np.all(np.isfinite(body_q_after))
    assert np.all(np.isfinite(state_1.body_qd.numpy()))
    assert np.any(body_q_after[:, 2] < body_q_before[:, 2])
    if algorithm == "admm_capacity":
        assert int(solver._admm_internal_contacts.rigid_contact_count.numpy()[0]) >= 1
        assert solver._admm_internal_contacts.rigid_contact_match_index is not None
        assert int(reducer.ht_insert_failures.numpy()[0]) == 0

    solver.reset(state_1)
    assert solver.entry_output_state_valid() is False
