# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Kitless runtime tests for Newton's coupled-solver configurations."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager, XPBDSolverCfg
from newton import CollisionPipeline, Model, ModelBuilder
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
        "_model": None,
        "_solver": None,
        "_use_single_state": None,
        "_contacts": None,
        "_collision_pipeline": None,
        "_collision_cfg": None,
        "_needs_collision_pipeline": False,
        "_supports_contact_sensors": True,
        "_report_contacts": False,
    }
    for name, value in clean_values.items():
        monkeypatch.setattr(NewtonManager, name, value)
    yield


def _build_overlapping_body_model() -> Model:
    """Build two labeled free bodies with one rigid contact on the CPU."""
    builder = ModelBuilder(gravity=-9.81)
    for x, label in ((-0.09, "/World/Source/body"), (0.09, "/World/Destination/body")):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label=label,
        )
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

    NewtonManager._model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)

    assert NewtonManager._solver._entries["destination"].proxy_body_local_indices.numpy().tolist() == [0]


@pytest.mark.parametrize(
    ("algorithm", "expected_solver_type"),
    [
        pytest.param("proxy", SolverCoupledProxy, id="proxy"),
        pytest.param("admm", SolverCoupledADMM, id="admm"),
    ],
)
def test_real_coupler_constructs_resets_and_steps(
    algorithm: str,
    expected_solver_type: type,
    isolated_newton_manager,
):
    """Construct, prepare contacts, reset, and step the pinned Newton solver."""
    model = _build_overlapping_body_model()
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

    NewtonManager._model = model
    NewtonCouplerManager._build_solver(model, solver_cfg)
    solver = NewtonManager._solver

    assert isinstance(solver, expected_solver_type)
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

    solver.reset(state_1)
    assert solver.entry_output_state_valid() is False
