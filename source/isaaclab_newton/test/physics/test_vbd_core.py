# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.physics import NewtonBackendCfg, NewtonBuilderCfg, NewtonCfg, NewtonManager, NewtonSoftContactCfg
from isaaclab_newton.physics.newton_solver import solver_kwargs
from newton import ModelBuilder
from newton.solvers import SolverVBD

from isaaclab.sim import BackendCfg, SimulationContext
from isaaclab.utils import replace

NOT_GUARANTEED = wp.DeterministicMode.NOT_GUARANTEED


# The soft-contact and simulation axes are independent, so each value is covered once.
@pytest.mark.parametrize(
    ("soft_contact_cfg", "expected", "simulation"),
    [
        pytest.param(None, (7.0, 8.0, 9.0), False, id="preserve-render"),
        pytest.param(
            NewtonSoftContactCfg(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            True,
            id="override-physics",
        ),
    ],
)
def test_soft_contact_cfg_updates_finalized_model(monkeypatch, soft_contact_cfg, expected, simulation):
    """Cfg-keyed construction shares the builder and applies physics settings before state allocation."""
    state_values = []

    class Model:
        soft_contact_ke = 7.0
        soft_contact_kd = 8.0
        soft_contact_mu = 9.0
        device = "cpu"
        world_count = 0
        articulation_count = 0

        def state(self):
            state_values.append((self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
            return object()

        def control(self):
            return object()

    model = Model()
    monkeypatch.setattr(ModelBuilder, "finalize", lambda self, device: model)
    physics_cfg = NewtonCfg(soft_contact_cfg=soft_contact_cfg) if simulation else object()
    builder_cfg = NewtonBuilderCfg(physics_cfg=physics_cfg)
    assert not isinstance(builder_cfg, BackendCfg)
    assert not hasattr(builder_cfg, "close")
    cfg = NewtonBackendCfg(physics_cfg=physics_cfg, device="cpu")
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    builder = sim.get_or_create_backend(builder_cfg)
    backend = sim.get_or_create_backend(cfg, dt=0.01)
    assert not {"_builder", "set_builder", "_model", "_state_0", "_state_1"}.intersection(vars(NewtonManager))
    assert not {"sync_transforms_to_fabric", "sync_transforms_to_usd"}.intersection(vars(NewtonManager))
    assert cfg.physics_cfg is physics_cfg
    assert backend is sim.get_or_create_backend(replace(cfg))
    assert builder is sim.get_or_create_backend(builder_cfg)
    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected
    assert state_values == [expected] * (2 if simulation else 1)
    assert (backend.state_1 is not None) == simulation
    assert (backend.control is not None) == simulation
    sim.close_backend(backend)
    assert backend.model is backend.state_0 is backend.state_1 is backend.control is None
    assert sim.get_or_create_backend(builder_cfg) is builder
    sim.close_backend(builder)
    assert sim._backend_registry == []


def test_vbd_colors_builder_before_finalization():
    """Particles coupled by active cloth bending must receive distinct solver colors."""
    physics = importlib.import_module("isaaclab_newton.physics")
    builder = ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.01,
        edge_ke=1.0,
    )
    physics.VBDSolver.prepare_builder(physics.VBDSolverCfg(), builder)
    colors = {int(particle): color for color, group in enumerate(builder.particle_color_groups) for particle in group}
    assert len(set(colors.values())) == builder.particle_count


def test_vbd_solver_force_input_capability():
    """VBD rejects rigid forces when an external solver integrates rigid bodies.

    The default (VBD integrates rigid bodies itself) is covered end to end by
    ``test_reset_binds_solver_and_capabilities``.
    """
    physics = importlib.import_module("isaaclab_newton.physics")
    solver_cfg = physics.VBDSolverCfg(integrate_with_external_rigid_solver=True)
    solver = physics.VBDSolver.build(solver_cfg, ModelBuilder().finalize(device="cpu"), NOT_GUARANTEED)

    assert physics.VBDSolver.applies_body_forces(solver_cfg, solver) is False


@pytest.mark.parametrize(
    "overrides", [{}, {"rigid_compliant_alm": False, "friction_epsilon": 1.0e-3}], ids=["defaults", "legacy"]
)
def test_vbd_rigid_solver_controls(overrides):
    """VBD configuration preserves native defaults and forwards explicit controls."""
    physics = importlib.import_module("isaaclab_newton.physics")
    kwargs = solver_kwargs(SolverVBD, physics.VBDSolverCfg(**overrides), wp.DeterministicMode.NOT_GUARANTEED)
    assert kwargs["rigid_body_contact_buffer_size"] == 64
    model = ModelBuilder().finalize(device="cpu")
    native_solver = SolverVBD(model, **overrides)
    configured_solver = physics.VBDSolver.build(physics.VBDSolverCfg(**overrides), model, NOT_GUARANTEED)
    for name in (
        "rigid_compliant_alm",
        "rigid_joint_linear_ke",
        "rigid_joint_angular_ke",
        "friction_epsilon",
    ):
        assert getattr(configured_solver, name) == getattr(native_solver, name)


def test_vbd_compliant_alm_cable_stiffness():
    """The manager's ALM solver retains finite cable stiffness under gravity."""
    physics = importlib.import_module("isaaclab_newton.physics")
    gravity = 9.81
    stretch_stiffness = 1.0e3
    builder = ModelBuilder(gravity=(0.0, 0.0, -gravity))
    body = builder.add_link()
    builder.add_shape_capsule(body=body, radius=0.01, half_height=0.1)
    mass = builder.body_mass[body]
    joint = builder.add_joint_rod(
        parent=-1,
        child=body,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=2.0 * (mass * stretch_stiffness) ** 0.5,
        bend_stiffness=5.0,
        bend_damping=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device="cpu")
    solver_cfg = physics.VBDSolverCfg(rigid_compliant_alm=True, rigid_body_contact_buffer_size=256)
    solver = physics.VBDSolver.build(solver_cfg, model, NOT_GUARANTEED)
    assert solver.rigid_compliant_alm is True
    assert solver.body_body_contact_indices.size == model.body_count * 256

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    # At equilibrium the spring force balances the weight: k * extension = m * g.
    expected_extension = mass * gravity / stretch_stiffness
    assert state_0.body_q.numpy()[body, 2] == pytest.approx(-expected_extension, rel=0.01)


def test_vbd_structural_joint_stiffness_under_load():
    """A fixed joint balances gravity with independently configured linear and angular springs."""
    physics = importlib.import_module("isaaclab_newton.physics")
    gravity, lever = 9.81, 0.1
    builder = ModelBuilder(gravity=(0.0, 0.0, -gravity))
    body = builder.add_link(xform=wp.transform(wp.vec3(lever, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=0.05)
    mass = builder.body_mass[body]
    joint = builder.add_joint_fixed(
        parent=-1, child=body, child_xform=wp.transform(wp.vec3(-lever, 0.0, 0.0), wp.quat_identity())
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device="cpu")
    linear_ke, angular_ke = 1.0e3, 1.0e2
    solver_cfg = replace(
        physics.VBDSolverCfg(rigid_compliant_alm=True, iterations=15),
        rigid_joint_linear_ke=linear_ke,
        rigid_joint_angular_ke=angular_ke,
    )
    solver = physics.VBDSolver.build(solver_cfg, model, NOT_GUARANTEED)
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(480):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    pose = state_0.body_q.numpy()[body]
    angle = 2.0 * math.atan2(float(pose[4]), float(pose[6]))
    anchor_height = float(pose[2]) + lever * math.sin(angle)
    # The joint's linear force balances weight; its angular torque balances the offset load.
    assert -anchor_height * linear_ke == pytest.approx(mass * gravity, rel=0.01)
    assert angle * angular_ke == pytest.approx(mass * gravity * lever * math.cos(angle), rel=0.01)


def test_vbd_rebuilds_particle_bvh_before_physics_step():
    """VBD rebuilds its particle BVH before the solver advances each physics step."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []

    class State:
        body_count = particle_count = 0

        def clear_forces(self):
            pass

        def assign(self, other):
            pass

    class NativeSolver:
        model = SimpleNamespace(particle_count=1, articulation_count=0, joint_coord_count=0)

        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

        def step(self, state_in, state_out, control, contacts, dt):
            events.append(("step", state_in))

    state_0, state_1 = State(), State()
    manager = NewtonManager(NewtonCfg())
    manager._cfg = NewtonCfg(solver_cfg=physics.VBDSolverCfg())
    manager.backend = SimpleNamespace(
        model=SimpleNamespace(particle_count=1),
        solver=NativeSolver(),
        state_0=state_0,
        state_1=state_1,
        control=None,
        collision_pipeline=None,
        contacts=None,
        num_substeps=1,
        solver_dt=0.01,
        contact_sensors={},
        imu_sensors=[],
    )

    manager._simulate_physics_only()

    assert events == [("rebuild", state_0), ("step", state_0)]
