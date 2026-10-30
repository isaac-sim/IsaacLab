# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`NewtonManager` and its :class:`NewtonBackend`: lifecycle, the step, capture, reset masks and
queries."""

from __future__ import annotations

import gc
import logging
from functools import partial as P
from types import SimpleNamespace

import isaaclab_newton.physics.newton_manager as newton_manager_module
import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonBackend,
    NewtonBackendCfg,
    NewtonBuilderCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSolver,
    NewtonSolverCfg,
    VBDSolverCfg,
    XPBDSolverCfg,
    create_builder,
    newton_backend,
)
from isaaclab_newton.physics.newton_scene_data import NewtonSceneDataBackend
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
from isaaclab_newton.test.fixtures.mock_newton import backend_stub
from newton import ModelFlags
from newton.actuators import DrivePID
from newton.solvers import SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverVBD

from isaaclab.physics import PhysicsEvent
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

# ----- Configuration and lifecycle -------------------------------------------------------


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, should_warn",
    [(8, 0, False), (8, 7, False), (8, 8, True), (8, 16, True)],
)
def test_newton_cfg_warns_when_collision_decimation_never_fires(
    num_substeps, collision_decimation, should_warn, caplog
):
    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.newton_manager_cfg"):
        NewtonCfg(num_substeps=num_substeps, collision_decimation=collision_decimation)

    assert any("collision_decimation" in record.getMessage() for record in caplog.records) is should_warn


def test_newton_shape_cfg_defaults_match_newton():
    """Applying the default shape settings never overrides the materials Newton authors."""
    upstream = newton.ModelBuilder().default_shape_cfg
    shape_cfg = NewtonShapeCfg()

    assert (shape_cfg.ke, shape_cfg.kd, shape_cfg.mu) == (upstream.ke, upstream.kd, upstream.mu)


def test_reset_lets_consumers_bind_each_new_backend_before_the_solver_exists(monkeypatch):
    """Each hard reset finalizes a new backend, dispatches ``PHYSICS_READY``, then builds the solver.

    Graph-replay invalidation from before a hard reset belongs to the discarded graph, so the new model's consumers
    can read transforms before the solver is rebuilt.
    """
    events, backends = [], []
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0, device="cpu", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False)
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:

        def on_physics_ready(_):
            events.append("ready")
            backend = sim.get_or_create_backend(NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device))
            assert backend is sim.backend_manager.backend
            assert backend.solver is None
            backends.append(backend)
            sim.get_scene_data_provider().get_transforms(SceneDataFormat.Transform())
            # Asset pose reads resolve pending FK through the manager.
            sim.backend_manager.forward()

        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        builder.add_joint_revolute(parent=-1, child=builder.add_body(mass=1.0), axis=(0, 0, 1))
        monkeypatch.setattr(sim, "_prepare_newton_visualizer_for_capture", lambda: events.append("prepare"))
        sim.backend_manager.register_callback(on_physics_ready, PhysicsEvent.PHYSICS_READY, wrap_weak_ref=False)

        sim.reset()
        assert isinstance(sim.backend_manager.backend.solver, SolverMuJoCo)
        # as a write during graph capture does
        sim.backend_manager.backend.transforms_may_change_on_graph_replay = True
        sim.reset()

    assert events == ["ready", "prepare"] * 2
    assert backends[0] is not backends[1]


@pytest.mark.parametrize(
    ("solver_cfg_factory", "solver_cls", "collision_pipeline"),
    [
        pytest.param(P(MJWarpSolverCfg, use_mujoco_contacts=True), SolverMuJoCo, False, id="mjwarp_internal"),
        pytest.param(P(MJWarpSolverCfg, use_mujoco_contacts=False), SolverMuJoCo, True, id="mjwarp_pipeline"),
        pytest.param(XPBDSolverCfg, newton.solvers.SolverXPBD, True, id="xpbd"),
        pytest.param(VBDSolverCfg, SolverVBD, True, id="vbd"),
        pytest.param(FeatherstoneSolverCfg, newton.solvers.SolverFeatherstone, True, id="featherstone"),
        pytest.param(P(KaminoPADMMSolverCfg, use_collision_detector=True), SolverKamino, False, id="kamino_internal"),
        pytest.param(P(KaminoPADMMSolverCfg, use_collision_detector=False), SolverKamino, True, id="kamino_pipeline"),
        pytest.param(P(MPMSolverCfg, max_iterations=2, voxel_size=0.05), SolverImplicitMPM, False, id="mpm"),
    ],
)
def test_reset_builds_the_configured_solver_and_steps(solver_cfg_factory, solver_cls, collision_pipeline):
    """Every solver configuration builds its Newton solver through the manager and advances a step."""
    solver_cfg = solver_cfg_factory()
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0, device="cuda:0", physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False)
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        if solver_cls is SolverImplicitMPM:
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        elif solver_cls is SolverVBD:
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.1),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.0, 0.1, 0.0)],
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=0.01,
            )
        else:
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
            if solver_cls is SolverKamino and solver_cfg.use_collision_detector:
                builder.add_shape_sphere(body=body, radius=0.05)
                builder.add_ground_plane()

        sim.reset()
        sim.step(render=False)

        backend = sim.backend_manager.backend
        assert isinstance(sim.backend_manager, NewtonManager)
        assert isinstance(backend.solver, solver_cls)
        assert (backend.collision_pipeline is not None) is collision_pipeline


def test_step_after_a_hard_reset_never_reads_the_old_model():
    """Collision and the captured step bind the new model's buffers (CUDA error 700 before the fix)."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), num_substeps=2),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        for _ in range(8):
            body = builder.add_body(mass=1.0)
            builder.add_joint_free(child=body)
            builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_ground_plane()
        sim.reset()
        old_backend = sim.backend_manager.backend
        old_model = old_backend.collision_pipeline.model
        sim.step(render=False)

        sim.reset()
        assert sim.backend_manager.backend is not old_backend
        assert old_backend.model is old_backend.state_0 is old_backend.control is None
        assert sum(isinstance(cfg, NewtonBackendCfg) for cfg, _ in sim._backend_registry) == 1

        # Free the arrays collide() reads off the old model and reuse their memory, so stale reads see garbage.
        for attr in ("shape_transform", "shape_body", "shape_type", "shape_scale", "shape_collision_radius"):
            setattr(old_model, attr, None)
        gc.collect()
        wp.synchronize_device("cuda:0")
        _churn = [wp.zeros(1 << 16, dtype=wp.float32, device="cuda:0") for _ in range(128)]  # noqa: F841

        sim.step(render=False)
        sim.step(render=False)
        wp.synchronize_device("cuda:0")


class _State:
    """Named state stand-in that records force clearing and copies."""

    body_count = particle_count = 0

    def __init__(self, name, events):
        self.name, self._events = name, events

    def clear_forces(self):
        self._events.append(("clear", self.name))

    def assign(self, other):
        self._events.append(("assign", self.name, other.name))


def _recording_backend(
    events, *, single_state=True, num_substeps=1, device="cpu", captures=True, ignored_model_changes=None, **cfg
):
    """A backend whose solver hooks and states record every operation of a step."""

    class RecordingSolver(NewtonSolver):
        @classmethod
        def build(cls, cfg, model, mode, env_template=None):
            return SimpleNamespace(notify_model_changed=lambda flags: events.append(("notify", flags)))

        @classmethod
        def supports_capture(cls, cfg, solver):
            return captures

        @classmethod
        def uses_collision_pipeline(cls, cfg, solver):
            return False

        @classmethod
        def pre_step(cls, cfg, solver, state):
            events.append(("pre_step", state.name))

        @classmethod
        def step(cls, cfg, solver, state_in, state_out, *_args):
            events.append(("step", state_in.name, state_out.name))

        @classmethod
        def post_step(cls, cfg, solver, state):
            events.append(("post_step", state.name))

        @classmethod
        def eval_fk(cls, cfg, solver, state, worlds, articulations):
            events.append(("fk",))

        @classmethod
        def check_step(cls, cfg, solver, captured, debug):
            events.append(("check", captured))

    RecordingSolver.single_state = single_state
    RecordingSolver.ignored_model_changes = ignored_model_changes or {}
    physics_cfg = NewtonCfg(num_substeps=num_substeps, solver_cfg=NewtonSolverCfg(class_type=RecordingSolver), **cfg)
    backend = backend_stub(
        cfg=physics_cfg,
        device=wp.get_device(device),
        deterministic_mode=wp.DeterministicMode.NOT_GUARANTEED,
        model=object(),
        state_0=_State("state_0", events),
        state_1=_State("state_1", events),
        control=object(),
        collision_pipeline=None,
        contacts=None,
        dt=0.002,
        num_substeps=num_substeps,
        solver_dt=0.002 / num_substeps,
        actuators=None,
        actuators_active=False,
        contact_sensors={},
        imu_sensors=[],
        world_mask=wp.zeros(2, dtype=wp.bool, device=device),
        fk_mask=wp.zeros(1, dtype=wp.bool, device=device),
        kinematics_dirty=False,
        transforms_may_change_on_graph_replay=False,
        model_changes=set(),
        warned_model_changes=set(),
        pre_actuator_callbacks=[],
        post_actuator_callbacks=[],
        state_force_callbacks=[],
        post_step_callbacks=[],
        solver_init_callbacks=[],
        decimation=1,
        capture=None,
        graph=None,
    )
    backend.init_solver()
    events.clear()
    return backend


# ----- Solver initialization and contacts ----------------------------------------------------


@pytest.mark.parametrize("solver_cfg", [KaminoPADMMSolverCfg(), MPMSolverCfg()], ids=["kamino", "mpm"])
def test_init_solver_rejects_a_determinism_guarantee_the_solver_cannot_honor(solver_cfg):
    cfg = NewtonCfg(solver_cfg=solver_cfg, deterministic_mode="gpu_to_gpu")
    backend = backend_stub(cfg=cfg, model=None, deterministic_mode=newton_backend._resolve_deterministic_mode(cfg))

    with pytest.raises(ValueError, match="not supported"):
        backend.init_solver()


@pytest.mark.parametrize(
    ("solver_cfg", "deterministic", "deterministic_mode", "expected"),
    [
        pytest.param(XPBDSolverCfg(), False, "not_guaranteed", wp.DeterministicMode.NOT_GUARANTEED, id="no_request"),
        pytest.param(XPBDSolverCfg(), True, "not_guaranteed", wp.DeterministicMode.RUN_TO_RUN, id="generic_request"),
        pytest.param(MJWarpSolverCfg(), True, "gpu_to_gpu", wp.DeterministicMode.GPU_TO_GPU, id="explicit_mode_wins"),
        pytest.param(
            MJWarpSolverCfg(use_mujoco_cpu=True), True, "not_guaranteed", wp.DeterministicMode.NOT_GUARANTEED, id="cpu"
        ),
    ],
)
def test_deterministic_request_resolves_to_the_mode_the_solver_applies(
    solver_cfg, deterministic, deterministic_mode, expected
):
    cfg = NewtonCfg(solver_cfg=solver_cfg, deterministic=deterministic, deterministic_mode=deterministic_mode)

    assert newton_backend._resolve_deterministic_mode(cfg) == expected


@pytest.mark.parametrize("collision_cfg", [None, NewtonCollisionPipelineCfg()], ids=["default", "configured"])
def test_deterministic_collision_pipeline_grows_with_the_solver_contact_capacity(collision_cfg):
    """A deterministic pipeline sizes its sort buffer at construction, so it is rebuilt for the solver's contacts."""
    pipelines, prepared = [], []

    class FakePipeline:
        def __init__(self, **kwargs):
            pipelines.append(kwargs)
            self.rigid_contact_max = kwargs.get("rigid_contact_max") or 1

        def contacts(self):
            return SimpleNamespace(rigid_contact_max=self.rigid_contact_max)

    class RecordingSolver(NewtonSolver):
        @classmethod
        def create_collision_pipeline(cls, cfg, solver, **kwargs):
            return FakePipeline(**kwargs)

        @classmethod
        def prepare_contacts(cls, cfg, solver, contacts, pipeline):
            prepared.append(contacts)

    backend = backend_stub(
        cfg=NewtonCfg(collision_cfg=collision_cfg, solver_cfg=NewtonSolverCfg(class_type=RecordingSolver)),
        model=None,
        solver=SimpleNamespace(get_max_contact_count=lambda: 2),
        collision_pipeline=None,
        device="cpu",
        deterministic_mode=wp.DeterministicMode.GPU_TO_GPU,
    )

    backend._allocate_contacts()

    assert pipelines[1] == {**pipelines[0], "rigid_contact_max": 2}
    assert pipelines[0]["deterministic"] is True
    assert backend.contacts.rigid_contact_max == 2
    assert prepared == [backend.contacts]


def test_collision_cfg_is_rejected_when_the_solver_does_not_use_the_collision_pipeline():
    backend = backend_stub(
        cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True), collision_cfg=NewtonCollisionPipelineCfg()),
        model=None,
        solver=None,
    )

    with pytest.raises(ValueError, match="collision_cfg cannot be set"):
        backend._allocate_contacts()


# ----- The step ----------------------------------------------------------------------------


@pytest.mark.parametrize("single_state", [True, False], ids=["single_state", "double_state"])
def test_state_forces_apply_before_every_substep_and_the_step_ends_in_state_0(single_state):
    events = []
    backend = _recording_backend(events, single_state=single_state, num_substeps=2)
    backend.state_force_callbacks.append(lambda state: events.append(("force", state.name)))

    backend.step()

    if single_state:
        substeps = [("force", "state_0"), ("step", "state_0", "state_0"), ("clear", "state_0")] * 2
    else:
        substeps = [
            ("force", "state_0"),
            ("step", "state_0", "state_1"),
            ("clear", "state_1"),
            ("force", "state_1"),
            ("step", "state_1", "state_0"),
            ("clear", "state_0"),
        ]
    assert events == [("pre_step", "state_0"), *substeps, ("post_step", "state_0"), ("check", False)]


def test_decimation_loop_runs_actuator_callbacks_per_physics_step_and_post_step_callbacks_once():
    events = []
    backend = _recording_backend(events, single_state=False)
    backend.pre_actuator_callbacks.append(lambda: events.append(("pre_actuator",)))
    backend.post_actuator_callbacks.append(lambda: events.append(("post_actuator",)))
    backend.post_step_callbacks.append(lambda: events.append(("post_physics_step",)))
    backend.actuators_active = True
    backend.set_decimation(2)

    backend.step()

    physics_step = [
        ("pre_step", "state_0"),
        ("pre_actuator",),
        ("post_actuator",),
        ("step", "state_0", "state_1"),
        ("assign", "state_0", "state_1"),
        ("clear", "state_0"),
        ("post_step", "state_0"),
    ]
    assert events == physics_step * 2 + [("post_physics_step",), ("check", False)]


@pytest.mark.parametrize(
    ("collision_decimation", "mid_loop_collides"),
    [(0, 0), (2, 3), (7, 1)],
    ids=["disabled", "every_2nd", "only_7th"],
)
def test_collision_decimation_recollides_between_substeps_except_after_the_last(
    collision_decimation, mid_loop_collides
):
    events = []
    backend = _recording_backend(events, num_substeps=8, collision_decimation=collision_decimation)
    backend.collision_pipeline = SimpleNamespace(collide=lambda state, contacts: events.append(("collide",)))
    backend.contacts = object()

    backend.step()

    assert events.count(("collide",)) == 1 + mid_loop_collides


def test_model_changes_notify_the_solver_once_per_step_and_warn_once_when_ignored(caplog):
    events = []
    ignored = ModelFlags.BODY_INERTIAL_PROPERTIES
    backend = _recording_backend(events, ignored_model_changes={ignored: "Ignored model change"})

    with caplog.at_level(logging.WARNING, logger=newton_backend.__name__):
        for flag in (ignored, ModelFlags.JOINT_DOF_PROPERTIES, ignored):
            backend.model_changes.add(flag)
        backend.step()
        backend.step()
        backend.model_changes.add(ignored)
        backend.step()

    notifications = [event for event in events if event[0] == "notify"]
    assert notifications == [("notify", ignored | ModelFlags.JOINT_DOF_PROPERTIES), ("notify", ignored)]
    assert [record.message for record in caplog.records].count("Ignored model change") == 1


@pytest.mark.parametrize("supported", [True, False])
def test_step_runs_eagerly_once_then_replays_a_graph_only_when_the_solver_supports_capture(monkeypatch, supported):
    events, replays = [], []
    monkeypatch.setattr(newton_backend, "capture_graph", lambda device, segment, relaxed=False: segment)
    monkeypatch.setattr(wp, "capture_launch", lambda segment: (replays.append(segment), segment()))
    backend = _recording_backend(events, device="cuda:0", captures=supported)

    captured = [backend.step() for _ in range(2)]

    assert captured == [supported, supported]
    assert len(replays) == int(supported)
    assert events.count(("step", "state_0", "state_0")) == 2
    assert [event for event in events if event[0] == "check"] == [("check", supported)] * 2


# ----- Reset masks -------------------------------------------------------------------------


def test_forward_resolves_flagged_worlds_once_and_not_while_an_outer_graph_records_the_read():
    """Reading published state resolves pending FK once; a raw read during capture leaves it pending."""
    resets, fks = [], []

    class RecordingSolver(NewtonSolver):
        @classmethod
        def reset(cls, cfg, solver, state, worlds):
            resets.append(worlds.numpy().tolist())

        @classmethod
        def eval_fk(cls, cfg, solver, state, worlds, articulations):
            fks.append((worlds.numpy().tolist(), articulations.numpy().tolist()))

    manager = NewtonManager(NewtonCfg(solver_cfg=NewtonSolverCfg(class_type=RecordingSolver)))
    manager._device = "cpu"
    world_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    manager.backend = backend_stub(
        cfg=manager._cfg,
        state_0=object(),
        solver=object(),
        world_mask=world_mask,
        fk_mask=fk_mask,
        kinematics_dirty=True,
        transforms_may_change_on_graph_replay=False,
    )
    scene_data = NewtonSceneDataBackend(manager)

    with wp.ScopedCapture(device="cpu"):
        assert scene_data.state is manager.backend.state_0
    assert fks == resets == []
    assert scene_data.state is scene_data.state is manager.backend.state_0

    assert fks == [([False, True], [True, False])]
    assert resets == [[False, True]]
    assert not world_mask.numpy().any() and not fk_mask.numpy().any()


@pytest.mark.parametrize("selector", ["mask", "ids_int32", "ids_int64", "ids_torch"])
def test_manager_flags_the_selected_worlds_from_a_mask_or_ids(selector):
    """Asset writes select environments by mask or ids; ids become a mask before reaching the backend."""
    manager = NewtonManager(NewtonCfg())
    manager.backend = backend_stub(
        device=wp.get_device("cpu"),
        model=SimpleNamespace(world_count=3),
        world_mask=wp.zeros(4, dtype=wp.bool, device="cpu"),
        fk_mask=wp.zeros(6, dtype=wp.bool, device="cpu"),
        ids_mask=wp.zeros(3, dtype=wp.bool, device="cpu"),
        kinematics_dirty=False,
        transforms_may_change_on_graph_replay=False,
    )
    articulation_ids = wp.array([[0, 1], [2, 3], [4, 5]], dtype=wp.int32, device="cpu")
    if selector == "mask":
        selection = {"env_mask": wp.array([True, False, True], dtype=wp.bool, device="cpu")}
    elif selector == "ids_torch":
        selection = {"env_ids": torch.tensor([2, 0], dtype=torch.int64)}
    else:
        dtype = wp.int32 if selector == "ids_int32" else wp.int64
        selection = {"env_ids": wp.array([2, 0], dtype=dtype, device="cpu")}

    manager.invalidate_fk(articulation_ids=articulation_ids, **selection)

    assert manager.backend.world_mask.numpy().tolist() == [True, False, True, False]
    assert manager.backend.fk_mask.numpy().tolist() == [True, True, False, False, True, True]

    manager.backend.world_mask.zero_()
    manager.invalidate_body_state(**selection)

    assert manager.backend.world_mask.numpy().tolist() == [True, False, True, False]
    assert manager.backend.kinematics_dirty
    # A later selection must not inherit this one's environments.
    assert not manager.backend.ids_mask.numpy().any()


# ----- Independent backends ----------------------------------------------------------------


def _pendulums(physics_cfg: NewtonCfg, decimation: int, device: str, *, captured=False) -> NewtonBackend:
    """A backend of four pendulum worlds, released from a different angle each, advancing ``decimation`` steps."""
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
    physics_cfg.solver_cfg.class_type.prepare_builder(physics_cfg.solver_cfg, builder)
    backend = NewtonBackend(NewtonBackendCfg(physics_cfg=physics_cfg, device=device), builder, dt=0.01)
    backend.init_solver()
    if not captured:
        backend.capture = None
    # Without Newton actuators, an active actuator path lets one step run the whole decimation loop.
    backend.actuators_active = True
    backend.set_decimation(decimation)
    return backend


def _two_solver_configs() -> list[tuple[NewtonCfg, int]]:
    return [
        (NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True), num_substeps=2), 3),
        (NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), num_substeps=4), 2),
    ]


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_backends_with_different_solvers_step_side_by_side(device, captured):
    """Each backend advances only its own model and matches a backend stepped alone."""
    pair = [_pendulums(cfg, decimation, device, captured=captured) for cfg, decimation in _two_solver_configs()]
    alone = [_pendulums(cfg, decimation, device) for cfg, decimation in _two_solver_configs()]
    start = [backend.state_0.joint_q.numpy().copy() for backend in pair]

    for _ in range(5):
        pair[0].step()
    np.testing.assert_array_equal(pair[1].state_0.joint_q.numpy(), start[1])
    for _ in range(5):
        pair[1].step()
        alone[0].step()
        alone[1].step()

    for backend, reference, initial in zip(pair, alone, start, strict=True):
        assert (backend.graph is not None) is captured
        assert np.all(np.abs(backend.state_0.joint_q.numpy() - initial) > 1e-3)
        np.testing.assert_allclose(backend.state_0.joint_q.numpy(), reference.state_0.joint_q.numpy(), atol=1e-5)

    # Invalidating one world of one backend recomputes only that world's body state.
    body_q = [backend.state_0.body_q.numpy().copy() for backend in pair]
    pair[0].state_0.joint_q.zero_()
    pair[0].invalidate_fk(
        wp.array([True, False, False, False], dtype=wp.bool, device=device),
        wp.array(np.arange(4, dtype=np.int32).reshape(4, 1), dtype=wp.int32, device=device),
    )
    assert not pair[1].world_mask.numpy().any()
    for backend in pair:
        backend.forward()
    assert np.any(pair[0].state_0.body_q.numpy() != body_q[0], axis=1).tolist() == [True, False, False, False]
    np.testing.assert_array_equal(pair[1].state_0.body_q.numpy(), body_q[1])


# ----- Queries -----------------------------------------------------------------------------


@pytest.mark.parametrize("cloth", [False, True])
@pytest.mark.parametrize("device", test_devices())
def test_queries_share_one_bvh_refit_per_publication(monkeypatch, cloth, device):
    """Two renders of one publication refit once; new transforms or geometry refit again."""
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    cfg = NewtonBackendCfg(physics_cfg=object(), device=device)
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
    builder.add_shape_sphere(builder.add_body(label="/Object"))
    for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        builder.add_particle(pos=wp.vec3(position), vel=wp.vec3(), mass=1.0)
    if cloth:
        builder.add_triangle(0, 1, 2)
    backend = sim.get_or_create_backend(cfg)
    backend.geometry_offsets = {"/Cloth": 0}
    transforms, points = SceneDataFormat.Transform(), SceneDataFormat.Points()
    transforms.transforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    points.points = wp.array(np.tile([1.0, 2.0, 3.0], (3, 1)), dtype=wp.vec3, device=device)
    publication = SimpleNamespace(
        transforms=transforms,
        transform_paths=["/Object"],
        transforms_timestamp=0,
        geometry_timestamp=0,
        get_transforms=lambda output_format: transforms,
        get_geometry_batches=lambda output_format: [(points, {"/Cloth": (0, 3)})],
    )
    provider = SceneDataProvider(publication)
    renderer = object.__new__(NewtonWarpRenderer)
    renderer.backend = backend
    renderer._scene_data_provider = provider
    renderer._transform_mapping = provider.create_mapping(list(backend.model.body_label))
    # Triangle rendering performs host work and must stay eager even when capture is enabled.
    renderer.cfg = SimpleNamespace(use_cuda_graph=cloth)
    refits, observed = [], []
    refit = backend.model.bvh_refit_shapes
    monkeypatch.setattr(backend.model, "bvh_refit_shapes", lambda state: (refits.append(state), refit(state)))

    def read(_data):
        assert backend.state_0.body_q is transforms.transforms
        np.testing.assert_array_equal(backend.state_0.particle_q.numpy(), points.points.numpy())
        observed.append(backend.state_0.body_q.numpy().copy())

    monkeypatch.setattr(renderer, "_launch_render", read)
    render_data = SimpleNamespace(graph=None, ppisp_pipeline=None)
    renderer.render(render_data)
    renderer.render(render_data)
    assert render_data.graph is None
    assert len(refits) == 1
    transforms.transforms = wp.array(
        [wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_identity())], dtype=wp.transform, device=device
    )
    points.points.assign(np.tile(np.asarray([7.0, 8.0, 9.0], dtype=np.float32), (3, 1)))
    publication.transforms_timestamp += 1
    publication.geometry_timestamp += 1
    renderer.render(render_data)
    assert len(refits) == 2
    np.testing.assert_array_equal(observed[-1][0, :3], [4.0, 5.0, 6.0])
    for expected, attribute in enumerate(("geometry_timestamp", "transforms_timestamp"), start=3):
        setattr(publication, attribute, getattr(publication, attribute) + 1)
        renderer.render(render_data)
        renderer.render(render_data)
        assert len(refits) == expected
    backend.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_bvh_refit_captures_a_model_without_rigid_bodies(device):
    """A particle-only model has no ``body_q``; its refit is still captured and replayed once per publication."""
    builder = newton.ModelBuilder()
    for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        builder.add_particle(pos=wp.vec3(position), vel=wp.vec3(), mass=1.0)
    backend = NewtonBackend(NewtonBackendCfg(physics_cfg=object(), device=device), builder)
    assert backend.state_0.body_q is None

    backend.refit_bvh(1)
    backend.refit_bvh(2)

    assert backend.bvh_refit.data[0] == backend.state_pointers()


# ----- Captured steps in a simulation ------------------------------------------------------


@wp.kernel
def _count_steps(counter: wp.array(dtype=wp.int32)):
    counter[0] += 1


@pytest.mark.parametrize(
    ("solver_cfg", "relaxed_capture", "num_substeps", "decimation"),
    [
        pytest.param(MJWarpSolverCfg(use_mujoco_contacts=True), True, 1, 1, id="single_state_relaxed"),
        pytest.param(KaminoPADMMSolverCfg(), False, 2, 3, id="double_state_substeps_decimation"),
    ],
)
def test_captured_step_holds_staged_forces_and_recaptures_after_decimation_changes(
    monkeypatch, solver_cfg, relaxed_capture, num_substeps, decimation
):
    """The first step runs eagerly, later steps replay; staged forces hold for every substep and callbacks run once."""
    sim_cfg = SimulationCfg(
        dt=0.005,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=num_substeps, use_cuda_graph=True),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        builder.add_body(mass=1.0)
        sim.reset()
        if relaxed_capture:
            monkeypatch.setattr(newton_manager_module, "has_kit", lambda: True)
            monkeypatch.setattr(sim, "_has_offscreen_render", True)
        manager = sim.backend_manager
        manager.activate_newton_actuator_path()
        counter = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        manager.register_post_step_callback(lambda: wp.launch(_count_steps, 1, inputs=[counter]))
        wrench = wp.array([[0.0, 0.0, 9.81, 0.0, 0.0, 0.0]], dtype=wp.spatial_vector, device="cuda:0")
        for step in range(3):
            if step != 1:
                manager.set_decimation(decimation)
            manager.backend.state_0.body_f.assign(wrench)
            sim.step(render=False)
            assert manager.backend.graph is not None
            np.testing.assert_allclose(manager.backend.state_0.joint_qd.numpy(), 0.0, atol=1e-6)
            assert counter.numpy()[0] == step + 1


def test_stateful_actuators_replay_like_eager_steps_across_decimation_changes(monkeypatch):
    """PID and delay history survive replay, recapture, and odd and even decimation."""
    trajectories = []
    for use_graph in (False, True):
        sim_cfg = SimulationCfg(
            dt=0.005,
            device="cuda:0",
            gravity=(0.0, 0.0, 0.0),
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True), num_substeps=1, use_cuda_graph=use_graph
            ),
        )
        with build_simulation_context(sim_cfg=sim_cfg) as sim:
            builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
            builder.begin_world()
            body = builder.add_link(mass=1.0)
            joint = builder.add_joint_prismatic(parent=-1, child=body, axis=(1.0, 0.0, 0.0))
            builder.add_articulation([joint])
            builder.add_actuator(DrivePID, index=0, kp=1.0, kd=0.0, ki=0.5, delay_steps=2)
            builder.end_world()
            sim.reset()
            manager = sim.backend_manager
            manager.activate_newton_actuator_path()
            samples = []
            for decimation in (1, 2, 3):
                manager.set_decimation(decimation)
                for target in (1.0, -2.0, 3.0):
                    manager.backend.control.joint_target_q.fill_(target)
                    sim.step(render=False)
                    state, control = manager.backend.state_0, manager.backend.control
                    samples.append(
                        np.concatenate((state.joint_q.numpy(), state.joint_qd.numpy(), control.joint_f.numpy()))
                    )
            trajectories.append(np.asarray(samples))
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1e-6, rtol=1e-5)
