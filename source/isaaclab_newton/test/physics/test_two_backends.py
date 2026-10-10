# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backends driven through the functional core without the Newton manager.

Covers what an outer runner such as a graph-captured MDP needs from physics: several backends stepped side by side,
steps recorded into a caller-owned CUDA graph, world-masked reset commits, and heterogeneous worlds.
"""

from __future__ import annotations

from functools import partial

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonBackend,
    NewtonCfg,
    NewtonManager,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics import newton_backend as nb

DEVICE = "cuda:0"
DT = 0.01


def _pendulums(
    physics_cfg: NewtonCfg,
    steps: int,
    *,
    captured: bool = False,
    num_worlds: int = 4,
    links_per_world: list[int] | None = None,
) -> NewtonBackend:
    """Build a backend of one revolute pendulum chain per world, released from a different angle in each world.

    Each :func:`nb.step` advances ``steps`` physics steps, replayed from CUDA graphs when ``captured``.
    ``links_per_world`` gives each world its own chain length; by default every world holds one link.
    """
    builder = _pendulum_builder(physics_cfg, num_worlds, links_per_world)
    physics_cfg.solver_cfg.class_type.prepare_solver_builder(builder, physics_cfg.solver_cfg)
    model = builder.finalize(device=DEVICE)
    backend = NewtonBackend(model, physics_cfg, dt=DT)
    nb.init_solver(backend)
    backend.steps_per_call = steps
    if not captured:
        backend.capture = None
    return backend


def _pendulum_builder(physics_cfg, num_worlds=4, links_per_world=None):
    """Construct the same pendulum worlds for standalone managers or the functional backend."""
    builder = physics_cfg.solver_cfg.class_type.create_builder(physics_cfg=physics_cfg)
    for world, num_links in enumerate(links_per_world or [1] * num_worlds):
        builder.begin_world()
        parent, joints = -1, []
        for _ in range(num_links):
            link = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
            joint = builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 if parent < 0 else -0.5), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            )
            joints.append(joint)
            builder.joint_q[-1] = 0.2 + 0.2 * world
            parent = link
        builder.add_articulation(joints)
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


@pytest.mark.parametrize("owner", ["warp", "torch"])
def test_backends_with_different_solvers_record_into_one_graph(owner):
    """Both backends' steps replay from one caller-owned CUDA graph and match eager stepping.

    The steps are recorded right after :func:`nb.prepare`, without a warm-up step, into a Warp capture or into a Torch
    capture with Warp launching on Torch's capture stream.
    """
    pair = [_pendulums(cfg, steps) for cfg, steps in _configs()]
    eager = [_pendulums(cfg, steps) for cfg, steps in _configs()]
    for backend in pair:
        nb.prepare(backend)

    if owner == "warp":
        with wp.ScopedCapture(device=DEVICE) as capture:
            for backend in pair:
                nb.record_step(backend)
        replay = partial(wp.capture_launch, capture.graph)
    else:
        # Torch owns the capture; Warp joins it as an external capture so that its allocations are graph safe.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream = wp.stream_from_torch(torch.cuda.current_stream())
            with wp.ScopedStream(stream, sync_enter=False):
                wp.capture_begin(stream=stream, external=True)
                try:
                    for backend in pair:
                        nb.record_step(backend)
                finally:
                    wp.capture_end(stream=stream)
        replay = graph.replay

    for _ in range(10):
        replay()
        for backend in eager:
            nb.step(backend)
    torch.cuda.synchronize()
    for backend, reference in zip(pair, eager, strict=True):
        np.testing.assert_allclose(_angles(backend), _angles(reference), atol=1e-5)


@wp.kernel
def _set_masked_joint_q(worlds: wp.array(dtype=wp.bool), value: float, joint_q: wp.array(dtype=float)):
    world = wp.tid()
    if worlds[world]:
        joint_q[world] = value


def test_world_masked_reset_commit_follows_the_mask_of_each_replay():
    """A recorded reset commit updates exactly the worlds masked at replay time, and leaves the others untouched."""
    cfg, steps = _configs()[0]
    backend = _pendulums(cfg, steps)
    num_worlds = backend.model.world_count
    worlds = wp.zeros(num_worlds, dtype=wp.bool, device=DEVICE)
    with wp.ScopedCapture(device=DEVICE) as capture:
        wp.launch(_set_masked_joint_q, num_worlds, [worlds, 0.0, backend.state_0.joint_q], device=DEVICE)
        nb.invalidate_worlds(backend, worlds)
        nb.forward(backend, force=True)

    rng = np.random.default_rng(0)
    for selected in ([True, False, True, False], [False, True, False, False], [False] * 4):
        before = backend.state_0.body_q.numpy().copy()
        # uncommitted joint coordinates in every world: only the masked worlds may turn them into body poses
        backend.state_0.joint_q.assign(rng.uniform(-1.0, 1.0, num_worlds).astype(np.float32))
        worlds.assign(np.array(selected))
        wp.capture_launch(capture.graph)
        # reference body poses from the updated joint coordinates of every world
        expected = backend.model.state()
        newton.eval_fk(backend.model, backend.state_0.joint_q, backend.state_0.joint_qd, expected)
        after = backend.state_0.body_q.numpy()
        for world, reset in enumerate(selected):
            np.testing.assert_allclose(
                after[world], expected.body_q.numpy()[world] if reset else before[world], atol=1e-6
            )
        assert not backend.fk_mask.numpy().any() and not backend.world_mask.numpy().any()


@pytest.mark.parametrize(
    "solver_cfg",
    [
        MJWarpSolverCfg(use_mujoco_contacts=True),
        FeatherstoneSolverCfg(),
        XPBDSolverCfg(),
        VBDSolverCfg(),
        KaminoPADMMSolverCfg(),
    ],
    ids=lambda cfg: type(cfg).__name__,
)
def test_heterogeneous_worlds_match_solver_capability(solver_cfg):
    """Solvers that declare heterogeneous-world support step worlds with different chains; the others refuse them."""
    cfg = NewtonCfg(solver_cfg=solver_cfg, num_substeps=4)
    links = [1, 2, 1, 3]
    if not cfg.solver_cfg.class_type.supports_heterogeneous_worlds:
        with pytest.raises(ValueError, match="homogeneous"):
            _pendulums(cfg, 1, links_per_world=links)
        return
    eager = _pendulums(cfg, 1, links_per_world=links)
    captured = _pendulums(cfg, 1, captured=True, links_per_world=links)
    # maximal-coordinate solvers advance body poses only, so compare those
    start = eager.state_0.body_q.numpy().copy()
    for _ in range(5):
        nb.step(eager)
        nb.step(captured)
    moved = eager.state_0.body_q.numpy()
    assert np.all(np.isfinite(moved)) and np.any(np.abs(moved - start) > 1e-4)
    np.testing.assert_array_equal(captured.state_0.body_q.numpy(), moved)


def test_record_step_requires_a_prepared_graph():
    """Recording refuses a step that was not prepared, so a caller's capture never allocates."""
    cfg, steps = _configs()[1]
    backend = _pendulums(cfg, steps)
    with pytest.raises(RuntimeError, match="Prepare the Newton step"):
        nb.record_step(backend)


def test_independent_managers_and_custom_context_injection():
    """Managers isolate construction, callbacks and solver state; a custom instance plugs into Isaac Lab."""
    from isaaclab.sim import SimulationCfg, SimulationContext

    class CustomManager(NewtonManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = 0

        def step(self):
            super().step()
            self.steps += 1

    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True), class_type=CustomManager)
    first = cfg.class_type(_pendulum_builder(cfg), cfg, dt=DT, device=DEVICE)
    second = NewtonManager(_pendulum_builder(cfg), cfg, dt=DT, device=DEVICE)
    for manager in (first, second):
        manager.register_site(None, wp.transform_identity())
        manager.reset()
    try:
        assert first.backend.model.shape_count == second.backend.model.shape_count == 1
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

    injected = CustomManager(_pendulum_builder(cfg))
    sim = SimulationContext(SimulationCfg(physics=cfg, device=DEVICE, dt=DT), physics_manager=injected)
    try:
        sim.reset()
        sim.step(render=False)
        assert sim.physics_manager is injected
        assert injected.steps == 1
        assert isinstance(injected.backend.solver, newton.solvers.SolverMuJoCo)
    finally:
        SimulationContext.clear_instance()
    assert injected.backend is None
    assert injected._sim is None


@wp.kernel
def _count_refreshes(count: wp.array(dtype=wp.int32)):
    count[0] += 1


def test_masked_property_transaction_skips_empty_replays_and_combines_flags(monkeypatch):
    """An empty replay skips solver constants; selected edits rebuild them once with all categories."""
    backend = _pendulums(_configs()[0][0], 1)
    mask = wp.zeros(4, dtype=wp.bool, device=DEVICE)
    refreshes = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    flags_seen = []
    original = backend.solver.notify_model_changed

    def notify(flags):
        flags_seen.append(flags)
        wp.launch(_count_refreshes, 1, [refreshes], device=DEVICE)
        original(flags)

    monkeypatch.setattr(backend.solver, "notify_model_changed", notify)

    def edits():
        nb.mark_model_changed(backend, newton.ModelFlags.BODY_INERTIAL_PROPERTIES, mask)
        nb.mark_model_changed(backend, newton.ModelFlags.JOINT_DOF_PROPERTIES, mask)
        nb.notify_model_changes(backend)

    graph = nb.capture_graph(DEVICE, edits)
    replay = graph.launch
    replay()
    assert refreshes.numpy()[0] == 0
    mask.assign(np.array([False, True, False, False]))
    replay()
    assert refreshes.numpy()[0] == 1
    mask.zero_()
    replay()
    assert refreshes.numpy()[0] == 1
    assert flags_seen == [newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES]
    backend.close()
