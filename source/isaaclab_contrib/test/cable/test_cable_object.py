# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cable asset, registry, and replicate-hook plumbing."""

import math

import pytest
import warp as wp
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

from isaaclab_contrib.cable.cable_object import CableRegistryEntry


def test_install_cable_builder_hooks_is_idempotent(monkeypatch):
    """Repeated install must not duplicate registrations on _per_world_builder_hooks."""
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    from isaaclab_contrib.cable.cable_object import (
        add_registered_cables_to_builder,
        install_cable_builder_hooks,
    )

    # Reset state so the test is self-contained.
    monkeypatch.setattr(SimulationManager, "_per_world_builder_hooks", [], raising=False)
    monkeypatch.delattr(SimulationManager, "_cable_registry", raising=False)

    install_cable_builder_hooks()
    install_cable_builder_hooks()
    install_cable_builder_hooks()

    assert SimulationManager._cable_registry == []
    matches = [h for h in SimulationManager._per_world_builder_hooks if h is add_registered_cables_to_builder]
    assert len(matches) == 1, "install_cable_builder_hooks must be idempotent"


def test_add_registered_cables_iterates_registry(monkeypatch):
    """The loop function dispatches to add_cable_entry_to_builder per registry entry."""
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    from isaaclab_contrib.cable.cable_object import add_registered_cables_to_builder

    monkeypatch.setattr(SimulationManager, "_per_world_builder_hooks", [], raising=False)

    calls = []

    def _fake_entry_hook(builder, entry, env_idx, env_pos, env_rot, cable_idx=0):
        calls.append((entry.prim_path, env_idx, cable_idx))

    monkeypatch.setattr(
        "isaaclab_contrib.cable.cable_object.add_cable_entry_to_builder",
        _fake_entry_hook,
    )
    entries = [
        CableRegistryEntry(
            prim_path="/World/cable_a",
            node_positions=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0)],
            edges=[(0, 1)],
            radius=0.005,
        ),
        CableRegistryEntry(
            prim_path="/World/cable_b",
            node_positions=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0)],
            edges=[(0, 1)],
            radius=0.005,
        ),
    ]
    monkeypatch.setattr(SimulationManager, "_cable_registry", entries, raising=False)

    add_registered_cables_to_builder(builder=None, world_idx=3, env_position=[0, 0, 0], env_rotation=[0, 0, 0, 1])

    assert calls == [("/World/cable_a", 3, 0), ("/World/cable_b", 3, 1)]


class _FakeBuilder:
    """Records the arguments passed to add_rod_graph for assertion."""

    def __init__(self):
        self.calls = []
        self.body_count = 0

    def add_rod_graph(self, **kwargs):
        self.calls.append(kwargs)
        self.body_count += len(kwargs.get("edges", []))


@pytest.mark.parametrize(
    "env_rotation, env_position, init_pos, init_rot, expected_np0, expected_np1",
    [
        # Identity case (was test 4): verifies field-forwarding + translation composition.
        (
            [0.0, 0.0, 0.0, 1.0],  # env identity
            [1.0, 0.0, 0.0],  # env_t = (1, 0, 0)
            (0.0, 0.0, 1.0),  # init_t = (0, 0, 1)
            (0.0, 0.0, 0.0, 1.0),  # init identity
            (1.0, 0.0, 1.0),  # node[0] world = env_t + init_t = (1, 0, 1)
            (1.1, 0.0, 1.0),  # node[1] world = (1.1, 0, 1)
        ),
        # 90° CCW about Z (was test 5): verifies composed rotation.
        (
            [0.0, 0.0, math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0],
            [0.0, 0.0, 0.0],
            (0.0, 1.0, 0.0),  # init_t = (0, 1, 0)
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),  # R_z(90°)·(0, 1, 0) = (-1, 0, 0)
            (-1.0, 0.1, 0.0),  # node[1] = (-1, 0, 0) + R_z(90°)·(0.1, 0, 0) = (-1, 0.1, 0)
        ),
    ],
    ids=["identity", "env_rotation_z90"],
)
def test_add_cable_entry_to_builder(env_rotation, env_position, init_pos, init_rot, expected_np0, expected_np1):
    """add_cable_entry_to_builder transforms positions correctly and forwards
    all material/geometry params to add_rod_graph."""
    from isaaclab_contrib.cable.cable_object import add_cable_entry_to_builder

    entry = CableRegistryEntry(
        prim_path="/World/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1)],
        radius=0.005,
        init_pos=init_pos,
        init_rot=init_rot,
        stretch_stiffness=2.0e9,
        bend_stiffness=1.0e-3,
        stretch_damping=0.0,
        bend_damping=1.0e-4,
        density=1200.0,
    )
    builder = _FakeBuilder()
    add_cable_entry_to_builder(builder, entry, env_idx=0, env_position=env_position, env_rotation=env_rotation)

    assert len(builder.calls) == 1
    call = builder.calls[0]

    np0 = call["node_positions"][0]
    np1 = call["node_positions"][1]
    assert float(np0[0]) == pytest.approx(expected_np0[0], abs=1e-5)
    assert float(np0[1]) == pytest.approx(expected_np0[1], abs=1e-5)
    assert float(np0[2]) == pytest.approx(expected_np0[2], abs=1e-5)
    assert float(np1[0]) == pytest.approx(expected_np1[0], abs=1e-5)
    assert float(np1[1]) == pytest.approx(expected_np1[1], abs=1e-5)
    assert float(np1[2]) == pytest.approx(expected_np1[2], abs=1e-5)

    # Field forwarding (only need to assert once; same across all rows).
    assert call["edges"] == [(0, 1)]
    assert call["radius"] == pytest.approx(0.005)
    assert call["stretch_stiffness"] == pytest.approx(2.0e9)
    assert call["bend_stiffness"] == pytest.approx(1.0e-3)
    assert call["bend_damping"] == pytest.approx(1.0e-4)
    assert call["label"] == "/World/Cable/cable"
    assert float(call["cfg"].density) == pytest.approx(1200.0)


def test_add_cable_entry_populates_body_offsets_and_last_edge_length():
    """``add_cable_entry_to_builder`` records per-env body offsets and the last edge length."""
    from isaaclab_contrib.cable.cable_object import add_cable_entry_to_builder

    class _BodyCountingBuilder:
        def __init__(self):
            self.body_count = 0

        def add_rod_graph(self, *, edges, **_kwargs):
            self.body_count += len(edges)

    entry = CableRegistryEntry(
        prim_path="/World/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.2, 0.0, 0.0), wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.9, 0.0, 0.0)],
        edges=[(0, 1), (1, 2), (2, 3)],
        radius=0.005,
    )
    builder = _BodyCountingBuilder()
    builder.body_count = 7
    add_cable_entry_to_builder(builder, entry, env_idx=0, env_position=[0, 0, 0], env_rotation=[0, 0, 0, 1])
    builder.body_count += 5
    add_cable_entry_to_builder(builder, entry, env_idx=1, env_position=[1, 0, 0], env_rotation=[0, 0, 0, 1])

    assert entry.body_offsets == [7, 15]
    assert entry.last_edge_length == pytest.approx(0.4)


def test_cable_object_cfg_defaults():
    """CableObjectCfg overrides actuators and articulation_root_prim_path."""
    import isaaclab.sim as sim_utils

    from isaaclab_contrib.cable import CableObjectCfg

    cfg = CableObjectCfg(
        prim_path="/World/Cable",
        spawn=sim_utils.CableCfg(
            positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
            width=0.01,
            physics_material=NewtonCableMaterialCfg(),
        ),
    )
    assert cfg.articulation_root_prim_path == "/cable_articulation"
    assert cfg.actuators == {}


@pytest.mark.parametrize(
    "setup_registry, spawn, expected_exc, expected_match",
    [
        # spawn=None → ValueError mentioning "CableCfg"
        (True, None, ValueError, "CableCfg"),
        # registry not installed → RuntimeError mentioning the VBD solver requirement
        (False, "valid", RuntimeError, "VBD"),
    ],
    ids=["spawn_none", "hooks_not_installed"],
)
def test_cable_object_init_failure_paths(monkeypatch, setup_registry, spawn, expected_exc, expected_match):
    """CableObject.__init__ raises clear errors on invalid cfg or missing setup."""
    from isaaclab_newton.assets.articulation.articulation import Articulation
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    import isaaclab.sim as sim_utils

    from isaaclab_contrib.cable import CableObject, CableObjectCfg

    if setup_registry:
        monkeypatch.setattr(SimulationManager, "_cable_registry", [], raising=False)
    else:
        monkeypatch.delattr(SimulationManager, "_cable_registry", raising=False)
    monkeypatch.setattr(Articulation, "__init__", lambda self, cfg: setattr(self, "cfg", cfg))

    # "valid" sentinel → construct a real CableCfg
    if spawn == "valid":
        spawn_value = sim_utils.CableCfg(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            width=0.01,
            physics_material=NewtonCableMaterialCfg(),
        )
    else:
        spawn_value = spawn

    cfg = CableObjectCfg(prim_path="/World/Cable", spawn=spawn_value)
    with pytest.raises(expected_exc, match=expected_match):
        CableObject(cfg)


def test_cable_replicate_body_count():
    """Spawn 2 cables in env_0, replicate to 4 envs, verify total body count.

    Each cable has 3 control points → 2 segments per cable.
    Total cable bodies in builder = 4 envs × 2 cables × 2 segments = 16.
    """
    from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg as _NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils.configclass import configclass

    from isaaclab_contrib.cable import CableObjectCfg
    from isaaclab_contrib.cable.cable_object import install_cable_builder_hooks

    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        width=0.01,
        physics_material=_NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 4
        env_spacing: float = 1.0
        cable_a: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/CableA", spawn=cable_spawn)
        cable_b: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/CableB", spawn=cable_spawn)

    # Cables need install_cable_builder_hooks called once before scene init.
    # This mirrors how NewtonVBDManager.initialize() calls
    # install_deformable_builder_hooks() before the deformable scene is set up.
    install_cable_builder_hooks()

    newton_sim_cfg = SimulationCfg(
        physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg()),
    )

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg())
        sim.reset()  # triggers newton_physics_replicate, materializing cable bodies

        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()

        # Newton labels each cable body as "{prim_path}_cable_edge_body_{i}" before
        # label renaming and "{env_dest}/cable_edge_body_{i}" after.
        # Both forms contain the substring "cable_edge_body_".
        cable_body_count = sum(1 for label in model.body_label if "cable_edge_body_" in label)
        assert cable_body_count == 16, f"expected 16 cable bodies, got {cable_body_count}"


def test_forward_preserves_cable_body_q():
    """Regression test for the eval_fk cable patch (commit fd115a500f6).

    Newton's ``eval_fk`` has no case for :attr:`newton.JointType.CABLE`, so
    without the patch any FK pass would collapse cable rod segments onto their
    parent anchors. :meth:`NewtonVBDManager.forward` builds an articulation
    mask in :meth:`start_simulation` that excludes cable articulations.

    To verify the test fails without the fix, force the mask to ``None`` after
    ``start_simulation`` and observe that the new defensive check in
    :meth:`forward` raises ``RuntimeError``; previously, the unmasked
    ``eval_fk`` call would silently mutate ``body_q``.
    """
    import numpy as np
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg as _NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils.configclass import configclass

    from isaaclab_contrib.cable import CableObjectCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        width=0.01,
        physics_material=_NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/Cable", spawn=cable_spawn)

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg()))

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg())
        sim.reset()  # triggers replicate + start_simulation + _build_non_cable_articulation_mask

        # The mask must have been built since cables are registered.
        assert NewtonVBDManager._non_cable_articulation_mask is not None, (
            "Expected _non_cable_articulation_mask to be built when cables are registered."
        )

        body_q_before = NewtonVBDManager._state_0.body_q.numpy().copy()

        # forward() is what Kit-style visualizers invoke each render. With the
        # patch, cable articulations are excluded from the FK pass and body_q
        # is bit-identical. Without the patch, JointType.CABLE relative
        # transforms fall through to identity, snapping each rod segment onto
        # its parent anchor.
        NewtonVBDManager.forward()

        body_q_after = NewtonVBDManager._state_0.body_q.numpy()
        np.testing.assert_array_equal(
            body_q_after,
            body_q_before,
            err_msg="forward() altered body_q — cable mask did not exclude cable articulations.",
        )


def test_start_simulation_preserves_curved_cable_body_q():
    """Regression test for the cable body_q restoration after start_simulation's eval_fk.

    :meth:`NewtonManager.start_simulation` ends with an unmasked ``eval_fk`` to seed
    ``state_0.body_q`` from joint coordinates. Newton's ``eval_fk`` has no case for
    :attr:`newton.JointType.CABLE`, so cable joints fall through to identity and each
    child capsule collapses onto its parent joint anchor — rotating curved cables onto
    the root segment's local +Z axis.

    For a *straight* cable the corruption is invisible (eval_fk's identity output matches
    the layout produced by ``add_rod_graph``), so a non-collinear node layout is required
    to expose the bug. :meth:`NewtonVBDManager._restore_cable_body_q` undoes the corruption
    by copying ``model.body_q`` (untouched by ``eval_fk``) back into ``state_0.body_q`` for
    cable bodies.
    """
    import numpy as np
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg as _NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils.configclass import configclass

    from isaaclab_contrib.cable import CableObjectCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    # Curved cable: three nodes whose edges (0->1 along +x, 1->2 along +y) point in
    # different directions, so adjacent capsule orientations differ. eval_fk's identity
    # output would collapse body[1] onto body[0]'s +Z axis (still pointing +x), but the
    # rest pose has body[1] rotated to align +Z with +y.
    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0)],
        width=0.01,
        physics_material=_NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/Cable", spawn=cable_spawn)

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg()))

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg())
        sim.reset()  # triggers start_simulation -> unmasked eval_fk -> _restore_cable_body_q

        # ``model.body_q`` holds the rest pose produced by ``add_rod_graph`` and is never
        # written by ``eval_fk``. With the restoration in place, ``state_0.body_q`` for cable
        # bodies must match ``model.body_q`` bit-for-bit. Without the fix, the second cable
        # body's quaternion differs (eval_fk reuses the root segment's orientation).
        assert NewtonVBDManager._cable_registry, "Cable registry empty — replicate hook did not run."

        body_q_state = NewtonVBDManager._state_0.body_q.numpy()
        body_q_model = NewtonVBDManager._model.body_q.numpy()

        cable_body_indices: list[int] = []
        for entry in NewtonVBDManager._cable_registry:
            for body_offset in entry.body_offsets:
                cable_body_indices.extend(range(body_offset, body_offset + len(entry.edges)))

        np.testing.assert_allclose(
            body_q_state[cable_body_indices],
            body_q_model[cable_body_indices],
            err_msg=(
                "Cable body_q in state_0 does not match model.body_q after start_simulation."
                " The unmasked eval_fk corrupted cable bodies and _restore_cable_body_q did not"
                " restore them."
            ),
        )


def test_cable_object_reset_restores_body_state():
    """``CableObject.reset()`` snaps the cable's body slice back to the rest pose.

    Steps the sim to drift the cable away from its spawn pose, calls
    :meth:`CableObject.reset`, and verifies that:

    1. ``state.body_q`` matches ``model.body_q`` for the cable's bodies.
    2. ``state.body_qd`` is zero for the cable's bodies.
    3. ``solver.body_q_prev`` is refreshed to the rest pose (otherwise AVBD's
       implicit velocity ``(body_q - body_q_prev) / dt`` would produce
       hundreds of m/s on the next step).
    4. ``solver.body_inertia_q`` is zero (matches solver-init default).
    5. One more ``sim.step()`` keeps ``|body_qd|`` bounded (regression for the
       ~700 m/s spurious-velocity bug).
    """
    import numpy as np
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg as _NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils.configclass import configclass

    from isaaclab_contrib.cable import CableObjectCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0)],
        width=0.01,
        physics_material=_NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cable",
            spawn=cable_spawn,
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=10), num_substeps=4), dt=0.01)

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_SceneCfg())
        sim.reset()

        cable = scene["cable"]
        entry = cable._registry_entry
        body_indices = list(range(entry.body_offsets[0], entry.body_offsets[0] + len(entry.edges)))

        body_q_model = NewtonVBDManager._model.body_q.numpy()[body_indices]

        # Step under gravity so the cable's body slice drifts away from the rest pose.
        for _ in range(20):
            sim.step()
        body_q_drifted = NewtonVBDManager._state_0.body_q.numpy()[body_indices]
        assert not np.allclose(body_q_drifted, body_q_model, atol=1e-4), (
            "Sim did not advance: cable body_q matches model.body_q without stepping."
        )

        cable.reset()

        body_q_after = NewtonVBDManager._state_0.body_q.numpy()[body_indices]
        body_qd_after = NewtonVBDManager._state_0.body_qd.numpy()[body_indices]
        body_q_prev_after = NewtonVBDManager._solver.body_q_prev.numpy()[body_indices]
        body_inertia_q_after = NewtonVBDManager._solver.body_inertia_q.numpy()[body_indices]

        np.testing.assert_allclose(
            body_q_after,
            body_q_model,
            err_msg="state.body_q was not restored to model.body_q after CableObject.reset().",
        )
        np.testing.assert_array_equal(
            body_qd_after,
            np.zeros_like(body_qd_after),
            err_msg="state.body_qd was not zeroed after CableObject.reset().",
        )
        np.testing.assert_allclose(
            body_q_prev_after,
            body_q_model,
            err_msg="solver.body_q_prev was not refreshed to model.body_q after CableObject.reset().",
        )
        np.testing.assert_array_equal(
            body_inertia_q_after,
            np.zeros_like(body_inertia_q_after),
            err_msg="solver.body_inertia_q was not zeroed after CableObject.reset().",
        )

        # One step of free-fall should add at most ~g*dt = ~0.1 m/s. A failure
        # here (e.g. ~700 m/s) indicates AVBD picked up stale solver-side state.
        sim.step()
        max_speed = float(np.abs(NewtonVBDManager._state_0.body_qd.numpy()[body_indices]).max())
        assert max_speed < 1.0, f"body_qd exploded after first post-reset step: |body_qd|_max={max_speed}"


def test_cable_object_reset_partial_envs_and_body_q_prev():
    """``CableObject.reset(env_ids=...)`` restores only the requested envs and
    refreshes ``solver.body_q_prev`` (not just ``state.body_q``).

    Sets up 4 envs, then writes a known non-rest perturbation into
    ``state.body_q`` *and* ``solver.body_q_prev`` for every cable body across
    all envs. Calls ``reset(env_ids=[0, 2])`` and asserts:

    - Envs 0 and 2: both buffers restored bit-for-bit to ``model.body_q``.
    - Envs 1 and 3: both buffers retain the perturbed values.

    This catches three regressions the existing
    :func:`test_cable_object_reset_restores_body_state` would miss:

    1. Off-by-one indexing into ``body_offsets`` when ``env_ids`` selects a
       subset of envs — exercising only ``num_envs=1`` cannot detect this.
    2. A reset that ignores ``env_ids`` and snaps all envs — exposed by the
       envs 1/3 retention assertions.
    3. Dropping the ``wp.copy(dest=solver.body_q_prev, ...)`` line — exposed
       by tight equality against a known perturbation, with no dependency on
       sim drift dynamics. Verified by commenting out that line and observing
       the envs 0/2 ``body_q_prev`` assertion fail.
    """
    import numpy as np
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg as _NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils.configclass import configclass

    from isaaclab_contrib.cable import CableObjectCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0)],
        width=0.01,
        physics_material=_NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 4
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cable",
            spawn=cable_spawn,
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg()), dt=0.01)

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_SceneCfg())
        sim.reset()

        cable = scene["cable"]
        entry = cable._registry_entry
        n = len(entry.edges)
        body_offsets = list(entry.body_offsets)
        assert len(body_offsets) == 4, f"expected 4 envs, got {len(body_offsets)}"

        state = NewtonVBDManager._state_0
        solver = NewtonVBDManager._solver

        # Perturb every cable body's z position by a value far outside any
        # plausible solver noise floor, so tight equality is unambiguous.
        all_cable_indices = [offset + i for offset in body_offsets for i in range(n)]
        perturbed_q = state.body_q.numpy().copy()
        perturbed_q[all_cable_indices, 2] += 17.0
        perturbed_src = wp.array(perturbed_q, dtype=state.body_q.dtype, device=state.body_q.device)
        wp.copy(dest=state.body_q, src=perturbed_src)
        wp.copy(dest=solver.body_q_prev, src=perturbed_src)

        cable.reset(env_ids=[0, 2])

        body_q_after = state.body_q.numpy()
        body_q_prev_after = solver.body_q_prev.numpy()
        body_q_model = NewtonVBDManager._model.body_q.numpy()

        for env_idx in (0, 2):
            slc = list(range(body_offsets[env_idx], body_offsets[env_idx] + n))
            np.testing.assert_array_equal(
                body_q_after[slc],
                body_q_model[slc],
                err_msg=f"env {env_idx}: state.body_q not restored to model.body_q after reset.",
            )
            np.testing.assert_array_equal(
                body_q_prev_after[slc],
                body_q_model[slc],
                err_msg=(
                    f"env {env_idx}: solver.body_q_prev not restored after reset."
                    " AVBD implicit velocity (body_q - body_q_prev)/dt blows up on the next step."
                ),
            )

        for env_idx in (1, 3):
            slc = list(range(body_offsets[env_idx], body_offsets[env_idx] + n))
            np.testing.assert_array_equal(
                body_q_after[slc],
                perturbed_q[slc],
                err_msg=f"env {env_idx}: state.body_q reset despite env not being in env_ids.",
            )
            np.testing.assert_array_equal(
                body_q_prev_after[slc],
                perturbed_q[slc],
                err_msg=f"env {env_idx}: solver.body_q_prev reset despite env not being in env_ids.",
            )
