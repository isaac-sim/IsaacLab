# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

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

    def _fake_entry_hook(builder, entry, env_idx, env_pos, env_rot):
        calls.append((entry.prim_path, env_idx))

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

    assert calls == [("/World/cable_a", 3), ("/World/cable_b", 3)]


class _FakeBuilder:
    """Records the arguments passed to add_rod_graph for assertion."""

    def __init__(self):
        self.calls = []

    def add_rod_graph(self, **kwargs):
        self.calls.append(kwargs)
        return [], []  # body_indices, joint_indices — match Newton's signature


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
        # registry not installed → RuntimeError mentioning "install_cable_builder_hooks"
        (False, "valid", RuntimeError, "install_cable_builder_hooks"),
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
    from isaaclab.utils import configclass

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
