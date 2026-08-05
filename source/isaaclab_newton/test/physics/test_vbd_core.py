# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonSolverCfg

from isaaclab.utils.configclass import configclass


@configclass
class _LegacyModelSolverCfg(NewtonSolverCfg):
    """Non-VBD solver config carrying the legacy model field."""

    model_cfg: object | None = None


def test_vbd_symbols_are_exported_from_core():
    """Core exports the VBD manager and config with resolvable names."""
    physics = importlib.import_module("isaaclab_newton.physics")

    assert physics.NewtonVBDManager.__name__ == "NewtonVBDManager"
    assert physics.VBDSolverCfg.__name__ == "VBDSolverCfg"
    assert physics.VBDSolverCfg().class_type.__name__ == "NewtonVBDManager"
    assert issubclass(physics.NewtonVBDManager, NewtonManager)


def test_soft_contact_cfg_defaults_match_newton():
    """Soft-contact defaults match the pinned Newton model."""
    physics = importlib.import_module("isaaclab_newton.physics")
    cfg = physics.NewtonSoftContactCfg()

    assert cfg.soft_contact_ke == pytest.approx(1.0e3)
    assert cfg.soft_contact_kd == pytest.approx(10.0)
    assert cfg.soft_contact_mu == pytest.approx(0.5)
    assert NewtonCfg().soft_contact_cfg is None


def test_legacy_vbd_model_cfg_is_promoted():
    """The core VBD legacy field is promoted with a warning."""
    physics = importlib.import_module("isaaclab_newton.physics")
    legacy_cfg = physics.NewtonSoftContactCfg()

    with pytest.warns(DeprecationWarning, match=r"NewtonCfg\.soft_contact_cfg"):
        cfg = NewtonCfg(solver_cfg=physics.VBDSolverCfg(model_cfg=legacy_cfg))

    assert cfg.soft_contact_cfg == cfg.solver_cfg.model_cfg


def test_duck_typed_legacy_model_cfg_is_promoted():
    """Legacy promotion is not restricted to VBD config types."""
    physics = importlib.import_module("isaaclab_newton.physics")
    legacy_cfg = physics.NewtonSoftContactCfg(soft_contact_ke=321.0)

    with pytest.warns(DeprecationWarning, match=r"NewtonCfg\.soft_contact_cfg"):
        cfg = NewtonCfg(solver_cfg=_LegacyModelSolverCfg(model_cfg=legacy_cfg))

    assert cfg.soft_contact_cfg == cfg.solver_cfg.model_cfg


def test_outer_and_legacy_soft_contact_cfg_are_rejected():
    """Setting both soft-contact paths is ambiguous."""
    physics = importlib.import_module("isaaclab_newton.physics")

    with pytest.raises(ValueError, match="soft-contact configuration"):
        NewtonCfg(
            soft_contact_cfg=physics.NewtonSoftContactCfg(),
            solver_cfg=physics.VBDSolverCfg(model_cfg=physics.NewtonSoftContactCfg()),
        )


def test_vbd_usd_ignore_paths_include_registered_meshes(monkeypatch):
    """VBD excludes registered simulation and visual meshes from USD import."""
    physics = importlib.import_module("isaaclab_newton.physics")
    registry = [
        SimpleNamespace(sim_mesh_prim_path="/World/cloth/sim", vis_mesh_prim_path="/World/cloth/visual"),
        SimpleNamespace(sim_mesh_prim_path="/World/soft/sim", vis_mesh_prim_path="/World/soft/visual"),
    ]
    monkeypatch.setattr(physics.NewtonVBDManager, "_deformable_registry", registry)

    assert physics.NewtonVBDManager._get_usd_ignore_paths() == [
        "/World/cloth/sim",
        "/World/cloth/visual",
        "/World/soft/sim",
        "/World/soft/visual",
    ]


def test_vbd_pre_physics_step_calls_base_and_guards_rebuild(monkeypatch):
    """VBD calls the base hook and rebuilds only when supported."""
    physics = importlib.import_module("isaaclab_newton.physics")
    base_calls = []
    rebuild_calls = []
    state = object()
    model = SimpleNamespace(particle_count=1)

    def base_hook(cls):
        base_calls.append(cls)

    class SolverWithRebuild:
        def rebuild_bvh(self, solver_state):
            rebuild_calls.append(solver_state)

    monkeypatch.setattr(NewtonManager, "_pre_physics_step", classmethod(base_hook))
    monkeypatch.setattr(physics.NewtonVBDManager, "_state_0", state)
    monkeypatch.setattr(physics.NewtonVBDManager, "_model", model)
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", SolverWithRebuild())

    physics.NewtonVBDManager._pre_physics_step()

    model.particle_count = 0
    physics.NewtonVBDManager._pre_physics_step()

    model.particle_count = 1
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", object())
    physics.NewtonVBDManager._pre_physics_step()

    assert base_calls == [physics.NewtonVBDManager] * 3
    assert rebuild_calls == [state]


class _FakeNewtonModel:
    def __init__(self, events):
        self._events = events
        self.soft_contact_ke = 7.0
        self.soft_contact_kd = 8.0
        self.soft_contact_mu = 9.0
        self.body_label = ()
        self.world_count = 0
        self.articulation_count = 0

    def set_gravity(self, gravity):
        self.gravity = gravity

    def state(self):
        self._events.append(("state", self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
        return object()

    def control(self):
        self._events.append("control")
        return object()


class _FakeNewtonBuilder:
    def __init__(self, model):
        self._model = model
        self.up_axis = None

    def finalize(self, device):
        self._model._events.append("finalize")
        return self._model


def _start_with_fake_model(monkeypatch, soft_contact_cfg, hook=None):
    from isaaclab.physics import PhysicsManager

    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    model = _FakeNewtonModel(events)
    builder = _FakeNewtonBuilder(model)

    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=physics.VBDSolverCfg(), soft_contact_cfg=soft_contact_cfg),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_builder", builder, raising=False)
    monkeypatch.setattr(NewtonManager, "_up_axis", "Z", raising=False)
    monkeypatch.setattr(NewtonManager, "_gravity_vector", (0.0, 0.0, -9.81), raising=False)
    monkeypatch.setattr(NewtonManager, "_num_envs", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_clone_physics_only", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_state_attributes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_contact_attributes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_drain_stale_cuda_error", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_register_builder_attributes", classmethod(lambda cls, value: None))
    monkeypatch.setattr(NewtonManager, "_cl_inject_sites_fallback", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "dispatch_event", classmethod(lambda cls, event: None))
    monkeypatch.setattr(NewtonManager, "_post_start_simulation_hooks", [], raising=False)
    if hook is not None:
        NewtonManager._post_start_simulation_hooks.append(hook)

    NewtonManager.start_simulation()
    return model, events


@pytest.mark.parametrize(
    "soft_contact_cfg, expected",
    [
        pytest.param(None, (7.0, 8.0, 9.0), id="preserve_newton_defaults"),
        pytest.param(
            SimpleNamespace(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            id="apply_outer_cfg",
        ),
    ],
)
def test_soft_contact_cfg_is_applied_before_state_allocation(monkeypatch, soft_contact_cfg, expected):
    """Soft-contact values are finalized before Newton state allocation."""
    model, events = _start_with_fake_model(monkeypatch, soft_contact_cfg)

    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected
    assert events[0] == "finalize"
    assert events[1] == ("state", *expected)
    assert events[2] == ("state", *expected)


def test_post_start_hooks_run_and_clear(monkeypatch):
    """Post-start hooks run after allocation and are cleared globally."""
    hook_calls = []

    def hook(manager):
        hook_calls.append(manager)

    _, events = _start_with_fake_model(monkeypatch, None, hook)

    assert hook_calls == [NewtonManager]
    assert events[-1] == "control"
    NewtonManager.clear()
    assert NewtonManager._post_start_simulation_hooks == []


def test_pre_step_hook_precedes_collision_in_both_paths(monkeypatch):
    """The pre-step hook runs before collision in both stepping paths."""
    events = []
    monkeypatch.setattr(NewtonManager, "_pre_physics_step", classmethod(lambda cls: events.append("pre")))
    monkeypatch.setattr(
        NewtonManager,
        "_collision_pipeline",
        SimpleNamespace(collide=lambda state, contacts: events.append("collision")),
        raising=False,
    )
    monkeypatch.setattr(
        NewtonManager, "_run_solver_substeps", classmethod(lambda cls, contacts: events.append("solver"))
    )
    monkeypatch.setattr(NewtonManager, "_update_sensors", classmethod(lambda cls, contacts: events.append("sensors")))
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_decimation", 2, raising=False)
    monkeypatch.setattr(NewtonManager, "_adapter", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [], raising=False)
    monkeypatch.setattr(NewtonManager, "_post_step_callbacks", [], raising=False)

    NewtonManager._simulate_full()
    assert events == [
        "pre",
        "collision",
        "solver",
        "pre",
        "collision",
        "solver",
        "sensors",
    ]

    events.clear()
    NewtonManager._simulate_physics_only()
    assert events == ["pre", "collision", "solver", "sensors"]


class _FakePath:
    def __init__(self, value):
        self.pathString = value


class _FakePrim:
    def __init__(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def GetPath(self):
        return _FakePath(f"/World/{self._name}")


class _FakeWorldPrim:
    def __init__(self, children):
        self._children = children

    def IsValid(self):
        return True

    def GetChildren(self):
        return self._children


class _FakeStage:
    def __init__(self, env_names):
        self._world = _FakeWorldPrim([_FakePrim(name) for name in env_names])

    def GetPrimAtPath(self, path):
        if path == "/World":
            return self._world
        return _FakePrim(path.rsplit("/", 1)[-1])


class _FakeRotation:
    def GetImaginary(self):
        return (0.0, 0.0, 0.0)

    def GetReal(self):
        return 1.0


class _FakeMatrix:
    def ExtractTranslation(self):
        return (0.0, 0.0, 0.0)

    def ExtractRotationQuat(self):
        return _FakeRotation()


class _FakeXformCache:
    def GetLocalToWorldTransform(self, prim):
        return _FakeMatrix()


class _RecordingUsdBuilder:
    def __init__(self):
        self.calls = []

    def add_usd(self, stage, **kwargs):
        self.calls.append(kwargs)
        return {"path_shape_map": {}}


@pytest.mark.parametrize("env_names", [[], ["Env_0", "Env_1"]], ids=["flat", "replicated"])
def test_usd_ignore_paths_are_forwarded_to_all_importers(monkeypatch, env_names):
    """Solver ignore paths reach flat, global, and prototype USD imports."""
    manager_module = importlib.import_module("isaaclab_newton.physics.newton_manager")
    physics = importlib.import_module("isaaclab_newton.physics")
    stage = _FakeStage(env_names)
    builders = []
    ignore_paths = ["/World/cloth/sim", "/World/cloth/visual"]

    class FakeUsdGeom:
        XformCache = _FakeXformCache

        @staticmethod
        def GetStageUpAxis(stage):
            return "Z"

    def create_builder(cls, up_axis=None, **kwargs):
        builder = _RecordingUsdBuilder()
        builders.append(builder)
        return builder

    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(UsdGeom=FakeUsdGeom))
    monkeypatch.setattr(manager_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(manager_module, "replace_newton_builder_shape_colors", lambda builder, stage: None)
    monkeypatch.setattr(manager_module, "replicate_builder_mapping", lambda *args, **kwargs: ({}, []))
    monkeypatch.setattr(physics.NewtonVBDManager, "create_builder", classmethod(create_builder))
    monkeypatch.setattr(
        physics.NewtonVBDManager, "_inject_terrain_heightfields", classmethod(lambda cls, stage, builder: [])
    )
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_get_usd_ignore_paths",
        classmethod(lambda cls: ignore_paths),
    )
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_cl_inject_sites",
        classmethod(lambda cls, builder, sources: ({}, {}, {})),
    )
    monkeypatch.setattr(physics.NewtonVBDManager, "_per_world_builder_hooks", [], raising=False)

    physics.NewtonVBDManager.instantiate_builder_from_stage()

    if not env_names:
        assert builders[0].calls[0]["ignore_paths"] == ignore_paths
        assert "schema_resolvers" in builders[0].calls[0]
    else:
        assert builders[0].calls[0]["ignore_paths"] == ["/World/Env_0", "/World/Env_1", *ignore_paths]
        assert builders[1].calls[0]["root_path"] == "/World/Env_0"
        assert builders[1].calls[0]["ignore_paths"] == ignore_paths
