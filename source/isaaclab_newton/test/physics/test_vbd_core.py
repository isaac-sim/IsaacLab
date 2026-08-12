# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonCfg, NewtonManager


def test_vbd_symbols_are_exported_from_core():
    """Core exports the VBD manager and configuration."""
    physics = importlib.import_module("isaaclab_newton.physics")

    assert physics.NewtonVBDManager.__name__ == "NewtonVBDManager"
    assert physics.VBDSolverCfg.__name__ == "VBDSolverCfg"
    assert physics.VBDSolverCfg().class_type.__name__ == "NewtonVBDManager"
    assert issubclass(physics.NewtonVBDManager, NewtonManager)


def test_soft_contact_cfg_defaults_match_newton():
    """Soft-contact defaults match the pinned Newton model."""
    physics = importlib.import_module("isaaclab_newton.physics")
    cfg = physics.NewtonSoftContactCfg()

    assert cfg.soft_contact_ke == 1.0e3
    assert cfg.soft_contact_kd == 10.0
    assert cfg.soft_contact_mu == 0.5
    assert NewtonCfg().soft_contact_cfg is None


@pytest.mark.parametrize("env_paths", [(), ("/World/Env_0", "/World/Env_1")], ids=["flat", "replicated"])
def test_vbd_excludes_registered_deformable_meshes(monkeypatch, env_paths):
    """VBD excludes registered simulation and visual meshes from USD import."""
    physics = importlib.import_module("isaaclab_newton.physics")
    pxr = importlib.import_module("pxr")
    newton_module = importlib.import_module("isaaclab_newton.physics.newton_manager")
    builders = []
    hook_calls = []
    replicate_calls = []

    class Builder:
        def __init__(self):
            self.imports = []
            self.color_calls = 0

        def add_usd(self, stage, *, root_path=None, ignore_paths=(), schema_resolvers=()):
            self.imports.append((root_path, list(ignore_paths)))
            return {"path_shape_map": {}}

        def color(self):
            self.color_calls += 1

    children = [
        SimpleNamespace(
            GetName=lambda path=path: path.rsplit("/", 1)[-1],
            GetPath=lambda path=path: SimpleNamespace(pathString=path),
        )
        for path in env_paths
    ]
    world_prim = SimpleNamespace(IsValid=lambda: True, GetChildren=lambda: children)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: world_prim if path == "/World" else path)
    rotation = SimpleNamespace(GetImaginary=lambda: (0.0, 0.0, 0.0), GetReal=lambda: 1.0)
    matrix = SimpleNamespace(ExtractTranslation=lambda: (0.0, 0.0, 0.0), ExtractRotationQuat=lambda: rotation)
    usd_geom = SimpleNamespace(
        GetStageUpAxis=lambda stage: "Z",
        XformCache=lambda: SimpleNamespace(GetLocalToWorldTransform=lambda prim: matrix),
    )

    def create_builder(cls, *, up_axis):
        builder = Builder()
        builders.append(builder)
        return builder

    def replicate(*args, **kwargs):
        replicate_calls.append(kwargs)
        return {}, [object() for _ in env_paths]

    monkeypatch.setattr(newton_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(pxr, "UsdGeom", usd_geom)
    monkeypatch.setattr(newton_module, "_restore_visible_colliders_without_visual_shapes", lambda *args: None)
    monkeypatch.setattr(newton_module, "replace_newton_builder_shape_colors", lambda *args: None)
    monkeypatch.setattr(newton_module, "replicate_builder_mapping", replicate)
    monkeypatch.setattr(physics.NewtonVBDManager, "create_builder", classmethod(create_builder))
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_inject_terrain_heightfields",
        classmethod(lambda cls, stage, builder: ["/World/terrain"]),
    )
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_cl_inject_sites",
        classmethod(lambda cls, builder, source_builders: ({}, {}, {})),
    )
    monkeypatch.setattr(
        physics.NewtonVBDManager, "set_builder", classmethod(lambda cls, builder: setattr(cls, "_builder", builder))
    )

    def hook(builder, world_idx, position, rotation):
        hook_calls.append(world_idx)

    monkeypatch.setattr(physics.NewtonVBDManager, "_per_world_builder_hooks", [hook])
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_deformable_registry",
        [SimpleNamespace(sim_mesh_prim_path="/World/soft/sim", vis_mesh_prim_path="/World/soft/visual")],
    )
    monkeypatch.setattr(physics.NewtonVBDManager, "_builder", None)
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {})
    monkeypatch.setattr(NewtonManager, "_world_xforms", [])
    monkeypatch.setattr(NewtonManager, "_num_envs", 0)

    physics.NewtonVBDManager.instantiate_builder_from_stage()

    deformable_paths = ["/World/soft/sim", "/World/soft/visual"]
    if env_paths:
        assert builders[0].imports == [(None, [*env_paths, "/World/terrain", *deformable_paths])]
        assert builders[1].imports == [("/World/Env_0", deformable_paths)]
        assert replicate_calls[0]["per_world_builder_hooks"] == [hook]
    else:
        assert builders[0].imports == [(None, ["/World/terrain", *deformable_paths])]
        assert hook_calls == [0]
    assert builders[0].color_calls == 1


def test_vbd_colors_prebuilt_builder_before_start(monkeypatch):
    """VBD colors a prebuilt builder before starting simulation."""
    physics = importlib.import_module("isaaclab_newton.physics")
    deformable_module = importlib.import_module("isaaclab_contrib.deformable.deformable_object")
    events = []

    class Builder:
        def color(self):
            events.append("color")

    monkeypatch.setattr(physics.NewtonVBDManager, "_builder", Builder())
    monkeypatch.setattr(NewtonManager, "start_simulation", classmethod(lambda cls: events.append("start")))
    monkeypatch.setattr(deformable_module, "setup_registered_deformable_fabric_sync", lambda manager_cls: None)

    physics.NewtonVBDManager.start_simulation()

    assert events == ["color", "start"]


def test_vbd_rebuilds_particle_bvh_before_physics_step(monkeypatch):
    """VBD rebuilds its particle BVH before the base physics step."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    state = object()

    class Solver:
        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

    def simulate_physics_only(cls):
        events.append(("step", cls))

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(simulate_physics_only))
    monkeypatch.setattr(physics.NewtonVBDManager, "_model", SimpleNamespace(particle_count=1))
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", Solver())
    monkeypatch.setattr(physics.NewtonVBDManager, "_state_0", state)

    physics.NewtonVBDManager._simulate_physics_only()

    assert events == [("rebuild", state), ("step", physics.NewtonVBDManager)]
