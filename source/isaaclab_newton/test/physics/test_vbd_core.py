# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonSoftContactCfg

from isaaclab.physics import PhysicsManager


@pytest.mark.parametrize(
    ("soft_contact_cfg", "expected"),
    [
        pytest.param(None, (7.0, 8.0, 9.0), id="preserve"),
        pytest.param(
            NewtonSoftContactCfg(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            id="override",
        ),
    ],
)
def test_soft_contact_cfg_updates_finalized_model(monkeypatch, soft_contact_cfg, expected):
    """Soft-contact configuration updates the finalized model when provided."""
    state_values = []

    class Model:
        soft_contact_ke = 7.0
        soft_contact_kd = 8.0
        soft_contact_mu = 9.0
        world_count = 0
        articulation_count = 0

        def set_gravity(self, gravity):
            pass

        def state(self):
            state_values.append((self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
            return object()

        def control(self):
            return object()

    class Builder:
        body_label = ()
        up_axis = None

        def finalize(self, *, device):
            return model

    model = Model()
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(soft_contact_cfg=soft_contact_cfg), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_builder", Builder(), raising=False)
    monkeypatch.setattr(NewtonManager, "_up_axis", "Z", raising=False)
    monkeypatch.setattr(NewtonManager, "_gravity_vector", (0.0, 0.0, -9.81), raising=False)
    monkeypatch.setattr(NewtonManager, "_num_envs", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_clone_physics_only", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_state_attributes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_contact_attributes", set(), raising=False)
    for attr in (
        "_model",
        "_state_0",
        "_state_1",
        "_control",
        "_adapter",
        "_use_newton_actuators_active",
        "_world_reset_mask",
        "_fk_reset_mask",
    ):
        monkeypatch.setattr(NewtonManager, attr, getattr(NewtonManager, attr, None), raising=False)
    monkeypatch.setattr(NewtonManager, "_cl_pending_sites", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_drain_stale_cuda_error", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "dispatch_event", classmethod(lambda cls, event: None))

    NewtonManager.start_simulation()

    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected

    assert state_values == [expected, expected]


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
    monkeypatch.setattr(newton_module, "import_builder_visual_material_paths", lambda *args: None)
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


@pytest.mark.parametrize("external_rigid_solver", [False, True])
def test_vbd_solver_force_input_capability(monkeypatch, external_rigid_solver):
    """VBD accepts rigid forces only when it integrates rigid bodies."""
    physics = importlib.import_module("isaaclab_newton.physics")
    solver = object()
    monkeypatch.setattr(physics.NewtonVBDManager, "_create_solver", lambda model, cfg: solver)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", False)

    solver_cfg = physics.VBDSolverCfg(integrate_with_external_rigid_solver=external_rigid_solver)
    physics.NewtonVBDManager._build_solver(object(), solver_cfg)

    assert NewtonManager._solver is solver
    assert NewtonManager._supports_rigid_body_force_input is not external_rigid_solver


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
