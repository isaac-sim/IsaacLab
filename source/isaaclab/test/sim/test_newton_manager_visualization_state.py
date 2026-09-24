# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-owned Newton visualization models and SDP state synchronization under foreign physics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from isaaclab.cloner import ClonePlan
from isaaclab.scene_data.deformable_discovery import deformable_entries, deformable_prototypes

pytestmark = pytest.mark.integration


def _reset_newton_manager_state():
    from isaaclab_newton.physics import NewtonManager

    NewtonManager.clear()


def _add_api_schemas(prim, schemas: list[str]) -> None:
    from pxr import Sdf

    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _make_surface_cloth_stage(path: str = "/World/envs/env_0/Cloth"):
    """Author a surface deformable mesh prim at ``path``."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    # Ensure ancestor xforms exist.
    parts = path.strip("/").split("/")
    for i in range(1, len(parts)):
        UsdGeom.Xform.Define(stage, "/" + "/".join(parts[:i]))
    cloth = UsdGeom.Mesh.Define(stage, path)
    _add_api_schemas(cloth.GetPrim(), ["OmniPhysicsDeformableBodyAPI", "OmniPhysicsSurfaceDeformableSimAPI"])
    cloth.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    cloth.CreateFaceVertexCountsAttr([3])
    cloth.CreateFaceVertexIndicesAttr([0, 1, 2])
    return stage


def test_physics_manager_close_only_clears_active_manager_binding(monkeypatch):
    """Only the active physics manager can clear shared SimulationContext state."""
    from isaaclab.physics import PhysicsManager

    class _ActiveManager(PhysicsManager):
        _callbacks = {}

    class _InactiveManager(PhysicsManager):
        pass

    _ActiveManager.close()
    assert PhysicsManager._sim is None

    active_sim = SimpleNamespace(physics_manager=_ActiveManager)
    monkeypatch.setattr(PhysicsManager, "_sim", active_sim, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", "active-cfg", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.25, raising=False)

    monkeypatch.setattr(PhysicsManager, "_callbacks", {1: (None, lambda _: None, 0, "stale", None)}, raising=False)
    _InactiveManager.close()
    assert PhysicsManager._callbacks == {}
    assert (PhysicsManager._sim, PhysicsManager._cfg, PhysicsManager._sim_time) == (active_sim, "active-cfg", 1.25)

    _ActiveManager.close()
    assert (PhysicsManager._sim, PhysicsManager._cfg, PhysicsManager._sim_time) == (None, None, 0.0)


@pytest.mark.parametrize(("body_count", "particle_count"), [(0, 0), (3, 0), (0, 4)])
def test_visualization_model_is_built_during_clone_and_allocated_on_physics_ready(
    monkeypatch, body_count, particle_count
):
    """Cloning owns parsing; READY owns native allocation; getters never discover or allocate."""
    import warp as wp
    from isaaclab_newton.cloner import NewtonReplicateContext
    from isaaclab_newton.cloner import replicate as replicate_module
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelBuilder

    from pxr import Usd, UsdGeom

    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
    from isaaclab.sim import SimulationContext

    class ForeignPhysicsManager(PhysicsManager):
        _callbacks = {}

    _reset_newton_manager_state()
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    plan = ClonePlan(
        sources=("/Scene/Source",),
        destinations=("/Scene/Copy_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
        positions=np.zeros((2, 3), dtype=np.float32),
        context_rows={NewtonReplicateContext: (0,)},
    )
    sim = object.__new__(SimulationContext)
    sim.cfg = SimpleNamespace(physics=object(), device="cpu")
    sim.stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(sim.stage, "/Scene/Source")
    sim.physics_manager = ForeignPhysicsManager
    sim._backend_registry = []
    body_paths = [f"/Scene/Body_{index}" for index in range(body_count)]
    transforms = SceneDataFormat.Transform()
    transforms.transforms = wp.zeros(body_count, dtype=wp.transformf, device="cpu")
    sim._scene_data_provider = SceneDataProvider(
        SimpleNamespace(
            transforms=transforms,
            get_transforms=lambda _format: transforms,
            transforms_version=0,
            transform_paths=body_paths,
            transform_count=body_count,
        )
    )
    monkeypatch.setattr(SimulationContext, "_instance", sim)

    finalize = Mock(
        side_effect=lambda device: SimpleNamespace(
            body_count=body_count,
            body_label=body_paths,
            particle_count=particle_count,
            world_count=2,
            state=lambda: SimpleNamespace(
                body_q=wp.empty(body_count, dtype=wp.transformf, device="cpu") if body_count else None, particle_q=None
            ),
        )
    )
    monkeypatch.setattr(ModelBuilder, "finalize", finalize)
    build = Mock(wraps=replicate_module._replicate_newton)
    monkeypatch.setattr(replicate_module, "_replicate_newton", build)
    context = NewtonReplicateContext(sim)
    assert NewtonManager.get_model() is None
    assert NewtonManager.get_state() is None
    build.assert_not_called()

    builder, _, _ = context.replicate(plan)
    assert isinstance(builder, ModelBuilder)
    build.assert_called_once_with(sim.stage, plan, (0,), sim, up_axis="Z")
    finalize.assert_not_called()
    assert not sim._backend_registry

    ForeignPhysicsManager.dispatch_event(PhysicsEvent.PHYSICS_READY)
    if body_count:
        assert NewtonManager.get_state_0().body_q is transforms.transforms
    first_model = NewtonManager.get_model()
    first_state = NewtonManager.get_state()
    ForeignPhysicsManager.dispatch_event(PhysicsEvent.PHYSICS_READY)
    assert NewtonManager.get_model() is first_model
    assert NewtonManager.get_state() is first_state
    assert (first_model.body_count, first_model.particle_count) == (body_count, particle_count)
    assert first_model.num_envs == NewtonManager.get_num_envs() == 2
    finalize.assert_called_once_with(device="cpu")

    ForeignPhysicsManager.dispatch_event(PhysicsEvent.STOP)
    assert NewtonManager.get_model() is None and NewtonManager.get_state() is None
    assert not sim._backend_registry
    ForeignPhysicsManager.dispatch_event(PhysicsEvent.PHYSICS_READY)
    assert NewtonManager.get_model() is not first_model
    assert finalize.call_count == 2
    assert build.call_count == 1
    ForeignPhysicsManager.dispatch_event(PhysicsEvent.STOP)


@pytest.mark.parametrize("invalidate", ["invalidate_body_state", "invalidate_fk"])
def test_native_publication_reuses_clean_fk_and_refreshes_writes_and_swaps(monkeypatch, invalidate):
    """Clean native reads reuse FK and conversions; writes and solver-buffer swaps refresh their values."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager, NewtonXPBDManager
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    from isaaclab.physics import PhysicsManager
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    _reset_newton_manager_state()
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    state = SimpleNamespace(body_q=wp.array([[0, 0, 0, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu"))
    backend = NewtonSceneDataBackend()
    provider = SceneDataProvider(backend)
    monkeypatch.setattr(
        NewtonManager, "backend", SimpleNamespace(model=SimpleNamespace(body_count=1, world_count=1), state_0=state)
    )
    monkeypatch.setattr(NewtonManager, "_scene_data_backend", backend)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", wp.zeros(2, dtype=wp.bool, device="cpu"))
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", wp.zeros(1, dtype=wp.bool, device="cpu"))
    # Fabric may bind between native allocation and the solver's FK-hook initialization.
    assert backend.transforms.transforms is state.body_q
    monkeypatch.setattr(NewtonManager, "_eval_fk", Mock())
    monkeypatch.setattr(NewtonManager, "_reset_solver_internals_delegate", Mock())
    monkeypatch.setattr(wp, "launch", Mock(wraps=wp.launch))

    output = SceneDataFormat.Matrix44()
    assert provider.get_transforms(output)
    matrices = output.matrices
    NewtonManager.pre_render()
    NewtonManager._eval_fk.assert_not_called()
    NewtonManager.get_state(provider)
    assert provider.get_transforms(output)
    assert output.matrices is matrices
    assert wp.launch.call_count == 1
    NewtonManager._eval_fk.assert_not_called()

    state.body_q.assign([[1, 2, 3, 0, 0, 0, 1]])
    getattr(NewtonXPBDManager, invalidate)()
    assert provider.get_transforms(output)
    assert output.matrices is matrices
    np.testing.assert_allclose(output.matrices.numpy()[0, :3, 3], [1, 2, 3])
    NewtonManager._eval_fk.assert_called_once()
    assert provider.get_transforms(output)
    assert output.matrices is matrices
    NewtonManager.pre_render()
    NewtonManager._eval_fk.assert_called_once()
    assert wp.launch.call_count == 2

    replacement = wp.array([[3, 2, 1, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    NewtonManager.backend.state_0 = SimpleNamespace(body_q=replacement)
    native = SceneDataFormat.Transform()
    assert provider.get_transforms(native)
    assert native.transforms is replacement
    assert provider.get_transforms(output)
    assert output.matrices is matrices
    assert wp.launch.call_count == 3
    np.testing.assert_allclose(output.matrices.numpy()[0, :3, 3], [3, 2, 1])


def test_resolve_scene_data_body_paths_uses_joint_body_targets():
    """PhysX visualization sync maps Newton joint labels to the actual body prim path."""
    pytest.importorskip("pxr")
    from isaaclab_newton.physics import NewtonManager

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/robot0_forearm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/joints/robot0_forearm")
    joint.GetBody1Rel().SetTargets([body_prim.GetPath()])

    body_paths = ["/World/envs/env_0/Robot/joints/robot0_forearm"]
    resolved_paths = NewtonManager._resolve_scene_data_body_paths(body_paths, stage)

    assert resolved_paths == ["/World/envs/env_0/Robot/robot0_forearm"]


@pytest.mark.parametrize("layout", ["identity", "reordered", "missing", "duplicate"])
def test_update_visualization_state_shares_sdp_transforms(monkeypatch, layout):
    """Native and reordered layouts bind shared output once and refresh only on publication."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, provider=None: False))

    body_paths = ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    render_paths = {
        "identity": body_paths,
        "reordered": body_paths[::-1],
        "missing": [body_paths[0], "/World/Missing"],
        "duplicate": [body_paths[0], body_paths[0]],
    }[layout]
    source_transforms = wp.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=wp.transformf,
        device="cpu",
    )
    source_data = SceneDataFormat.Transform()
    source_data.transforms = source_transforms
    provider = SceneDataProvider(
        SimpleNamespace(
            transforms=source_data,
            get_transforms=lambda _format: source_data,
            transforms_version=0,
            transform_paths=body_paths,
            transform_count=len(body_paths),
        )
    )
    monkeypatch.setattr(SceneDataProvider, "usd_stage", property(lambda self: None))

    destination = wp.zeros(len(body_paths), dtype=wp.transformf, device="cpu")
    monkeypatch.setattr(
        NewtonManager,
        "backend",
        SimpleNamespace(
            model=SimpleNamespace(body_label=render_paths, body_count=len(body_paths)),
            state_0=SimpleNamespace(body_q=destination, particle_q=None),
        ),
    )

    if layout in ("missing", "duplicate"):
        with pytest.raises(ValueError, match="one unique SDP transform path"):
            NewtonManager.update_visualization_state(provider)
        return

    remapped = layout == "reordered"
    NewtonManager.update_visualization_state(provider)
    shared = NewtonManager.get_state(provider).body_q
    assert (shared is source_transforms) is not remapped
    np.testing.assert_allclose(shared.numpy(), source_transforms.numpy()[:: -1 if remapped else 1])
    assert NewtonManager.get_state(provider).body_q is shared

    source_data.transforms = wp.array(source_transforms.numpy() + 1.0, dtype=wp.transformf, device="cpu")
    provider.backend.transforms_version += 1
    NewtonManager.update_visualization_state(provider)
    np.testing.assert_allclose(
        NewtonManager.backend.state_0.body_q.numpy(), source_data.transforms.numpy()[:: -1 if remapped else 1]
    )


def test_update_visualization_state_writes_final_geometry_once_per_version(monkeypatch):
    """SDP writes directly into native render slots, preserving gaps and the destination allocation."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    _reset_newton_manager_state()
    data = SceneDataFormat.Points()
    data.points = wp.array(np.arange(18, dtype=np.float32).reshape(6, 3), dtype=wp.vec3f, device="cpu")
    ranges = {"/Cells/Cloth/mesh": (1, 2), "/Shared/Particles": (4, 1)}
    backend = SimpleNamespace(geometry_version=0, get_geometry_batches=lambda _format: [(data, ranges)])
    provider = SceneDataProvider(backend)
    particle_q = wp.zeros(6, dtype=wp.vec3f, device="cpu")
    monkeypatch.setattr(
        NewtonManager,
        "backend",
        SimpleNamespace(model=SimpleNamespace(), state_0=SimpleNamespace(body_q=None, particle_q=particle_q)),
    )
    monkeypatch.setattr(NewtonManager, "_scene_data_geometry_mapping", {"/Shared/Particles": 1, "/Cells/Cloth/mesh": 3})
    monkeypatch.setattr(NewtonManager, "_mark_sensor_state_dirty", Mock())
    monkeypatch.setattr(wp, "launch", Mock(wraps=wp.launch))

    for version in (0, 1):
        if version:
            data.points = wp.array(data.points.numpy() + 10.0, dtype=wp.vec3f, device="cpu")
            backend.geometry_version += 1
        NewtonManager.update_visualization_state(provider)
        NewtonManager.update_visualization_state(provider)
        expected = np.zeros((6, 3), dtype=np.float32)
        expected[1] = data.points.numpy()[4]
        expected[3:5] = data.points.numpy()[1:3]
        assert NewtonManager.backend.state_0.particle_q is particle_q
        np.testing.assert_array_equal(particle_q.numpy(), expected)
        assert wp.launch.call_count == version + 1
        assert NewtonManager._mark_sensor_state_dirty.call_count == version + 1


def test_shadow_deformable_placement_uses_parent_pose_not_root(monkeypatch):
    """Parent-frame baked verts must be placed with the parent world pose."""
    from isaaclab_newton.physics import visualization_deformables as vd

    from pxr import Gf, UsdGeom

    stage = _make_surface_cloth_stage("/World/envs/env_0/ClothRoot/mesh")
    parent = UsdGeom.Xform(stage.GetPrimAtPath("/World/envs/env_0/ClothRoot"))
    parent.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    root = UsdGeom.Mesh(stage.GetPrimAtPath("/World/envs/env_0/ClothRoot/mesh"))
    root.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    builder = Mock(particle_count=0)
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.asarray([0]),
    )
    vd.add_shadow_deformables_to_builder(builder, deformable_entries(plan, deformable_prototypes(stage, plan)))

    # Parent world translation is (10,0,0); root's extra (2,0,0) must not be used as placement.
    assert tuple(builder.add_cloth_mesh.call_args.kwargs["pos"]) == (10.0, 0.0, 0.0)
    builder.add_soft_mesh.assert_not_called()


@pytest.mark.parametrize("global_path", ["/World/Assets/Cloth", "/World/Assets"])
def test_clone_visualization_builder_imports_only_declared_global_deformables(monkeypatch, global_path):
    """Global ancestors do not route excluded clone sources into the shadow model."""
    from isaaclab_newton.cloner import NewtonReplicateContext
    from newton import ModelBuilder

    from pxr import Sdf, UsdGeom

    stage = _make_surface_cloth_stage(path="/World/Assets/Cloth")
    Sdf.CopySpec(stage.GetRootLayer(), "/World/Assets/Cloth", stage.GetRootLayer(), "/World/UnplannedCloth")
    sources = ("/World/Assets/Selected", "/World/Assets/Excluded")
    for source in sources:
        UsdGeom.Xform.Define(stage, source)
        Sdf.CopySpec(stage.GetRootLayer(), "/World/Assets/Cloth", stage.GetRootLayer(), f"{source}/Cloth")
    clone_plan = ClonePlan(
        sources=sources,
        destinations=("/Copies/env_{}/Selected", "/Copies/env_{}/Excluded"),
        env_ids=np.asarray([0, 1], dtype=np.int64),
        clone_mask=np.ones((2, 2), dtype=np.bool_),
        positions=np.zeros((2, 3), dtype=np.float32),
        global_paths=(global_path,),
        context_rows={NewtonReplicateContext: (0,)},
    )
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=object()),
        device="cpu",
        stage=stage,
        physics_manager=SimpleNamespace(register_callback=Mock()),
    )
    usd_imports = []
    add_usd = ModelBuilder.add_usd

    def import_usd(builder, *args, **kwargs):
        usd_imports.append(kwargs)
        return add_usd(builder, *args, **kwargs)

    monkeypatch.setattr(ModelBuilder, "add_usd", import_usd)

    builder, _, _ = NewtonReplicateContext(sim).replicate(clone_plan)
    callback = sim.physics_manager.register_callback.call_args.args[0]
    geometry = callback.args[1]

    assert [kwargs["root_path"] for kwargs in usd_imports] == [global_path, sources[0]]
    assert {"/World/Assets/Cloth", *sources} <= set(usd_imports[0]["ignore_paths"])
    assert set(geometry) == {
        "/World/Assets/Cloth",
        "/Copies/env_0/Selected/Cloth",
        "/Copies/env_1/Selected/Cloth",
    }
    assert sorted(geometry.values()) == [0, 3, 6]
    assert builder.particle_count == 9
