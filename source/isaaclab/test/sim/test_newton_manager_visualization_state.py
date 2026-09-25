# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-built, registry-owned Newton resources and native SDP publication."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from isaaclab.cloner import ClonePlan

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


def test_clone_inputs_create_one_registry_resource_until_closed(monkeypatch):
    """Cloning produces inputs; consumers acquire one native resource without rebuilding the scene."""
    from isaaclab_newton.cloner import NewtonReplicateContext
    from isaaclab_newton.physics import NewtonManager

    from pxr import Usd, UsdGeom

    from isaaclab.sim import SimulationContext

    _reset_newton_manager_state()
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
    sim.physics_manager = SimpleNamespace(get_device=lambda: "cpu")
    sim.stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(sim.stage, "/Scene/Source")
    sim._backend_registry = []
    context = NewtonReplicateContext(sim)
    builder, _, _ = context.replicate(plan)
    assert not sim._backend_registry
    finalize = Mock(wraps=builder.finalize)
    monkeypatch.setattr(builder, "finalize", finalize)
    first = sim.get_or_create_backend(context.backend_cfg)
    assert sim.get_or_create_backend(context.backend_cfg) is first
    assert first.model.world_count == len(plan.env_ids)
    finalize.assert_called_once_with(device="cpu")
    assert NewtonManager.backend is None
    sim.close_backend(first)
    assert not sim._backend_registry
    assert first.model is first.state_0 is None
    second = sim.get_or_create_backend(context.backend_cfg)
    assert second is not first
    assert finalize.call_count == 2
    sim.close_backend(second)


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
    assert provider.get_transforms(SceneDataFormat.Transform())
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

    context = NewtonReplicateContext(sim)
    builder, _, _ = context.replicate(clone_plan)
    geometry = context.backend_cfg.geometry_offsets

    assert [kwargs["root_path"] for kwargs in usd_imports] == [global_path, sources[0]]
    assert {"/World/Assets/Cloth", *sources} <= set(usd_imports[0]["ignore_paths"])
    assert set(geometry) == {
        "/World/Assets/Cloth",
        "/Copies/env_0/Selected/Cloth",
        "/Copies/env_1/Selected/Cloth",
    }
    assert sorted(geometry.values()) == [0, 3, 6]
    assert builder.particle_count == 9
