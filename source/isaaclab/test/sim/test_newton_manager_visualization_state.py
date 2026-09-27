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

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
from isaaclab.sim import SpawnerCfg

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
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    from pxr import Usd, UsdGeom

    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab.renderers import RenderContext
    from isaaclab.sim import SimulationContext

    class Manager(PhysicsManager):
        _callbacks = {}

    _reset_newton_manager_state()
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    asset = AssetBaseCfg(prim_path="/Scene/Copy_[^/]+", spawn=SpawnerCfg(spawn_path="/Scene/Source"))
    plan = make_clone_plan((asset,), ((0,),), 2, env_template="/Scene/Copy_{}")
    sim = object.__new__(SimulationContext)
    sim.cfg = SimpleNamespace(physics=object(), device="cpu")
    sim.physics_manager = Manager
    sim.stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(sim.stage, "/Scene/Source")
    sim._backend_registry = []
    sim._render_context = RenderContext(sim._backend_registry)
    sim.requires_usd_stage = sim.requires_newton_model = False
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    renderers = [sim.get_or_create_backend(NewtonWarpRendererCfg(enable_shadows=flag)) for flag in (False, True)]
    context = NewtonReplicateContext(sim)
    builder, _, _ = context.replicate(plan, (0,))
    assert len(sim._backend_registry) == 2  # Renderer objects only; native allocation waits for initialization.
    assert not hasattr(sim, "newton_cfg") and not hasattr(sim, "fabric_cfg")
    assert renderers[0].newton_cfg is renderers[1].newton_cfg
    finalize = Mock(wraps=builder.finalize)
    monkeypatch.setattr(builder, "finalize", finalize)
    first = sim.get_or_create_backend(renderers[0].newton_cfg)
    assert sim.get_or_create_backend(renderers[1].newton_cfg) is first
    assert first.model.world_count == len(plan.topology.world_prototype_layout)
    finalize.assert_called_once_with(device="cpu")
    assert NewtonManager.backend is None
    sim.close_backend(first)
    assert len(sim._backend_registry) == 2
    assert first.model is first.state_0 is None
    replacement_cfg = renderers[0].newton_cfg.replace(num_envs=2)
    Manager.dispatch_event(PhysicsEvent.BACKEND_CFG_READY, replacement_cfg)
    assert all(renderer.newton_cfg is replacement_cfg for renderer in renderers)
    second = sim.get_or_create_backend(renderers[1].newton_cfg)
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
    assets = tuple(
        AssetBaseCfg(prim_path=dst.format("[^/]+"), spawn=SpawnerCfg(spawn_path=src))
        for src, dst in zip(sources, ("/Copies/env_{}/Selected", "/Copies/env_{}/Excluded"), strict=True)
    )
    assets += (AssetBaseCfg(prim_path=global_path),)
    plan = make_clone_plan(assets, ((0, 1),), 2, shared_assets=(2,), env_template="/Copies/env_{}")
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=object()),
        device="cpu",
        stage=stage,
        physics_manager=SimpleNamespace(dispatch_event=Mock()),
    )
    usd_imports = []
    add_usd = ModelBuilder.add_usd

    def import_usd(builder, *args, **kwargs):
        usd_imports.append(kwargs)
        return add_usd(builder, *args, **kwargs)

    monkeypatch.setattr(ModelBuilder, "add_usd", import_usd)

    context = NewtonReplicateContext(sim)
    builder, _, _ = context.replicate(plan, (0, 2))
    geometry = sim.physics_manager.dispatch_event.call_args.args[1].geometry_offsets

    assert sorted(kwargs["root_path"] for kwargs in usd_imports) == sorted([global_path, sources[0]])
    assert set(geometry) == {"/World/Assets/Cloth", "/Copies/env_0/Selected/Cloth", "/Copies/env_1/Selected/Cloth"}
    assert sorted(geometry.values()) == [0, 3, 6]
    assert builder.particle_count == 9
