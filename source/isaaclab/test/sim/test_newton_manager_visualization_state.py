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
from isaaclab.scene_data.deformable_discovery import discover_deformables_on_stage

pytestmark = pytest.mark.integration

_SINGLE_ENV_PLAN = ClonePlan(
    sources=("/World/envs/env_0",),
    destinations=("/World/envs/env_{}",),
    clone_mask=np.ones((1, 1), dtype=np.bool_),
    env_ids=np.arange(1),
    positions=np.zeros((1, 3), dtype=np.float32),
)


def _reset_newton_manager_state():
    from isaaclab_newton.physics import NewtonManager

    NewtonManager.clear()


def _add_api_schemas(prim, schemas: list[str]) -> None:
    from pxr import Sdf

    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _make_points_backend(points, geometry_paths: list[str], geometry_counts: list[int]):
    """Build a minimal SceneData backend that exposes flattened geometry points."""
    import warp as wp

    from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat

    class _PointsBackend(SceneDataBackend):
        def __init__(self):
            self._points = wp.array(points, dtype=wp.vec3f, device="cpu")
            self._points_data = SceneDataFormat.Points()
            self._points_data.points = self._points
            self._geometry_paths = list(geometry_paths)
            self._geometry_counts = list(geometry_counts)

        @property
        def transforms(self) -> SceneDataFormat.Transform:
            return SceneDataFormat.Transform()

        @property
        def transform_count(self) -> int:
            return 0

        @property
        def transform_paths(self) -> list[str]:
            return []

        @property
        def points(self) -> SceneDataFormat.Points:
            return self._points_data

        @property
        def point_count(self) -> int:
            return int(self._points.shape[0])

        @property
        def geometry_paths(self) -> list[str]:
            return self._geometry_paths

        @property
        def geometry_counts(self) -> list[int]:
            return self._geometry_counts

    return _PointsBackend()


def _make_shadow_entity(
    root_path: str,
    *,
    sim_particle_offset: int = 0,
    sim_particle_count: int,
    vis_particle_offset: int | None = None,
    vis_particle_count: int | None = None,
    volume_vis_remap=None,
):
    from isaaclab_newton.physics.visualization_deformables import ShadowDeformableEntity

    if vis_particle_offset is None:
        vis_particle_offset = sim_particle_offset
    if vis_particle_count is None:
        vis_particle_count = sim_particle_count
    return ShadowDeformableEntity(
        root_path=root_path,
        sim_particle_offset=sim_particle_offset,
        sim_particle_count=sim_particle_count,
        vis_particle_offset=vis_particle_offset,
        vis_particle_count=vis_particle_count,
        volume_vis_remap=volume_vis_remap,
    )


def _prepare_physx_shadow_sync(monkeypatch, provider, *, model, state_0, entities, sim_particle_count: int):
    """Wire NewtonManager for a PhysX-backed visualization particle sync test."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model, state_0=state_0))
    NewtonManager._shadow_deformable_entities = list(entities)
    if sim_particle_count > 0:
        NewtonManager._sim_particle_q = wp.zeros(sim_particle_count, dtype=wp.vec3f, device="cpu")
    return NewtonManager


class _FakeShadowBuilder:
    """ModelBuilder stub for :func:`add_shadow_deformables_to_builder` tests."""

    def __init__(
        self,
        *,
        particle_count: int = 0,
        cloth_delta: int = 1,
        soft_delta: int = 4,
        reject_soft: bool = False,
    ):
        self.particle_count = particle_count
        self.cloth_calls = 0
        self.soft_calls = 0
        self._cloth_delta = cloth_delta
        self._soft_delta = soft_delta
        self._reject_soft = reject_soft
        self._captured: dict = {}

    def add_cloth_mesh(self, **kwargs):
        self.cloth_calls += 1
        self.particle_count += self._cloth_delta
        self._captured.update(kwargs)

    def add_soft_mesh(self, **kwargs):
        if self._reject_soft:
            raise AssertionError("surface cloth should not call add_soft_mesh")
        self.soft_calls += 1
        self.particle_count += self._soft_delta
        self._captured.update(kwargs)

    @property
    def captured(self) -> dict:
        return self._captured


def _make_volume_soft_stage(*, env_ids: tuple[int, ...] = (0,), root_name: str = "Soft"):
    """Author a volume deformable with tet sim mesh + single-vertex visual mesh."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    for env_id in env_ids:
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        root = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/{root_name}").GetPrim()
        _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
        tet = UsdGeom.TetMesh.Define(stage, f"/World/envs/env_{env_id}/{root_name}/simulation")
        _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
        points = [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 0.0, 1.0),
        ]
        tet.CreatePointsAttr(points)
        tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])
        vis = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_id}/{root_name}/visual")
        vis.CreatePointsAttr([Gf.Vec3f(0.25, 0.25, 0.25)])
        vis.CreateFaceVertexCountsAttr([3])
        vis.CreateFaceVertexIndicesAttr([0, 0, 0])
    return stage


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
    from isaaclab_newton.cloner import NewtonReplicateContext
    from isaaclab_newton.cloner import replicate as replicate_module
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelBuilder

    from pxr import Usd, UsdGeom

    from isaaclab.physics import PhysicsEvent, PhysicsManager
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
    sim._scene_data_provider = SimpleNamespace(backend=object(), point_count=0)
    monkeypatch.setattr(SimulationContext, "_instance", sim)

    finalize = Mock(
        side_effect=lambda device: SimpleNamespace(
            body_count=body_count,
            particle_count=particle_count,
            world_count=2,
            state=lambda: SimpleNamespace(body_q=None, particle_q=None),
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


def test_update_visualization_state_noop_when_backend_is_newton(monkeypatch):
    """When sim backend is Newton, update_visualization_state is a no-op."""
    from isaaclab_newton.physics import NewtonManager

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: True))
    monkeypatch.setattr(NewtonManager, "get_scene_data_provider", classmethod(lambda cls: SimpleNamespace()))

    # Pre-set sentinel values to ensure update doesn't touch them.
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model="live-model", state_0="live-state"))
    NewtonManager.update_visualization_state()
    assert NewtonManager.backend.model == "live-model"
    assert NewtonManager.backend.state_0 == "live-state"


@pytest.mark.parametrize("newton_active", [True, False])
def test_get_state_forwards_only_for_live_newton_state(monkeypatch, newton_active):
    """PhysX shadow state keeps its visualization update without entering Newton FK."""
    from isaaclab_newton.physics import NewtonManager

    events: list[str] = []
    state = object()
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", object(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_backend_is_newton",
        classmethod(lambda cls, provider=None: newton_active),
    )
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(
        NewtonManager,
        "update_visualization_state",
        classmethod(lambda cls, provider=None: events.append("visualization")),
    )
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))

    assert NewtonManager.get_state() is state
    expected = ["forward", "visualization"] if newton_active else ["visualization"]
    assert events == expected


def test_scene_data_reads_through_public_state_boundary(monkeypatch):
    """SceneData does not bypass the coherent Newton state accessor."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    events: list[str] = []
    body_q = wp.zeros(1, dtype=wp.transformf, device="cpu")
    state = SimpleNamespace(body_q=body_q)
    backend = nm.NewtonSceneDataBackend()
    monkeypatch.setattr(
        NewtonManager,
        "get_state",
        classmethod(lambda cls, provider=None: events.append("state") or state),
    )

    transforms = backend.transforms

    assert events == ["state"]
    assert transforms.transforms is body_q


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


def test_update_visualization_state_copies_identity_mapped_transforms(monkeypatch):
    """Identity-mapped transforms update the persistent Newton shadow buffer."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, provider=None: False))

    body_paths = ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
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
    provider_impl = SceneDataProvider(
        SimpleNamespace(transforms=source_data, transform_paths=body_paths, transform_count=len(body_paths))
    )
    provider = SimpleNamespace(
        usd_stage=None,
        create_mapping=provider_impl.create_mapping,
        get_transforms=provider_impl.get_transforms,
        point_count=0,
    )

    destination = wp.zeros(len(body_paths), dtype=wp.transformf, device="cpu")
    monkeypatch.setattr(
        NewtonManager,
        "backend",
        SimpleNamespace(
            model=SimpleNamespace(body_label=body_paths, body_count=len(body_paths)),
            state_0=SimpleNamespace(body_q=destination, particle_q=None),
        ),
    )

    NewtonManager.update_visualization_state(provider)

    assert NewtonManager.backend.state_0.body_q is destination
    assert NewtonManager._scene_data.transforms is destination
    np.testing.assert_allclose(destination.numpy(), source_transforms.numpy())


def test_update_visualization_state_syncs_shadow_particle_q(monkeypatch):
    """PhysX/OVPhysX shadow sync copies backend points into ``state.particle_q``.

    Uses a real :class:`~isaaclab.scene_data.SceneDataProvider` with an identity
    geometry mapping (``None``). Passthrough must not rebind away from the shadow
    ``particle_q`` buffer — that left OVRTX rendering rest-pose cloth under OVPhysX.
    """
    import warp as wp

    from isaaclab.scene_data.scene_data_provider import SceneDataProvider

    cloth_path = "/World/envs/env_0/Cloth"
    provider = SceneDataProvider(
        _make_points_backend(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [cloth_path],
            [2],
        )
    )
    monkeypatch.setattr(provider, "get_transforms", lambda output, mapping=None, allow_passthrough=True: True)

    particle_q = wp.zeros(2, dtype=wp.vec3f, device="cpu")
    NewtonManager = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=["/World/envs/env_0/Robot"]),
        state_0=SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
        entities=[_make_shadow_entity(cloth_path, sim_particle_count=2)],
        sim_particle_count=2,
    )

    NewtonManager.update_visualization_state(provider)

    # Shadow buffer must remain the same object and receive a copy of live points.
    assert NewtonManager.backend.state_0.particle_q is particle_q
    copied = particle_q.numpy()
    assert copied[0].tolist() == [1.0, 2.0, 3.0]
    assert copied[1].tolist() == [4.0, 5.0, 6.0]


def test_update_visualization_state_remaps_volume_vis_positions(monkeypatch):
    """Volume shadow sync barycentrically remaps sim nodes into vis-sized render slots."""
    import numpy as np
    import warp as wp

    from isaaclab.scene_data.deformable_vis_remap import build_volume_vis_barycentric_remap
    from isaaclab.scene_data.scene_data_provider import SceneDataProvider

    soft_path = "/World/envs/env_0/Soft"
    remap = build_volume_vis_barycentric_remap(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([[0.25, 0.25, 0.25]], dtype=np.float32),
    )
    assert remap is not None

    provider = SceneDataProvider(
        _make_points_backend(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [soft_path],
            [4],
        )
    )
    monkeypatch.setattr(provider, "get_transforms", lambda output, mapping=None, allow_passthrough=True: True)

    particle_q = wp.zeros(1, dtype=wp.vec3f, device="cpu")
    NewtonManager = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=["/World/envs/env_0/Robot"]),
        state_0=SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
        entities=[
            _make_shadow_entity(
                soft_path,
                sim_particle_count=4,
                vis_particle_count=1,
                volume_vis_remap=remap,
            )
        ],
        sim_particle_count=4,
    )

    NewtonManager.update_visualization_state(provider)

    np.testing.assert_allclose(particle_q.numpy()[0], [0.25, 0.25, 0.25], atol=1e-4)


def test_sync_skips_unmapped_deformable_rest_pose(monkeypatch):
    """Shadow entities with no SceneData mapping must keep rest pose, not copy zeros."""
    import numpy as np
    import warp as wp

    from isaaclab.scene_data.scene_data_provider import SceneDataProvider

    provider = SceneDataProvider(
        _make_points_backend(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ["/World/envs/env_0/ClothA"],
            [2],
        )
    )
    monkeypatch.setattr(provider, "get_transforms", lambda output, mapping=None, allow_passthrough=True: True)

    particle_q = wp.array(
        [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(9.0, 9.0, 9.0),
            wp.vec3(8.0, 8.0, 8.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    NewtonManager = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=["/World/envs/env_0/Robot"]),
        state_0=SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
        entities=[
            _make_shadow_entity("/World/envs/env_0/ClothA", sim_particle_count=2),
            _make_shadow_entity(
                "/World/envs/env_0/ClothB",
                sim_particle_offset=2,
                sim_particle_count=2,
                vis_particle_offset=2,
            ),
        ],
        sim_particle_count=4,
    )

    NewtonManager.update_visualization_state(provider)

    copied = particle_q.numpy()
    assert copied[0].tolist() == [1.0, 2.0, 3.0]
    assert copied[1].tolist() == [4.0, 5.0, 6.0]
    # Unmapped ClothB must keep rest pose rather than receiving zeroed sim slices.
    np.testing.assert_allclose(copied[2], [9.0, 9.0, 9.0], atol=1e-6)
    np.testing.assert_allclose(copied[3], [8.0, 8.0, 8.0], atol=1e-6)


def test_sync_skips_mismatched_volume_without_remap(monkeypatch):
    """Count-mismatched volume entities without a remap must not over-read sim particles."""
    import numpy as np
    import warp as wp

    from isaaclab.scene_data.scene_data_provider import SceneDataProvider

    soft_path = "/World/envs/env_0/Soft"
    provider = SceneDataProvider(
        _make_points_backend(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [soft_path],
            [4],
        )
    )
    monkeypatch.setattr(provider, "get_transforms", lambda output, mapping=None, allow_passthrough=True: True)

    # Vis-sized render buffer initialized to a sentinel rest pose.
    particle_q = wp.array([wp.vec3(9.0, 9.0, 9.0)], dtype=wp.vec3f, device="cpu")
    NewtonManager = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=["/World/envs/env_0/Robot"]),
        state_0=SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
        entities=[
            _make_shadow_entity(
                soft_path,
                sim_particle_count=4,
                vis_particle_count=1,
                volume_vis_remap=None,
            )
        ],
        sim_particle_count=4,
    )

    NewtonManager.update_visualization_state(provider)

    np.testing.assert_allclose(particle_q.numpy()[0], [9.0, 9.0, 9.0], atol=1e-6)


def test_shadow_deformable_volume_remap_registers_ovrtx_with_vis_mesh(monkeypatch):
    """Volume bodies with barycentric remap use visual mesh slots and register for OVRTX."""
    from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder

    stage = _make_volume_soft_stage()
    builder = _FakeShadowBuilder(cloth_delta=1, soft_delta=4)
    entries = discover_deformables_on_stage(stage, root_paths=_SINGLE_ENV_PLAN.sources)
    flat_entities, registry_groups = add_shadow_deformables_to_builder(builder, stage, entries, _SINGLE_ENV_PLAN, (0,))

    assert builder.cloth_calls == 1
    assert builder.soft_calls == 0
    assert len(flat_entities) == 1
    assert flat_entities[0].sim_particle_count == 4
    assert flat_entities[0].vis_particle_count == 1
    assert flat_entities[0].volume_vis_remap is not None
    assert len(registry_groups) == 1
    assert registry_groups[0].register_usd_vis_point_bindings is True
    assert registry_groups[0].particles_per_body == 1


def test_shadow_deformable_volume_remap_failure_falls_back_to_soft_mesh(monkeypatch):
    """Failed volume remap must allocate sim tet slots, not mismatched vis cloth slots."""
    from isaaclab_newton.physics import visualization_deformables as vd

    stage = _make_volume_soft_stage()
    monkeypatch.setattr(vd, "_build_volume_vis_remap", lambda entry, device: None)
    builder = _FakeShadowBuilder(cloth_delta=1, soft_delta=4)
    entries = discover_deformables_on_stage(stage, root_paths=_SINGLE_ENV_PLAN.sources)
    flat_entities, registry_groups = vd.add_shadow_deformables_to_builder(
        builder, stage, entries, _SINGLE_ENV_PLAN, (0,)
    )

    assert builder.cloth_calls == 0
    assert builder.soft_calls == 1
    assert len(flat_entities) == 1
    assert flat_entities[0].volume_vis_remap is None
    assert flat_entities[0].sim_particle_count == 4
    assert flat_entities[0].vis_particle_count == 4
    assert registry_groups[0].register_usd_vis_point_bindings is False


def test_shadow_deformable_placement_uses_parent_pose_not_root(monkeypatch):
    """Parent-frame baked verts must be placed with the parent world pose."""
    from isaaclab_newton.physics import visualization_deformables as vd

    from pxr import Gf, Usd, UsdGeom

    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    parent.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/ClothRoot")
    root.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    UsdGeom.Mesh.Define(stage, "/World/envs/env_0/ClothRoot/mesh")

    entry = DeformableStageEntry(
        root_path="/World/envs/env_0/ClothRoot",
        sim_mesh_path="/World/envs/env_0/ClothRoot/mesh",
        vis_mesh_path="/World/envs/env_0/ClothRoot/mesh",
        deformable_type="surface",
        vertex_count=3,
        vis_vertex_count=3,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        vis_vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        vis_indices=[0, 1, 2],
    )
    builder = _FakeShadowBuilder(cloth_delta=3, reject_soft=True)
    vd.add_shadow_deformables_to_builder(builder, stage, [entry], _SINGLE_ENV_PLAN, (0,))

    # Parent world translation is (10,0,0); root's extra (2,0,0) must not be used as placement.
    assert tuple(float(v) for v in builder.captured["pos"]) == (10.0, 0.0, 0.0)


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
    shadow_entities, registry_groups = callback.args[1]

    assert [kwargs["root_path"] for kwargs in usd_imports] == [global_path, sources[0]]
    assert {"/World/Assets/Cloth", *sources} <= set(usd_imports[0]["ignore_paths"])
    assert {entity.root_path for entity in shadow_entities} == {
        "/World/Assets/Cloth",
        "/Copies/env_0/Selected/Cloth",
        "/Copies/env_1/Selected/Cloth",
    }
    assert {group.prim_path for group in registry_groups} == {"/World/Assets/Cloth", "/Copies/env_[^/]+/Selected/Cloth"}
    assert builder.particle_count == sum(entity.vis_particle_count for entity in shadow_entities)
