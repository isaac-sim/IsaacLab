# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Newton render-state pointers and retained deformable conversion regressions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner.replicate import NewtonReplicateContext

from isaaclab.scene_data import SceneDataFormat, SceneDataProvider, SceneDataPublication

pytestmark = pytest.mark.integration


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
            self._publication = SceneDataPublication(SceneDataFormat.Transform())

        @property
        def transform_publication(self):
            return self._publication

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
    """Build the native resource around the supplied numerical test buffers."""
    sim = SimpleNamespace(cfg=SimpleNamespace(physics=None), device="cpu", get_scene_data_provider=lambda: provider)
    resource = NewtonReplicateContext(sim)
    model.body_count = len(model.body_label)
    resource.model, resource.state_0 = model, state_0
    resource._shadow_deformable_entities = list(entities)
    if sim_particle_count:
        resource._sim_particle_q = wp.zeros(sim_particle_count, dtype=wp.vec3f, device="cpu")
    return resource


class _FakeShadowBuilder:
    """ModelBuilder stub for :func:`add_shadow_deformables_to_builder` tests."""

    def __init__(
        self,
        *,
        particle_count: int = 0,
        cloth_delta: int = 1,
        soft_delta: int = 4,
        reject_soft: bool = False,
        body_count: int = 0,
        track_usd: bool = False,
    ):
        self.particle_count = particle_count
        self.cloth_calls = 0
        self.soft_calls = 0
        self.body_count = body_count
        self.ignore_paths = None
        self._cloth_delta = cloth_delta
        self._soft_delta = soft_delta
        self._reject_soft = reject_soft
        self._track_usd = track_usd
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

    def add_usd(self, stage, schema_resolvers=None, ignore_paths=None, **kwargs):
        if not self._track_usd:
            raise AssertionError("add_usd was not expected")
        self.ignore_paths = list(ignore_paths or [])
        return {"path_shape_map": {}}

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


def test_native_render_state_aliases_sdp_and_follows_pointer_swaps():
    """A foreign renderer reuses the producer pointer instead of copying into a shadow buffer."""
    paths = ["/World/Body"]
    data = SceneDataFormat.Transform()
    data.transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    publication = SceneDataPublication(data)
    backend = SimpleNamespace(
        transform_publication=publication,
        transforms=data,
        transform_count=1,
        transform_paths=paths,
        point_count=0,
    )
    provider = SceneDataProvider(backend)
    resource = NewtonReplicateContext(
        SimpleNamespace(cfg=SimpleNamespace(physics=None), device="cpu", get_scene_data_provider=lambda: provider)
    )
    resource.model = SimpleNamespace(body_label=paths, body_count=1)
    resource.state_0 = SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf, device="cpu"), particle_q=None)

    resource.update_transforms()
    assert resource.state_0.body_q is data.transforms
    generation = provider.transform_generation
    resource.update_transforms()
    assert provider.transform_generation == generation

    data.transforms = wp.array([[4, 5, 6, 0, 0, 0, 1]], dtype=wp.transformf, device="cpu")
    publication.dirty = True
    resource.update_transforms()
    assert resource.state_0.body_q is data.transforms
    np.testing.assert_allclose(resource.state_0.body_q.numpy()[0, :3], [4, 5, 6])


@pytest.mark.parametrize("body_paths", [["/World/Missing"], ["/World/Body", "/World/Body"]])
def test_native_render_state_requires_complete_unique_mapping(body_paths):
    """Unpublished or duplicate model bodies must not expose uninitialized converted poses."""
    provider = SceneDataProvider(SimpleNamespace(transform_paths=["/World/Body"]))
    resource = NewtonReplicateContext(
        SimpleNamespace(cfg=SimpleNamespace(physics=None), device="cpu", get_scene_data_provider=lambda: provider)
    )
    resource.model = SimpleNamespace(body_label=body_paths, body_count=len(body_paths))

    with pytest.raises(ValueError, match="one unique SDP transform path"):
        resource.update_transforms()


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
    resource = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=[]),
        state_0=SimpleNamespace(body_q=wp.zeros(0, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
        entities=[_make_shadow_entity(cloth_path, sim_particle_count=2)],
        sim_particle_count=2,
    )

    resource.update_transforms()

    # Shadow buffer must remain the same object and receive a copy of live points.
    assert resource.state_0.particle_q is particle_q
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
    resource = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=[]),
        state_0=SimpleNamespace(body_q=wp.zeros(0, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
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

    resource.update_transforms()

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
    resource = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=[]),
        state_0=SimpleNamespace(body_q=wp.zeros(0, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
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

    resource.update_transforms()

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
    resource = _prepare_physx_shadow_sync(
        monkeypatch,
        provider,
        model=SimpleNamespace(body_label=[]),
        state_0=SimpleNamespace(body_q=wp.zeros(0, dtype=wp.transformf, device="cpu"), particle_q=particle_q),
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

    resource.update_transforms()

    np.testing.assert_allclose(particle_q.numpy()[0], [9.0, 9.0, 9.0], atol=1e-6)


def test_shadow_deformable_volume_remap_registers_ovrtx_with_vis_mesh(monkeypatch):
    """Volume bodies with barycentric remap use visual mesh slots and register for OVRTX."""
    from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder

    stage = _make_volume_soft_stage()
    builder = _FakeShadowBuilder(cloth_delta=1, soft_delta=4)
    flat_entities, registry_groups = add_shadow_deformables_to_builder(builder, stage, [(0, "/World/envs/env_0")])

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
    flat_entities, registry_groups = vd.add_shadow_deformables_to_builder(builder, stage, [(0, "/World/envs/env_0")])

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
    monkeypatch.setattr(vd, "discover_deformables_on_stage", lambda stage: [entry])
    monkeypatch.setattr(vd, "sort_deformable_entries_for_geometry_sync", lambda entries: entries)

    builder = _FakeShadowBuilder(cloth_delta=3, reject_soft=True)
    vd.add_shadow_deformables_to_builder(builder, stage, [(0, "/World/envs/env_0")])

    # Parent world translation is (10,0,0); root's extra (2,0,0) must not be used as placement.
    assert tuple(float(v) for v in builder.captured["pos"]) == (10.0, 0.0, 0.0)


def test_clone_visualization_builder_ignores_non_env_deformables_on_world_import(monkeypatch):
    """Clone-path world ``add_usd`` must ignore deformables outside ``/World/envs``."""
    from isaaclab_newton.physics import visualization_builder as vb

    from pxr import UsdGeom

    stage = _make_surface_cloth_stage(path="/World/Assets/Cloth")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_1")

    fake_builder = _FakeShadowBuilder(body_count=1, cloth_delta=3, track_usd=True)
    fake_builder.shape_collision_filter_pairs = []
    fake_builder.shape_collision_group = []
    fake_builder.shape_count = 0
    fake_builder.body_label = []
    fake_builder.add_builder = lambda _builder: None
    clone_plan = SimpleNamespace(
        global_paths=("/World/Assets/Cloth",),
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        env_ids=np.asarray([0, 1], dtype=np.int64),
        clone_mask=np.asarray([[False, False]], dtype=np.bool_),
        positions=np.zeros((2, 3), dtype=np.float32),
    )
    monkeypatch.setattr(vb, "ModelBuilder", lambda up_axis="Z": fake_builder)
    monkeypatch.setattr(vb, "_restore_visible_colliders_without_visual_shapes", lambda *args, **kwargs: None)
    monkeypatch.setattr(vb, "import_builder_visual_material_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(vb, "build_source_builders", lambda *args, **kwargs: {})
    monkeypatch.setattr(vb, "replicate_builder_mapping", lambda *args, **kwargs: ({}, [], []))

    _builder, (shadow_entities, registry_groups) = vb.build_visualization_builder_from_plan(stage, clone_plan)

    assert "/World/Assets/Cloth" in fake_builder.ignore_paths
    assert any(entity.root_path == "/World/Assets/Cloth" for entity in shadow_entities)
    assert any(
        group.prim_path == "/World/Assets/Cloth" and "/World/envs/" not in group.prim_path for group in registry_groups
    )
