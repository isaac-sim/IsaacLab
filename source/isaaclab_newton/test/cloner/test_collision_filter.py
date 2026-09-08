# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative collision filtering in Newton replication."""

import warnings
from types import SimpleNamespace
from unittest import mock

import isaaclab_newton.cloner.newton_clone_utils as newton_clone_utils_module
import isaaclab_newton.cloner.replicate as replicate_module
import newton
import numpy as np
import pytest
from isaaclab_newton.cloner.collision_filter import (
    AuthoredCollisionFilterSnapshot,
    NewtonCollisionFilter,
    collider_shape_map,
    collision_endpoint_shape_map,
    snapshot_authored_collision_filter,
)
from isaaclab_newton.cloner.newton_clone_utils import replicate_builder_mapping
from isaaclab_newton.physics import NewtonManager

from isaaclab.cloner import ClonePlan
from isaaclab.physics import CollisionFilterCfg, CollisionGroupCfg, PhysicsManager


def _builder_with_colliders(*paths: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    for path in paths:
        body = builder.add_body(label=path.rpartition("/")[0])
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label=path)
    return builder


def _authored(*pairs: tuple[str, str], **kwargs) -> AuthoredCollisionFilterSnapshot:
    return AuthoredCollisionFilterSnapshot(filtered_pairs=frozenset(pairs), **kwargs)


def _configure_replicate_test_manager(monkeypatch) -> None:
    class _TestManager:
        @staticmethod
        def create_builder(up_axis="Z"):
            return newton.ModelBuilder(up_axis=up_axis)

        @staticmethod
        def _get_usd_import_schema_resolvers():
            return []

        @staticmethod
        def _inject_terrain_heightfields(stage, builder, root_paths):
            return []

    monkeypatch.setattr(
        PhysicsManager,
        "_sim",
        SimpleNamespace(physics_manager=_TestManager, cfg=SimpleNamespace(physics_prim_path="/physicsScene")),
    )
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [])
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [])
    monkeypatch.setattr(NewtonManager, "_cl_inject_sites", lambda *_: ({}, {}, {}))
    monkeypatch.setattr(replicate_module, "replace_newton_builder_shape_colors", lambda *_: None)
    monkeypatch.setattr(newton_clone_utils_module, "replace_newton_builder_shape_colors", lambda *_: None)


def _replicate(
    collision_filter: NewtonCollisionFilter,
    sources: tuple[str, ...],
    destinations: tuple[str, ...],
    mapping: np.ndarray,
    source_builders: dict[str, newton.ModelBuilder],
) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    source_shapes = {
        source: collider_shape_map(
            source_builder, {path: index for index, path in enumerate(source_builder.shape_label)}
        )
        for source, source_builder in source_builders.items()
    }
    source_endpoint_shapes = {
        source: collision_endpoint_shape_map(
            source_builder,
            source_shapes[source],
            {path: index for index, path in enumerate(source_builder.body_label)},
        )
        for source, source_builder in source_builders.items()
    }
    collision_filter.prepare(
        builder,
        {},
        source_builders,
        source_shapes,
        source_endpoint_shapes=source_endpoint_shapes,
    )
    num_worlds = mapping.shape[1]
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        sources,
        mapping,
        np.zeros((num_worlds, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (num_worlds, 1)),
        source_builders,
        destinations=destinations,
        env_ids=np.arange(num_worlds, dtype=np.int64),
    )
    collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)
    return builder


def test_three_groups_filter_only_robot_support_per_environment():
    sources = tuple(f"/World/envs/env_0/{name}" for name in ("Robot", "Object", "Support"))
    destinations = tuple(f"/World/envs/env_{{}}/{name}" for name in ("Robot", "Object", "Support"))
    source_builders = {source: _builder_with_colliders(f"{source}/collider") for source in sources}
    mapping = np.ones((len(sources), 2), dtype=np.bool_)
    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",), filtered_groups=("supports",)
            ),
            "objects": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Object/collider",)),
            "supports": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/collider",)),
        }
    )
    collision_filter = NewtonCollisionFilter(
        cfg, sources, destinations, np.arange(2, dtype=np.int64), mapping, "/World/envs/env_{}"
    )

    builder = _replicate(collision_filter, sources, destinations, mapping, source_builders)

    denied_labels = {
        tuple(sorted((builder.shape_label[first], builder.shape_label[second])))
        for first, second in builder.shape_collision_filter_pairs
    }
    assert denied_labels == {
        tuple(sorted((f"/World/envs/env_{env}/Robot/collider", f"/World/envs/env_{env}/Support/collider")))
        for env in range(2)
    }


def test_homogeneous_source_filters_replicate_all_generated_collider_shapes_once():
    source = "/World/cells/cell_0"
    destination = "/World/cells/cell_{}"
    source_builder = _builder_with_colliders(f"{source}/Robot/collider", f"{source}/Support/collider")
    robot_primary, support = 0, 1
    robot_body = source_builder.shape_body[robot_primary]
    robot_part = source_builder.add_shape_box(
        robot_body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        label=f"{source}/Robot/collider_convex_1",
    )
    source_builder.add_shape_collision_filter_pair(robot_primary, support)
    source_shapes = {
        source: collider_shape_map(
            source_builder,
            {
                f"{source}/Robot/collider": robot_primary,
                f"{source}/Support/collider": support,
            },
        )
    }
    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",), filtered_groups=("support",)
            ),
            "support": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/collider",)),
        }
    )
    mapping = np.ones((1, 3), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        cfg,
        (source,),
        (destination,),
        np.arange(3, dtype=np.int64),
        mapping,
        "/World/cells/cell_{}",
    )
    builder = newton.ModelBuilder()

    collision_filter.prepare(builder, {}, {source: source_builder}, source_shapes)
    assert set(source_builder.shape_collision_filter_pairs) == {
        (robot_primary, support),
        (robot_primary, robot_part),
        (support, robot_part),
    }
    with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
        *_, source_shape_offsets = replicate_builder_mapping(
            builder,
            (source,),
            mapping,
            np.zeros((3, 3), dtype=np.float32),
            np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (3, 1)),
            {source: source_builder},
            destinations=(destination,),
            env_ids=np.arange(3, dtype=np.int64),
        )
    collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    replicate.assert_called_once()
    assert len(builder.shape_collision_filter_pairs) == 9
    assert len(builder.shape_collision_filter_pairs) == len(set(builder.shape_collision_filter_pairs))
    assert all(
        builder.shape_world[first] == builder.shape_world[second]
        for first, second in builder.shape_collision_filter_pairs
    )


def test_homogeneous_policy_compiles_once_and_skips_final_many_world_expansion():
    source = "/World/envs/env_0"
    destination = "/World/envs/env_{}"
    source_builder = _builder_with_colliders(f"{source}/Robot/collider", f"{source}/Support/collider")
    source_shapes = {
        source: collider_shape_map(
            source_builder,
            {f"{source}/Robot/collider": 0, f"{source}/Support/collider": 1},
        )
    }
    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",), filtered_groups=("support",)
            ),
            "support": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/collider",)),
        }
    )
    num_worlds = 2048
    mapping = np.ones((1, num_worlds), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        cfg,
        (source,),
        (destination,),
        np.arange(num_worlds, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
    )
    builder = newton.ModelBuilder()
    assert collision_filter._compiler is not None

    with mock.patch.object(collision_filter._compiler, "pairs", wraps=collision_filter._compiler.pairs) as pairs:
        collision_filter.prepare(builder, {}, {source: source_builder}, source_shapes)
    pairs.assert_called_once()
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        (source,),
        mapping,
        np.zeros((num_worlds, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (num_worlds, 1)),
        {source: source_builder},
        destinations=(destination,),
        env_ids=np.arange(num_worlds, dtype=np.int64),
    )
    with mock.patch(
        "isaaclab_newton.cloner.collision_filter._rebase_path",
        side_effect=AssertionError("prototype-complete policy must not build final per-world maps"),
    ) as rebase:
        collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    rebase.assert_not_called()
    assert len(builder.shape_collision_filter_pairs) == num_worlds


def test_final_pair_insertion_deduplicates_pairs_added_by_world_hooks():
    sources = ("/World/envs/env_0/A", "/World/envs/env_0/B")
    destinations = ("/World/envs/env_{}/A", "/World/envs/env_{}/B")
    source_builders = {source: _builder_with_colliders(f"{source}/collider") for source in sources}
    source_shapes = {
        source: collider_shape_map(source_builder, {f"{source}/collider": 0})
        for source, source_builder in source_builders.items()
    }
    cfg = CollisionFilterCfg(
        groups={
            "a": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/A/collider",), filtered_groups=("b",)),
            "b": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/B/collider",)),
        }
    )
    mapping = np.ones((2, 1), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        cfg, sources, destinations, np.asarray((0,), dtype=np.int64), mapping, "/World/envs/env_{}"
    )
    builder = newton.ModelBuilder()
    collision_filter.prepare(builder, {}, source_builders, source_shapes)
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        sources,
        mapping,
        np.zeros((1, 3), dtype=np.float32),
        np.asarray(((0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
        source_builders,
        destinations=destinations,
        env_ids=np.asarray((0,), dtype=np.int64),
    )
    existing_pair = tuple(sorted((source_shape_offsets[0, 0], source_shape_offsets[1, 0])))
    builder.add_shape_collision_filter_pair(*existing_pair)

    with mock.patch.object(
        builder, "add_shape_collision_filter_pair", wraps=builder.add_shape_collision_filter_pair
    ) as add:
        collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    add.assert_not_called()
    assert builder.shape_collision_filter_pairs.count(existing_pair) == 1


def test_inverted_group_filters_ungrouped_colliders():
    source = "/World/envs/env_0"
    destination = "/World/envs/env_{}"
    source_builder = _builder_with_colliders(
        f"{source}/Tool/collider", f"{source}/Other/collider", f"{source}/Overlap/collider"
    )
    mapping = np.ones((1, 1), dtype=np.bool_)
    cfg = CollisionFilterCfg(
        groups={
            "tool": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Tool/collider",),
                filtered_groups=("allowed",),
                invert_filtered_groups=True,
            ),
            "allowed": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/(Allowed|Overlap)/.*",)),
            "unlisted": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Overlap/.*",)),
        }
    )
    collision_filter = NewtonCollisionFilter(
        cfg, (source,), (destination,), np.asarray((0,), dtype=np.int64), mapping, "/World/envs/env_{}"
    )

    builder = _replicate(collision_filter, (source,), (destination,), mapping, {source: source_builder})

    assert builder.shape_collision_filter_pairs == [(0, 1), (0, 2)]


def test_inverted_group_filters_unlabeled_default_collider():
    source = "/World/envs/env_0/Tool"
    destination = "/World/envs/env_{}/Tool"
    source_builder = _builder_with_colliders(f"{source}/collider")
    global_builder = newton.ModelBuilder()
    global_builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
    mapping = np.ones((1, 1), dtype=np.bool_)
    cfg = CollisionFilterCfg(
        groups={
            "tool": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Tool/collider",),
                invert_filtered_groups=True,
            )
        }
    )
    collision_filter = NewtonCollisionFilter(
        cfg,
        (source,),
        (destination,),
        np.asarray((0,), dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
    )
    source_shapes = {source: collider_shape_map(source_builder, {f"{source}/collider": 0})}

    collision_filter.prepare(global_builder, {}, {source: source_builder}, source_shapes)
    *_, source_shape_offsets = replicate_builder_mapping(
        global_builder,
        (source,),
        mapping,
        np.zeros((1, 3), dtype=np.float32),
        np.asarray(((0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
        {source: source_builder},
        destinations=(destination,),
        env_ids=np.asarray((0,), dtype=np.int64),
    )
    collision_filter.apply_to_replicated_builder(global_builder, source_shape_offsets)

    assert global_builder.shape_collision_filter_pairs == [(0, 1)]


def test_inverted_group_keeps_unlabeled_source_filter_compact():
    source = "/World/envs/env_0/Tool"
    destination = "/World/envs/env_{}/Tool"
    source_builder = _builder_with_colliders(f"{source}/selected")
    source_builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
    mapping = np.ones((1, 2), dtype=np.bool_)
    cfg = CollisionFilterCfg(
        groups={
            "selected": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Tool/selected",),
                invert_filtered_groups=True,
            )
        }
    )
    collision_filter = NewtonCollisionFilter(
        cfg,
        (source,),
        (destination,),
        np.arange(2, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
    )
    source_shapes = {source: collider_shape_map(source_builder, {f"{source}/selected": 0})}
    builder = newton.ModelBuilder()

    collision_filter.prepare(builder, {}, {source: source_builder}, source_shapes)
    assert source_builder.shape_collision_filter_pairs == [(0, 1)]
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        (source,),
        mapping,
        np.zeros((2, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (2, 1)),
        {source: source_builder},
        destinations=(destination,),
        env_ids=np.arange(2, dtype=np.int64),
    )
    compact_pairs = list(builder.shape_collision_filter_pairs)
    assert collision_filter._compiler is not None
    with (
        mock.patch.object(
            collision_filter._compiler._policy,
            "filters",
            wraps=collision_filter._compiler._policy.filters,
        ) as filters,
        mock.patch.object(
            builder,
            "add_shape_collision_filter_pair",
            wraps=builder.add_shape_collision_filter_pair,
        ) as add,
    ):
        collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    filters.assert_not_called()
    add.assert_not_called()
    assert builder.shape_collision_filter_pairs == compact_pairs
    assert compact_pairs == [(0, 1), (2, 3)]


def test_source_local_authored_pair_is_replicated_from_one_prototype_pair():
    source = "/World/envs/env_0"
    destination = "/World/envs/env_{}"
    source_builder = _builder_with_colliders(f"{source}/A/collider", f"{source}/B/collider")
    source_shapes = {
        source: collider_shape_map(
            source_builder,
            {f"{source}/A/collider": 0, f"{source}/B/collider": 1},
        )
    }
    source_endpoint_shapes = {
        source: collision_endpoint_shape_map(
            source_builder,
            source_shapes[source],
            {f"{source}/A": 0, f"{source}/B": 1},
        )
    }
    num_worlds = 4096
    mapping = np.ones((1, num_worlds), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        None,
        (source,),
        (destination,),
        np.arange(num_worlds, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
        _authored((f"{source}/A", f"{source}/B")),
    )
    builder = newton.ModelBuilder()

    collision_filter.prepare(
        builder,
        {},
        {source: source_builder},
        source_shapes,
        source_endpoint_shapes=source_endpoint_shapes,
    )
    assert source_builder.shape_collision_filter_pairs == [(0, 1)]
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        (source,),
        mapping,
        np.zeros((num_worlds, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (num_worlds, 1)),
        {source: source_builder},
        destinations=(destination,),
        env_ids=np.arange(num_worlds, dtype=np.int64),
    )
    with (
        mock.patch(
            "isaaclab_newton.cloner.collision_filter._rebase_path",
            side_effect=AssertionError("fully preapplied filters must not expand per-copy endpoint maps"),
        ) as rebase,
        mock.patch.object(
            builder, "add_shape_collision_filter_pair", wraps=builder.add_shape_collision_filter_pair
        ) as add,
    ):
        collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    rebase.assert_not_called()
    add.assert_not_called()
    assert len(builder.shape_collision_filter_pairs) == num_worlds


def test_authored_snapshot_is_root_restricted_and_preserves_relationship_direction():
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    source = "/World/envs/env_0/Source"
    target = "/World/Outside/Hierarchy"
    unrelated = "/World/envs/env_1/Unrelated"
    UsdGeom.Xform.Define(stage, source)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(source))
    source_collider = UsdGeom.Cube.Define(stage, f"{source}/collider")
    UsdPhysics.CollisionAPI.Apply(source_collider.GetPrim())
    UsdGeom.Xform.Define(stage, target)
    UsdPhysics.FilteredPairsAPI.Apply(source_collider.GetPrim()).CreateFilteredPairsRel().AddTarget(Sdf.Path(target))
    unrelated_collider = UsdGeom.Cube.Define(stage, f"{unrelated}/collider")
    UsdPhysics.CollisionAPI.Apply(unrelated_collider.GetPrim())
    UsdPhysics.FilteredPairsAPI.Apply(unrelated_collider.GetPrim()).CreateFilteredPairsRel().AddTarget(Sdf.Path(source))
    deformable = stage.DefinePrim(f"{source}/Deformable", "Xform")
    deformable.AddAppliedSchema("PhysicsDeformableBodyAPI")
    cable = UsdGeom.BasisCurves.Define(stage, f"{source}/Deformable/Cable").GetPrim()
    cable.AddAppliedSchema("PhysicsCurvesDeformableSimAPI")

    snapshot = snapshot_authored_collision_filter(stage, (source, f"{source}/Deformable"))

    assert snapshot.filtered_pairs == frozenset({(f"{source}/collider", target)})
    assert snapshot.articulation_paths == frozenset({source})
    assert snapshot.deformable_owner_simulations == ((f"{source}/Deformable", (f"{source}/Deformable/Cable",)),)


def test_instanceable_authored_filtered_pairs_work_without_collision_filter_cfg(monkeypatch):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    asset_stage = Usd.Stage.CreateInMemory()
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset_root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset_root.GetPrim())
    for link_name in ("base", "link"):
        link_path = f"/Asset/{link_name}"
        link = UsdGeom.Xform.Define(asset_stage, link_path)
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        collider = UsdGeom.Cube.Define(asset_stage, f"{link_path}/collider")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(asset_stage, "/Asset/joint")
    joint.CreateBody0Rel().SetTargets((Sdf.Path("/Asset/base"),))
    joint.CreateBody1Rel().SetTargets((Sdf.Path("/Asset/link"),))

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    sources = tuple(f"/World/envs/env_0/{name}" for name in ("A", "B"))
    destinations = tuple(f"/World/envs/env_{{}}/{name}" for name in ("A", "B"))
    for source in sources:
        prim = stage.DefinePrim(source, "Xform")
        prim.GetReferences().AddReference(asset_stage.GetRootLayer().identifier, "/Asset")
        prim.SetInstanceable(True)
    assert stage.GetPrimAtPath(f"{sources[0]}/base").IsInstanceProxy()
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(sources[0])).CreateFilteredPairsRel().AddTarget(
        Sdf.Path(sources[1])
    )
    _configure_replicate_test_manager(monkeypatch)

    builder, *_ = replicate_module._build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        destinations=destinations,
        env_ids=np.arange(2, dtype=np.int64),
        mapping=np.ones((2, 2), dtype=np.bool_),
        load_visual_shapes=False,
        collision_filter_cfg=None,
    )

    denied = {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }
    cross_asset = {
        pair for pair in denied if any("/A/" in path for path in pair) and any("/B/" in path for path in pair)
    }
    assert cross_asset == {
        frozenset(
            (
                f"/World/envs/env_{env}/A/{first_link}/collider",
                f"/World/envs/env_{env}/B/{second_link}/collider",
            )
        )
        for env in range(2)
        for first_link in ("base", "link")
        for second_link in ("base", "link")
    }


def test_partition_importer_warnings_are_deferred_to_one_final_diagnostic(monkeypatch):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    sources = ("/World/envs/env_0/A", "/World/envs/env_0/B")
    destinations = ("/World/envs/env_{}/A", "/World/envs/env_{}/B")
    colliders = []
    for source in sources:
        body = UsdGeom.Xform.Define(stage, source)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        collider = UsdGeom.Cube.Define(stage, f"{source}/collider")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        colliders.append(collider.GetPrim())
    relationship = UsdPhysics.FilteredPairsAPI.Apply(colliders[0]).CreateFilteredPairsRel()
    relationship.AddTarget(Sdf.Path(f"{sources[1]}/collider"))
    relationship.AddTarget(Sdf.Path("/World/Missing"))
    _configure_replicate_test_manager(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder, *_ = replicate_module._build_newton_builder_from_mapping(
            stage=stage,
            sources=sources,
            destinations=destinations,
            env_ids=np.arange(2, dtype=np.int64),
            mapping=np.ones((2, 2), dtype=np.bool_),
            load_visual_shapes=False,
            collision_filter_cfg=None,
        )

    filtered_pair_warnings = [str(item.message) for item in caught if "physics:filteredPairs" in str(item.message)]
    assert filtered_pair_warnings == [
        f"{sources[0]}/collider -> /World/Missing: physics:filteredPairs was not applied by Newton because "
        "the path does not exist."
    ]
    denied = {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }
    assert {
        frozenset(
            (
                f"/World/envs/env_{env}/A/collider",
                f"/World/envs/env_{env}/B/collider",
            )
        )
        for env in range(2)
    }.issubset(denied)


def test_filtered_pair_body_endpoint_owns_only_direct_shapes():
    builder = newton.ModelBuilder()
    parent_path = "/World/Parent"
    child_path = f"{parent_path}/Child"
    other_path = "/World/Other"
    parent = builder.add_body(label=parent_path)
    child = builder.add_body(label=child_path)
    other = builder.add_body(label=other_path)
    parent_shape = builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1, label=f"{parent_path}/collider")
    child_shape = builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1, label=f"{child_path}/collider")
    other_shape = builder.add_shape_box(other, hx=0.1, hy=0.1, hz=0.1, label=f"{other_path}/collider")
    collider_shapes = collider_shape_map(
        builder,
        {
            f"{parent_path}/collider": parent_shape,
            f"{child_path}/collider": child_shape,
            f"{other_path}/collider": other_shape,
        },
    )
    endpoint_shapes = collision_endpoint_shape_map(
        builder,
        collider_shapes,
        {parent_path: parent, child_path: child, other_path: other},
    )
    collision_filter = NewtonCollisionFilter(
        None,
        (),
        (),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.bool_),
        "/World/envs/env_{}",
        _authored((parent_path, other_path), (f"{parent_path}/PlainXform", other_path)),
    )

    collision_filter.prepare(builder, collider_shapes, {}, {}, endpoint_shapes, {})
    with pytest.warns(UserWarning, match="source did not produce a supported collider"):
        collision_filter.apply_to_replicated_builder(builder, {})

    assert endpoint_shapes[parent_path] == (parent_shape,)
    assert builder.shape_collision_filter_pairs == [(parent_shape, other_shape)]
    assert (child_shape, other_shape) not in builder.shape_collision_filter_pairs


def test_filtered_pair_target_hierarchy_includes_nested_rigid_bodies():
    builder = newton.ModelBuilder()
    hierarchy = "/World/Hierarchy"
    parent_path = f"{hierarchy}/Parent"
    child_path = f"{parent_path}/Child"
    source_path = "/World/Source"
    parent = builder.add_body(label=parent_path)
    child = builder.add_body(label=child_path)
    source = builder.add_body(label=source_path)
    parent_shape = builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1, label=f"{parent_path}/collider")
    child_shape = builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1, label=f"{child_path}/collider")
    source_shape = builder.add_shape_box(source, hx=0.1, hy=0.1, hz=0.1, label=f"{source_path}/collider")
    collider_shapes = collider_shape_map(
        builder,
        {
            f"{parent_path}/collider": parent_shape,
            f"{child_path}/collider": child_shape,
            f"{source_path}/collider": source_shape,
        },
    )
    endpoint_shapes = collision_endpoint_shape_map(
        builder,
        collider_shapes,
        {parent_path: parent, child_path: child, source_path: source},
    )
    collision_filter = NewtonCollisionFilter(
        None,
        (),
        (),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.bool_),
        "/World/envs/env_{}",
        _authored((source_path, hierarchy)),
    )

    collision_filter.prepare(builder, collider_shapes, {}, {}, endpoint_shapes, {})

    assert set(builder.shape_collision_filter_pairs) == {
        (min(source_shape, parent_shape), max(source_shape, parent_shape)),
        (min(source_shape, child_shape), max(source_shape, child_shape)),
    }


def test_cable_and_deformable_owner_endpoints_map_to_all_segment_shapes():
    builder = newton.ModelBuilder()
    cable_path = "/World/Deformable/Cable"
    owner_path = "/World/Deformable"
    source_path = "/World/Source"
    cable_bodies = [builder.add_body(label=f"{cable_path}/segment_{index}") for index in range(2)]
    cable_shapes = [
        builder.add_shape_capsule(body, radius=0.1, half_height=0.1, label=f"{cable_path}/segment_{index}")
        for index, body in enumerate(cable_bodies)
    ]
    source = builder.add_body(label=source_path)
    source_shape = builder.add_shape_box(source, hx=0.1, hy=0.1, hz=0.1, label=f"{source_path}/collider")
    collider_shapes = collider_shape_map(builder, {f"{source_path}/collider": source_shape})
    endpoint_shapes = collision_endpoint_shape_map(
        builder,
        collider_shapes,
        {source_path: source},
        path_cable_map={cable_path: (cable_bodies, [])},
        deformable_owner_simulations={owner_path: (cable_path,), "/World/Other": ("/World/Other/Cable",)},
    )
    collision_filter = NewtonCollisionFilter(
        None,
        (),
        (),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.bool_),
        "/World/envs/env_{}",
        _authored((source_path, owner_path)),
    )

    collision_filter.prepare(builder, collider_shapes, {}, {}, endpoint_shapes, {})

    assert endpoint_shapes[cable_path] == tuple(cable_shapes)
    assert endpoint_shapes[owner_path] == tuple(cable_shapes)
    assert "/World/Other" not in endpoint_shapes
    assert set(builder.shape_collision_filter_pairs) == {
        (min(source_shape, shape), max(source_shape, shape)) for shape in cable_shapes
    }


def test_particle_deformable_filtered_pair_emits_one_targeted_final_warning():
    builder = _builder_with_colliders("/World/Source/collider")
    collider_shapes = collider_shape_map(builder, {"/World/Source/collider": 0})
    endpoint_shapes = collision_endpoint_shape_map(builder, collider_shapes, {"/World/Source": 0})
    collision_filter = NewtonCollisionFilter(
        None,
        (),
        (),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.bool_),
        "/World/envs/env_{}",
        _authored(
            ("/World/Source", "/World/Cloth"),
            unsupported_endpoint_reasons=(
                ("/World/Cloth", "cloth particle deformables cannot be represented by shape filter pairs"),
            ),
        ),
    )

    collision_filter.prepare(builder, collider_shapes, {}, {}, endpoint_shapes, {})
    with pytest.warns(UserWarning, match="cloth particle deformables cannot be represented") as caught:
        collision_filter.apply_to_replicated_builder(builder, {})

    assert len(caught) == 1
    assert builder.shape_collision_filter_pairs == []


def test_filtered_pair_articulation_endpoint_owns_all_link_shapes():
    builder = newton.ModelBuilder()
    robot_path = "/World/Robot"
    first = builder.add_link(label=f"{robot_path}/First")
    second = builder.add_link(label=f"{robot_path}/Second")
    other = builder.add_body(label="/World/Other")
    root_joint = builder.add_joint_free(child=first)
    link_joint = builder.add_joint_fixed(parent=first, child=second)
    builder.add_articulation([root_joint, link_joint], label=robot_path)
    first_shape = builder.add_shape_box(first, hx=0.1, hy=0.1, hz=0.1, label=f"{robot_path}/First/collider")
    second_shape = builder.add_shape_box(second, hx=0.1, hy=0.1, hz=0.1, label=f"{robot_path}/Second/collider")
    other_shape = builder.add_shape_box(other, hx=0.1, hy=0.1, hz=0.1, label="/World/Other/collider")
    collider_shapes = collider_shape_map(
        builder,
        {
            f"{robot_path}/First/collider": first_shape,
            f"{robot_path}/Second/collider": second_shape,
            "/World/Other/collider": other_shape,
        },
    )
    endpoint_shapes = collision_endpoint_shape_map(
        builder,
        collider_shapes,
        {f"{robot_path}/First": first, f"{robot_path}/Second": second, "/World/Other": other},
        articulation_paths=(robot_path,),
    )
    collision_filter = NewtonCollisionFilter(
        None,
        (),
        (),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=np.bool_),
        "/World/envs/env_{}",
        _authored((robot_path, "/World/Other")),
    )

    existing_pairs = set(builder.shape_collision_filter_pairs)
    collision_filter.prepare(builder, collider_shapes, {}, {}, endpoint_shapes, {})

    assert endpoint_shapes[robot_path] == (first_shape, second_shape)
    assert set(builder.shape_collision_filter_pairs) - existing_pairs == {
        (first_shape, other_shape),
        (second_shape, other_shape),
    }


def test_authored_filtered_pairs_join_only_common_cross_owner_worlds():
    sources = ("/World/envs/env_0/A", "/World/envs/env_0/B")
    destinations = ("/World/envs/env_{}/A", "/World/envs/env_{}/B")
    source_builders = {source: _builder_with_colliders(f"{source}/collider") for source in sources}
    mapping = np.asarray(((True, True, False), (False, True, True)), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        None,
        sources,
        destinations,
        np.arange(3, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
        _authored(sources),
    )

    builder = _replicate(collision_filter, sources, destinations, mapping, source_builders)

    denied_labels = {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }
    assert denied_labels == {frozenset(("/World/envs/env_1/A/collider", "/World/envs/env_1/B/collider"))}


def test_filtered_pair_hierarchy_target_spans_clone_owners_in_every_world():
    sources = ("/World/envs/env_0/A", "/World/envs/env_0/B")
    destinations = ("/World/envs/env_{}/A", "/World/envs/env_{}/B")
    source_builders = {source: _builder_with_colliders(f"{source}/collider") for source in sources}
    mapping = np.ones((2, 2), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        None,
        sources,
        destinations,
        np.arange(2, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
        _authored((sources[0], "/World/envs/env_0")),
    )

    builder = _replicate(collision_filter, sources, destinations, mapping, source_builders)

    denied_labels = {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }
    assert denied_labels == {
        frozenset((f"/World/envs/env_{env}/A/collider", f"/World/envs/env_{env}/B/collider")) for env in range(2)
    }


def test_destination_specific_filtered_pair_is_not_misclassified_as_global():
    source = "/World/envs/env_0"
    destination = "/World/envs/env_{}"
    source_builder = _builder_with_colliders(f"{source}/A/collider", f"{source}/B/collider")
    source_shapes = {
        source: collider_shape_map(
            source_builder,
            {f"{source}/A/collider": 0, f"{source}/B/collider": 1},
        )
    }
    source_endpoint_shapes = {
        source: collision_endpoint_shape_map(
            source_builder,
            source_shapes[source],
            {f"{source}/A": 0, f"{source}/B": 1},
        )
    }
    mapping = np.ones((1, 2), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        None,
        (source,),
        (destination,),
        np.arange(2, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
        _authored(("/World/envs/env_1/A", "/World/envs/env_1/B")),
    )
    builder = newton.ModelBuilder()

    collision_filter.prepare(
        builder,
        {},
        {source: source_builder},
        source_shapes,
        source_endpoint_shapes=source_endpoint_shapes,
    )
    assert source_builder.shape_collision_filter_pairs == []
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        (source,),
        mapping,
        np.zeros((2, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (2, 1)),
        {source: source_builder},
        destinations=(destination,),
        env_ids=np.arange(2, dtype=np.int64),
    )
    collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    assert [
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    ] == [frozenset(("/World/envs/env_1/A/collider", "/World/envs/env_1/B/collider"))]


def test_authored_filtered_pairs_survive_local_to_global_import_partition():
    source = "/World/envs/env_0/Robot"
    destination = "/World/envs/env_{}/Robot"
    source_builder = _builder_with_colliders(f"{source}/collider")
    builder = _builder_with_colliders("/World/Ground/collider")
    mapping = np.ones((1, 2), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        CollisionFilterCfg(),
        (source,),
        (destination,),
        np.arange(2, dtype=np.int64),
        mapping,
        "/World/envs/env_{}",
        _authored((source, "/World/Ground")),
    )
    source_shapes = {source: collider_shape_map(source_builder, {f"{source}/collider": 0})}
    global_shapes = collider_shape_map(builder, {"/World/Ground/collider": 0})
    source_endpoint_shapes = {source: collision_endpoint_shape_map(source_builder, source_shapes[source], {source: 0})}
    global_endpoint_shapes = collision_endpoint_shape_map(builder, global_shapes, {"/World/Ground": 0})

    collision_filter.prepare(
        builder,
        global_shapes,
        {source: source_builder},
        source_shapes,
        global_endpoint_shapes,
        source_endpoint_shapes,
    )
    *_, source_shape_offsets = replicate_builder_mapping(
        builder,
        (source,),
        mapping,
        np.zeros((2, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (2, 1)),
        {source: source_builder},
        destinations=(destination,),
        env_ids=np.arange(2, dtype=np.int64),
    )
    collision_filter.apply_to_replicated_builder(builder, source_shape_offsets)

    denied_labels = {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }
    assert denied_labels == {
        frozenset(("/World/Ground/collider", f"/World/envs/env_{env}/Robot/collider")) for env in range(2)
    }


def test_multiple_collider_representations_can_select_exact_interactions():
    source = "/World/envs/env_0"
    destination = "/World/envs/env_{}"
    relative_paths = (
        "Nut/mesh/colliders/sdf",
        "Nut/mesh/colliders/convex",
        "Bolt/mesh/colliders/sdf",
        "Bolt/mesh/colliders/convex",
        "Other/mesh/colliders/convex",
    )
    source_builder = _builder_with_colliders(*(f"{source}/{path}" for path in relative_paths))
    mapping = np.ones((1, 1), dtype=np.bool_)
    cfg = CollisionFilterCfg(
        groups={
            "nut_sdf": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Nut/mesh/colliders/sdf",),
                filtered_groups=("bolt_sdf",),
                invert_filtered_groups=True,
            ),
            "bolt_sdf": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Bolt/mesh/colliders/sdf",),
                filtered_groups=("nut_sdf",),
                invert_filtered_groups=True,
            ),
            "nut_convex": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Nut/mesh/colliders/convex",),
                filtered_groups=("bolt_convex",),
            ),
            "bolt_convex": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Bolt/mesh/colliders/convex",),
            ),
        }
    )
    collision_filter = NewtonCollisionFilter(
        cfg, (source,), (destination,), np.asarray((0,), dtype=np.int64), mapping, "/World/envs/env_{}"
    )

    builder = _replicate(collision_filter, (source,), (destination,), mapping, {source: source_builder})

    denied = {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }

    def path(name: str) -> str:
        return f"/World/envs/env_0/{name}"

    assert denied == {
        frozenset((path("Nut/mesh/colliders/sdf"), path("Nut/mesh/colliders/convex"))),
        frozenset((path("Nut/mesh/colliders/sdf"), path("Bolt/mesh/colliders/convex"))),
        frozenset((path("Nut/mesh/colliders/sdf"), path("Other/mesh/colliders/convex"))),
        frozenset((path("Bolt/mesh/colliders/sdf"), path("Nut/mesh/colliders/convex"))),
        frozenset((path("Bolt/mesh/colliders/sdf"), path("Bolt/mesh/colliders/convex"))),
        frozenset((path("Bolt/mesh/colliders/sdf"), path("Other/mesh/colliders/convex"))),
        frozenset((path("Nut/mesh/colliders/convex"), path("Bolt/mesh/colliders/convex"))),
    }
    assert frozenset((path("Nut/mesh/colliders/sdf"), path("Bolt/mesh/colliders/sdf"))) not in denied
    assert frozenset((path("Nut/mesh/colliders/convex"), path("Other/mesh/colliders/convex"))) not in denied
    assert frozenset((path("Bolt/mesh/colliders/convex"), path("Other/mesh/colliders/convex"))) not in denied


def test_newton_rejects_disabling_environment_isolation():
    context_type = object
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
        context_rows={context_type: (0,)},
    )
    with mock.patch.object(NewtonManager, "clone_context_type", context_type):
        with pytest.raises(NotImplementedError, match="shape_world partitions"):
            NewtonManager._apply_collision_filter_impl(plan, None, isolate_environments=False, replicate_physics=True)


def test_newton_rejects_collision_filtering_without_native_replication():
    cfg = CollisionFilterCfg(groups={"selected": CollisionGroupCfg(prim_path_exprs=(r"/World/selected",))})
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )
    with pytest.raises(NotImplementedError, match="require replicate_physics=True"):
        NewtonManager._apply_collision_filter_impl(plan, cfg, isolate_environments=True, replicate_physics=False)


def test_newton_rejects_isolation_without_native_replication():
    context_type = object
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
        context_rows={context_type: (0,)},
    )

    with mock.patch.object(NewtonManager, "clone_context_type", context_type):
        with pytest.raises(NotImplementedError, match="environment isolation.*require"):
            NewtonManager._apply_collision_filter_impl(plan, None, isolate_environments=True, replicate_physics=False)


def test_newton_allows_usd_only_isolation_without_native_replication():
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )

    NewtonManager._apply_collision_filter_impl(plan, None, isolate_environments=True, replicate_physics=False)


def test_newton_rejects_policy_without_native_context_rows():
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )
    cfg = CollisionFilterCfg(groups={"selected": CollisionGroupCfg(prim_path_exprs=(r"/World/selected",))})

    with pytest.raises(NotImplementedError, match="populated NewtonReplicateContext rows"):
        NewtonManager._apply_collision_filter_impl(plan, cfg, isolate_environments=True, replicate_physics=True)


def test_newton_close_clears_collision_plan_when_stop_callback_fails(monkeypatch):
    plan = object()
    monkeypatch.setattr(NewtonManager, "_cl_collision_filter_plan", plan)

    def clear():
        NewtonManager._cl_collision_filter_plan = None

    with (
        mock.patch.object(PhysicsManager, "close", side_effect=RuntimeError("STOP listener failed")),
        mock.patch.object(NewtonManager, "clear", side_effect=clear) as clear_mock,
        pytest.raises(RuntimeError, match="STOP listener failed"),
    ):
        NewtonManager.close()

    clear_mock.assert_called_once_with()
    assert NewtonManager._cl_collision_filter_plan is None
