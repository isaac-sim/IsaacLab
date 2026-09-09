# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative collision filtering in Newton replication."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import isaaclab_newton.cloner.newton_clone_utils as newton_clone_utils_module
import isaaclab_newton.cloner.replicate as replicate_module
import newton
import numpy as np
import pytest
from isaaclab_newton.cloner.collision_filter import NewtonCollisionFilter, collider_shape_map
from isaaclab_newton.cloner.newton_clone_utils import replicate_builder_mapping
from isaaclab_newton.physics import NewtonManager

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan
from isaaclab.physics import CollisionGroupCfg, PhysicsManager


def _builder_with_colliders(*paths: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    for path in paths:
        body = builder.add_body(label=path.rpartition("/")[0])
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label=path)
    return builder


def _replicate(
    groups: dict[str, CollisionGroupCfg],
    sources: tuple[str, ...],
    mapping: np.ndarray,
    source_builders: dict[str, newton.ModelBuilder],
    builder: newton.ModelBuilder | None = None,
    global_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[newton.ModelBuilder, dict[tuple[int, int], int]]:
    builder = newton.ModelBuilder() if builder is None else builder
    destinations = tuple(source.replace("env_0", "env_{}") for source in sources)
    collision_filter = NewtonCollisionFilter(
        groups, sources, destinations, np.arange(mapping.shape[1]), mapping, "/World/envs/env_{}"
    )
    source_shapes = {
        source: collider_shape_map(
            source_builder,
            {
                path: index
                for index, path in enumerate(source_builder.shape_label)
                if isinstance(path, str) and path.startswith("/")
            },
        )
        for source, source_builder in source_builders.items()
    }
    collision_filter.prepare(builder, global_shapes or {}, source_builders, source_shapes)
    num_worlds = mapping.shape[1]
    *_, shape_offsets = replicate_builder_mapping(
        builder,
        sources,
        mapping,
        np.zeros((num_worlds, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (num_worlds, 1)),
        source_builders,
        destinations=destinations,
        env_ids=np.arange(num_worlds, dtype=np.int64),
    )
    collision_filter.apply_to_replicated_builder(builder, shape_offsets)
    return builder, shape_offsets


def _denied_labels(builder: newton.ModelBuilder) -> set[frozenset[str]]:
    return {
        frozenset((builder.shape_label[first], builder.shape_label[second]))
        for first, second in builder.shape_collision_filter_pairs
    }


def _configure_replicate_test_manager(monkeypatch) -> None:
    manager = SimpleNamespace(
        create_builder=lambda up_axis="Z": newton.ModelBuilder(up_axis=up_axis),
        _get_usd_import_schema_resolvers=lambda: [],
        _inject_terrain_heightfields=lambda stage, builder, root_paths: [],
    )

    monkeypatch.setattr(
        PhysicsManager,
        "_sim",
        SimpleNamespace(physics_manager=manager, cfg=SimpleNamespace(physics_prim_path="/physicsScene")),
    )
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [])
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [])
    monkeypatch.setattr(NewtonManager, "_cl_inject_sites", lambda *_: ({}, {}, {}))
    monkeypatch.setattr(replicate_module, "replace_newton_builder_shape_colors", lambda *_: None)
    monkeypatch.setattr(newton_clone_utils_module, "replace_newton_builder_shape_colors", lambda *_: None)


def test_three_groups_filter_only_robot_support_per_environment():
    sources = tuple(f"/World/envs/env_0/{name}" for name in ("Robot", "Object", "Support"))
    source_builders = {source: _builder_with_colliders(f"{source}/collider") for source in sources}
    mapping = np.ones((3, 2), dtype=np.bool_)
    groups = {
        "robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",), filtered_groups=("supports",)),
        "objects": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Object/collider",)),
        "supports": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/collider",)),
    }
    builder, _ = _replicate(groups, sources, mapping, source_builders)

    assert _denied_labels(builder) == {
        frozenset((f"/World/envs/env_{env}/Robot/collider", f"/World/envs/env_{env}/Support/collider"))
        for env in range(2)
    }


def test_generated_collider_shapes_use_compact_homogeneous_filtering():
    source = "/World/envs/env_0"
    source_builder = _builder_with_colliders(f"{source}/Robot/collider", f"{source}/Support/collider")
    robot_primary, support = 0, 1
    robot_part = source_builder.add_shape_box(
        source_builder.shape_body[robot_primary],
        hx=0.1,
        hy=0.1,
        hz=0.1,
        label=f"{source}/Robot/collider_convex_1",
    )
    source_shapes = {
        source: collider_shape_map(
            source_builder,
            {f"{source}/Robot/collider": robot_primary, f"{source}/Support/collider": support},
        )
    }
    groups = {
        "robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/collider",), filtered_groups=("support",)),
        "support": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/collider",)),
    }
    num_worlds = 1024
    mapping = np.ones((1, num_worlds), dtype=np.bool_)
    collision_filter = NewtonCollisionFilter(
        groups, (source,), ("/World/envs/env_{}",), np.arange(num_worlds), mapping, "/World/envs/env_{}"
    )
    builder = newton.ModelBuilder()
    collision_filter.prepare(builder, {}, {source: source_builder}, source_shapes)

    assert source_shapes[source][f"{source}/Robot/collider"] == (robot_primary, robot_part)
    local_pairs = set(source_builder.shape_collision_filter_pairs)
    assert {(min(shape, support), max(shape, support)) for shape in (robot_primary, robot_part)} <= local_pairs
    *_, offsets = replicate_builder_mapping(
        builder,
        (source,),
        mapping,
        np.zeros((num_worlds, 3), dtype=np.float32),
        np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (num_worlds, 1)),
        {source: source_builder},
        destinations=("/World/envs/env_{}",),
        env_ids=np.arange(num_worlds),
    )
    with mock.patch(
        "isaaclab_newton.cloner.collision_filter._rebase_path",
        side_effect=AssertionError("homogeneous policy must remain prototype-local"),
    ):
        collision_filter.apply_to_replicated_builder(builder, offsets)
    assert len(builder.shape_collision_filter_pairs) == num_worlds * len(local_pairs)


def test_inverted_group_filters_unmatched_local_and_global_shapes():
    source = "/World/envs/env_0/Tool"
    source_builder = _builder_with_colliders(f"{source}/selected")
    source_builder.add_shape_box(source_builder.add_body(), hx=0.1, hy=0.1, hz=0.1)
    global_builder = _builder_with_colliders("/World/Ground/collider")
    global_shapes = collider_shape_map(global_builder, {"/World/Ground/collider": 0})
    mapping = np.ones((1, 2), dtype=np.bool_)
    groups = {
        "selected": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Tool/selected",), invert_filtered_groups=True)
    }
    builder, offsets = _replicate(
        groups,
        (source,),
        mapping,
        {source: source_builder},
        global_builder,
        global_shapes,
    )

    expected = set()
    for world in range(2):
        selected, unmatched = offsets[0, world], offsets[0, world] + 1
        expected.update({(0, selected), (min(selected, unmatched), max(selected, unmatched))})
    assert set(builder.shape_collision_filter_pairs) == expected


def test_nut_bolt_selects_sdf_contact_and_convex_contact_with_other_objects():
    source = "/World/envs/env_0"
    relative_paths = (
        "Nut/mesh/colliders/sdf",
        "Nut/mesh/colliders/convex",
        "Bolt/mesh/colliders/sdf",
        "Bolt/mesh/colliders/convex",
        "Other/mesh/colliders/convex",
    )
    source_builder = _builder_with_colliders(*(f"{source}/{path}" for path in relative_paths))
    groups = {
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
            prim_path_exprs=(r"{ENV_REGEX_NS}/Nut/mesh/colliders/convex",), filtered_groups=("bolt_convex",)
        ),
        "bolt_convex": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Bolt/mesh/colliders/convex",)),
    }
    mapping = np.ones((1, 1), dtype=np.bool_)
    builder, _ = _replicate(groups, (source,), mapping, {source: source_builder})

    assert _denied_labels(builder) == {
        frozenset((f"{source}/{first}", f"{source}/{second}"))
        for first, second in (
            (relative_paths[0], relative_paths[1]),
            (relative_paths[0], relative_paths[3]),
            (relative_paths[0], relative_paths[4]),
            (relative_paths[2], relative_paths[1]),
            (relative_paths[2], relative_paths[3]),
            (relative_paths[2], relative_paths[4]),
            (relative_paths[1], relative_paths[3]),
        )
    }


def test_importer_filtered_pairs_remain_replicated_alongside_declarative_policy(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    source = "/World/envs/env_0"
    colliders = {}
    for name in ("A", "B", "C"):
        body = UsdGeom.Xform.Define(stage, f"{source}/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        collider = UsdGeom.Cube.Define(stage, f"{source}/{name}/collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        colliders[name] = collider
    UsdPhysics.FilteredPairsAPI.Apply(colliders["A"]).CreateFilteredPairsRel().AddTarget(colliders["B"].GetPath())
    groups = {
        "b": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/B/collider",), filtered_groups=("c",)),
        "c": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/C/collider",)),
    }
    _configure_replicate_test_manager(monkeypatch)

    builder, *_ = replicate_module._build_newton_builder_from_mapping(
        stage,
        (source,),
        ("/World/envs/env_{}",),
        np.arange(2),
        np.ones((1, 2), dtype=np.bool_),
        load_visual_shapes=False,
        collision_filter_groups=groups,
    )

    assert _denied_labels(builder) == {
        frozenset((f"/World/envs/env_{env}/{first}/collider", f"/World/envs/env_{env}/{second}/collider"))
        for env in range(2)
        for first, second in (("A", "B"), ("B", "C"))
    }


def test_newton_manager_enforces_native_collision_filter_boundaries(monkeypatch):
    context_type = object
    rows_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
        context_rows={context_type: (0,)},
    )
    empty_plan = replace(rows_plan, context_rows={})
    groups = {"selected": CollisionGroupCfg(prim_path_exprs=(r"/World/selected",))}
    monkeypatch.setattr(NewtonManager, "clone_context_type", context_type)
    monkeypatch.setattr(NewtonManager, "_cl_collision_filter_groups", None)

    NewtonManager._apply_collision_filter_impl(rows_plan, groups)
    assert NewtonManager._cl_collision_filter_groups is not groups
    assert NewtonManager._cl_collision_filter_groups == groups
    with pytest.raises(NotImplementedError, match="shape_world partitions"):
        NewtonManager._apply_collision_filter_impl(replace(rows_plan, isolate_environments=False), None)
    with pytest.raises(NotImplementedError, match="require replicate_physics=True"):
        NewtonManager._apply_collision_filter_impl(replace(rows_plan, replicate_physics=False), groups)
    with pytest.raises(NotImplementedError, match="populated NewtonReplicateContext rows"):
        NewtonManager._apply_collision_filter_impl(empty_plan, groups)
    groups.clear()
    assert set(NewtonManager._cl_collision_filter_groups) == {"selected"}
