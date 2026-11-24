# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, factorized scene combinations and the task-package ownership boundary."""

import ast
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner import make_clone_plan, make_valid_clone_combinations

from isaaclab_tasks.contrib.heterogeneous_tasks import heterogeneous_tasks_env_cfg as catalog
from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_object_spawning_reduces_collision_geometry_without_changing_visuals(tmp_path):
    """The real catalog spawner preserves the visual mesh and the hole in a reduced collision surface."""
    path = tmp_path / "ring.usda"
    source = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(source, 1.0)
    root = UsdGeom.Xform.Define(source, "/Asset").GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    ring = trimesh.creation.annulus(r_min=0.04, r_max=0.05, height=0.03, sections=64)
    visual = UsdGeom.Mesh.Define(source, "/Asset/visual")
    # Duplicated vertices model UV/normal seams in the catalog's authored visual meshes.
    visual.CreatePointsAttr(ring.vertices[ring.faces].reshape(-1, 3))
    visual.CreateFaceVertexCountsAttr(np.full(len(ring.faces), 3, dtype=np.int32))
    visual.CreateFaceVertexIndicesAttr(np.arange(ring.faces.size, dtype=np.int32))
    UsdGeom.Xformable(visual).AddTranslateOp().Set((0.03, -0.02, 0.01))
    UsdPhysics.CollisionAPI.Apply(visual.GetPrim()).CreateCollisionEnabledAttr(True)
    source.GetRootLayer().Save()
    cfg = next(iter(catalog.ASSET_CATALOG.values())).replace(usd_path=str(path))
    assert cfg.to_dict()["triangle_reduction"] == 0.75
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        cfg.func("/World/Object", cfg)
    spawned = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Object/visual"))
    collision = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Object/visual/collision"))
    np.testing.assert_array_equal(spawned.GetPointsAttr().Get(), visual.GetPointsAttr().Get())
    np.testing.assert_array_equal(spawned.GetFaceVertexIndicesAttr().Get(), visual.GetFaceVertexIndicesAttr().Get())
    assert not UsdPhysics.CollisionAPI(spawned.GetPrim()).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.CollisionAPI(collision.GetPrim()).GetCollisionEnabledAttr().Get()
    assert len(collision.GetFaceVertexIndicesAttr().Get()) <= ring.faces.size * 0.3
    assert collision.GetPrim().GetAttribute("newton:sdfTargetVoxelSize").Get() == pytest.approx(0.00075)
    visual_transform = UsdGeom.XformCache().GetLocalToWorldTransform(spawned.GetPrim())
    assert UsdGeom.XformCache().GetLocalToWorldTransform(collision.GetPrim()) == visual_transform
    vertices = np.asarray(collision.GetPointsAttr().Get())
    faces = np.asarray(collision.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    # Every triangle remains on one side of the central opening, rather than bridging it with a hull.
    assert np.linalg.norm(vertices[faces].mean(axis=1)[:, :2], axis=1).min() > 0.035


def test_bounded_combinations_do_not_expand_the_object_product():
    """A large catalog samples worlds with bounded cardinality, without requiring task categories."""
    root = Path(catalog.__file__).parent
    assert not (root / "tasks").exists() and not (root / "objects.py").exists()
    assert not (root / "tasks.yaml").exists() and not hasattr(catalog, "TASKS")
    assert not (root.parents[3] / "isaaclab/isaaclab/managers/heterogeneous_task_manager.py").exists()
    objects = {
        f"item_{i}": sim_utils.CuboidCfg(
            size=(0.02, 0.03, 0.04), semantic_tags=[("category", "common" if i < 900 else f"rare_{(i - 900) // 25}")]
        )
        for i in range(1000)
    }
    tree = ast.parse(Path(catalog.__file__).read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)), (
        "The scene must be declared through config composition, without factories or post-init rewriting"
    )
    cfg = catalog.VLAHeterogeneousEnvCfg()
    source = catalog.OBJECTS_CFG
    cfg.scene.objects = source.replace(
        rigid_objects={
            name: member.replace(
                spawn=member.spawn.replace(
                    assets_cfg=[
                        spawn.replace(
                            semantic_tags=[
                                *spawn.semantic_tags,
                                ("_catalog_key", label),
                                *dict.fromkeys(
                                    (tag, value)
                                    for candidate in member.spawn.assets_cfg[:-1]
                                    for tag, value in candidate.semantic_tags
                                    if tag == "object_count"
                                ),
                            ]
                        )
                        for label, spawn in objects.items()
                    ]
                    + [member.spawn.assets_cfg[-1].copy()]
                )
            )
            for name, member in source.rigid_objects.items()
        }
    )
    candidates = [member.spawn.assets_cfg for member in cfg.scene.objects.rigid_objects.values()]
    names = ["objects"] * len(candidates) + ["work_table"]
    candidates.append([table.spawn for table in catalog.TABLE_VARIANTS])
    groups, weights = make_valid_clone_combinations(names, candidates, cfg.scene.clone_cfg.clone_combinations)
    assert len(groups) <= 8, "Object combinations must stay factorized, rather than expanding the product"
    flattened = [candidate for row in candidates for candidate in row]
    state = np.random.get_state()
    try:
        np.random.seed(42)
        plan = make_clone_plan(
            flattened, groups, 128, weights=weights, clone_strategy=cfg.scene.clone_cfg.clone_strategy
        )
        np.random.seed(42)
        repeat = make_clone_plan(
            flattened, groups, 128, weights=weights, clone_strategy=cfg.scene.clone_cfg.clone_strategy
        )
    finally:
        np.random.set_state(state)
    topology = plan.topology
    np.testing.assert_array_equal(topology.world_prototypes, repeat.topology.world_prototypes)
    np.testing.assert_array_equal(topology.world_prototype_layout, repeat.topology.world_prototype_layout)
    assert len(topology.world_prototype_starts) - 2 <= 128
    counts = []
    chosen = set()
    category_counts = {}
    common_halves = [0, 0]
    selected_tables = set()
    for composition in topology.world_prototype_layout:
        start, end = topology.world_prototype_starts[composition + 1 : composition + 3]
        members = [
            dict(flattened[index].semantic_tags).get("_catalog_key") for index in topology.world_prototypes[start:end]
        ]
        selected_tables.update(
            flattened[index].usd_path
            for index in topology.world_prototypes[start:end]
            if dict(flattened[index].semantic_tags).get("category") == "table"
        )
        real = [name for name in members if name is not None and name != "__empty__"]
        counts.append(len(real))
        chosen.update(real)
        for name in real:
            category = dict(objects[name].semantic_tags)["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            index = int(name.removeprefix("item_"))
            if index < 900:
                common_halves[index // 450] += 1
    assert len(selected_tables) == len(catalog.TABLE_VARIANTS) > 2, "Include the table in world sampling"
    assert min(counts) == 2 and max(counts) == 8
    assert chosen <= objects.keys() and len(chosen) > 100
    # A 900-asset category must not overwhelm four categories of only 25 assets each.
    frequencies = np.array(list(category_counts.values())) / sum(counts)
    assert len(frequencies) == 5 and ((frequencies > 0.1) & (frequencies < 0.3)).all()
    assert 0.3 < common_halves[0] / sum(common_halves) < 0.7
    assert all(dict(spawn.semantic_tags).get("object_count") is None for spawn in objects.values())
    resolved, _ = resolve_task_config("IsaacContrib-VLA-Heterogeneous-Franka", "", overrides=[])
    assert type(resolved) is catalog.VLAHeterogeneousEnvCfg
    assert len(catalog.ASSET_CATALOG) == 100
    assert set(resolved.tasks) == {"put_in_container", "put_on_tray"}
    assert resolved.scene.objects is not catalog.OBJECTS_CFG
    assert len(resolved.scene.objects.rigid_objects) == catalog.OBJECT_COUNT[1]
    tables = resolved.scene.work_table.spawn
    assert isinstance(tables, sim_utils.MultiAssetSpawnerCfg) and not tables.static
    assert (
        len(tables.assets_cfg)
        == len(resolved.events.place_objects.params["table_variants"])
        == len(catalog.TABLE_VARIANTS)
    )
    assert all(isinstance(table, sim_utils.UsdFileCfg) for table in tables.assets_cfg), "Use real table assets"

    assert "Big_Plastic_Plate_A01" not in catalog.ASSET_CATALOG
    assert len({spawn.usd_path for spawn in catalog.ASSET_CATALOG.values()}) == 100
    # The default catalog must retain its size limits when composing slot-level count tags.
    candidates = [member.spawn.assets_cfg for member in resolved.scene.objects.rigid_objects.values()]
    candidates.append(tables.assets_cfg)
    groups, weights = make_valid_clone_combinations(names, candidates, resolved.scene.clone_cfg.clone_combinations)
    flattened = [candidate for row in candidates for candidate in row]
    plan = make_clone_plan(flattened, groups, 128, weights=weights)
    slots = np.repeat(np.arange(len(candidates)), [len(row) for row in candidates])
    receivers = {rule["target"] for rule in resolved.tasks.values()}
    seen_large = set()
    for composition in plan.topology.world_prototype_layout:
        start, end = plan.topology.world_prototype_starts[composition + 1 : composition + 3]
        members = [flattened[index] for index in plan.topology.world_prototypes[start:end]]
        labels = [dict(spawn.semantic_tags or ()).get("_catalog_key") for spawn in members]
        count = sum(label not in (None, "__empty__") for label in labels)
        # Every world offers a task: its first object is a receiver, its second fits it, the others are no receivers.
        world = dict(zip(slots[plan.topology.world_prototypes[start:end]].tolist(), members, strict=True))
        assert dict(world[0].semantic_tags)["_catalog_key"] in catalog.RECEIVER_ASSETS
        assert dict(world[1].semantic_tags)["_catalog_key"] in catalog.MOVABLE_ASSETS
        assert not any(dict(spawn.semantic_tags)["category"] in receivers for slot, spawn in world.items() if slot > 1)
        for label in labels:
            if label in ("Tray_A01", "Bowl_N01"):
                assert count <= {"Tray_A01": 2, "Bowl_N01": 4}[label]
                seen_large.add(label)
    assert seen_large == {"Tray_A01", "Bowl_N01"}
