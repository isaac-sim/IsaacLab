# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, factorized scene combinations and the task-package ownership boundary."""

import ast
from pathlib import Path

import numpy as np
import pytest

import isaaclab.sim as sim_utils
from isaaclab.cloner import make_clone_plan, make_valid_clone_combinations

from isaaclab_tasks.contrib.heterogeneous_tasks import heterogeneous_tasks_env_cfg as catalog
from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


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
                                *[
                                    (tag, value)
                                    for tag, value in member.spawn.assets_cfg[0].semantic_tags
                                    if tag == "object_count"
                                ],
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
    assert set(resolved.tasks) == {"put_on_plate", "put_in_container", "put_on_tray"}
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
    seen_large = set()
    for composition in plan.topology.world_prototype_layout:
        start, end = plan.topology.world_prototype_starts[composition + 1 : composition + 3]
        members = [flattened[index] for index in plan.topology.world_prototypes[start:end]]
        labels = [dict(spawn.semantic_tags or ()).get("_catalog_key") for spawn in members]
        count = sum(label not in (None, "__empty__") for label in labels)
        for label in labels:
            if label in ("Tray_A01", "Bowl_N01"):
                assert count <= {"Tray_A01": 2, "Bowl_N01": 4}[label]
                seen_large.add(label)
    assert seen_large == {"Tray_A01", "Bowl_N01"}
