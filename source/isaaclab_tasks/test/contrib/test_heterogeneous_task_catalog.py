# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, factorized scene combinations and the task-package ownership boundary."""

import ast
from pathlib import Path

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
    tree = ast.parse(Path(catalog.__file__).read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)), (
        "The scene must be declared through config composition, without factories or post-init rewriting"
    )
    assert all(type(spawn) is sim_utils.UsdFileCfg for spawn in catalog.ASSET_CATALOG.values()), (
        "Catalog assets must compose the existing USD config, without task-local cooking subclasses"
    )
    default_spawner = sim_utils.UsdFileCfg(usd_path="").func
    assert all(spawn.func == default_spawner for spawn in catalog.ASSET_CATALOG.values()), (
        "Collision cooking belongs to the backend, not custom catalog spawning hooks"
    )
    cfg, _ = resolve_task_config("IsaacContrib-TableTop-PickPlace-Franka", "", overrides=[])
    assert type(cfg) is catalog.TabletopPickPlaceEnvCfg
    assert len(catalog.ASSET_CATALOG) == 100
    assert set(cfg.tasks) == {"put_in_container", "put_on_tray"}
    assert cfg.events.place_objects.params["object_count"] == catalog.OBJECT_COUNT
    members = tuple(cfg.scene.objects.rigid_objects.values())
    assert len(members) == catalog.OBJECT_COUNT[1]
    expected = (*catalog.ASSET_CATALOG, "__empty__")
    for member in members:
        assert not member.spawn.static
        assert tuple(dict(spawn.semantic_tags)["_catalog_key"] for spawn in member.spawn.assets_cfg) == expected
        assert not member.spawn.assets_cfg[-1].collision_props.collision_enabled
        assert all(tag != "object_count" for spawn in member.spawn.assets_cfg for tag, _ in spawn.semantic_tags)
    assert "Big_Plastic_Plate_A01" not in catalog.ASSET_CATALOG
    tables = cfg.scene.work_table.spawn
    table_paths = {table.usd_path for table in tables.assets_cfg}
    assert len(table_paths) == 9
    assert all(path.startswith(f"{catalog.ISAAC_NUCLEUS_DIR}/SimReady/") for path in table_paths)
    assert cfg.scene.table_fixture.spawn.usd_path in table_paths
    assert len(tables.assets_cfg) == len(cfg.events.place_objects.params["table_variants"])
    # Thousands of entries remain independent choices in each slot, never a Cartesian product of worlds.
    catalog_assets = [
        sim_utils.CuboidCfg(size=(0.02, 0.03, 0.04), semantic_tags=[("category", f"group_{i % 5}")])
        for i in range(1000)
    ]
    candidates = [catalog_assets for _ in members] + [tables.assets_cfg]
    names = ["objects"] * len(members) + ["work_table"]
    groups, weights = make_valid_clone_combinations(names, candidates, cfg.scene.clone_cfg.clone_combinations)
    assert len(groups) == 1
    plan = make_clone_plan([candidate for row in candidates for candidate in row], groups, 128, weights=weights)
    assert len(plan.topology.world_prototype_starts) - 2 <= 128
