# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, factorized scene combinations and the task-package ownership boundary."""

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
    cfg = catalog.VLAHeterogeneousEnvCfg(objects=objects)
    names = cfg.events.place_objects.params["objects"]
    candidates = [getattr(cfg.scene, name).spawn.assets_cfg for name in names]
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
    for composition in topology.world_prototype_layout:
        start, end = topology.world_prototype_starts[composition + 1 : composition + 3]
        members = [
            dict(flattened[index].semantic_tags)["_catalog_key"] for index in topology.world_prototypes[start:end]
        ]
        real = [name for name in members if name != "__empty__"]
        counts.append(len(real))
        chosen.update(real)
        for name in real:
            category = dict(objects[name].semantic_tags)["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            index = int(name.removeprefix("item_"))
            if index < 900:
                common_halves[index // 450] += 1
    assert min(counts) == 2 and max(counts) == 8
    assert chosen <= objects.keys() and len(chosen) > 100
    # A 900-asset category must not overwhelm four categories of only 25 assets each.
    frequencies = np.array(list(category_counts.values())) / sum(counts)
    assert len(frequencies) == 5 and ((frequencies > 0.1) & (frequencies < 0.3)).all()
    assert 0.3 < common_halves[0] / sum(common_halves) < 0.7
    assert all(dict(spawn.semantic_tags).get("object_count") is None for spawn in objects.values())
    ten = catalog.VLAHeterogeneousEnvCfg(objects={"one": objects["item_0"]}, object_count=(8, 10))
    ten_names = ten.events.place_objects.params["objects"]
    ten_candidates = [getattr(ten.scene, name).spawn.assets_cfg for name in ten_names]
    ten_groups, _ = make_valid_clone_combinations(ten_names, ten_candidates, ten.scene.clone_cfg.clone_combinations)
    assert len(ten_names) == 10 and len(ten_groups) == 3
    resolved, _ = resolve_task_config("IsaacContrib-VLA-Heterogeneous-Franka", "", overrides=[])
    assert type(resolved) is catalog.VLAHeterogeneousEnvCfg
    assert len(resolved.objects) == 159 and set(resolved.tasks) == {"put_on_plate", "put_in_container"}
