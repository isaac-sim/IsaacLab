# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative benchmark records and their shared scene/registration boundary."""

from pathlib import Path

import pytest

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import HeterogeneousTaskCfg
from isaaclab_tasks.contrib.heterogeneous_tasks import heterogeneous_tasks_env_cfg as catalog
from isaaclab_tasks.contrib.heterogeneous_tasks.mdp.utils import entity_names
from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_flat_catalog_owns_task_records_and_individual_registrations():
    """All benchmark identities bind one config class; assets, poses and success references stay consistent."""
    root = Path(catalog.__file__).parent
    assert not (root / "tasks").exists(), "Do not recreate per-task modules"
    assert not (root / "objects.py").exists(), "The flat scene owns the asset catalog"
    env_classes = {
        value for value in vars(catalog).values()
        if isinstance(value, type) and value.__module__ == catalog.__name__ and issubclass(value, ManagerBasedRLEnvCfg)
    }
    assert env_classes == {catalog.VLAHeterogeneousEnvCfg}, "One shared environment class owns every task record"
    assert len(catalog.TASKS) == 256
    for task in catalog.TASKS.values():
        assert isinstance(task, HeterogeneousTaskCfg) and not isinstance(task, ManagerBasedRLEnvCfg)
        assert task.instruction and task.episode_length_s > 0
        assets = set(task.inclusion.assets)
        assert assets <= catalog.ASSET_CATALOG.keys()
        assert task.init_states.keys() <= assets
        assert set(entity_names([term.params for term in task.terminations.values()])) <= assets
    for index in (0, 34, 255):
        name = f"PickPlace{index:03d}"
        cfg, _ = resolve_task_config(f"IsaacContrib-VLA-{name}-Franka", "", overrides=[])
        assert type(cfg) is catalog.VLAHeterogeneousEnvCfg
        assert list(cfg.tasks) == [name]
        assert cfg.instruction == catalog.TASKS[name].instruction
        assert cfg.episode_length_s == catalog.TASKS[name].episode_length_s
        assert all(vars(cfg.scene).get(asset) is None for asset in catalog.ASSET_CATALOG)
        assert set(cfg.events.select_task.params["objects"]) == set(catalog.TASKS[name].inclusion.assets)
