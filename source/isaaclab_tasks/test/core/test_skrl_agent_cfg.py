# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import pytest

from isaaclab_rl.skrl import SkrlRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, resolve_task_config


def _core_skrl_entry_points() -> list[tuple[str, str]]:
    entry_points = []
    for spec in gym.registry.values():
        env_cfg_entry_point = spec.kwargs.get("env_cfg_entry_point", "")
        if not isinstance(env_cfg_entry_point, str) or not env_cfg_entry_point.startswith("isaaclab_tasks.core"):
            continue
        for key in spec.kwargs:
            if key.startswith("skrl") and key.endswith("_cfg_entry_point"):
                entry_points.append((spec.id, key))
    return sorted(entry_points)


@pytest.mark.parametrize(("task_name", "entry_point_key"), _core_skrl_entry_points())
def test_core_skrl_entry_points_load_configclasses(task_name: str, entry_point_key: str) -> None:
    cfg = load_cfg_from_registry(task_name, entry_point_key)

    assert isinstance(cfg, SkrlRunnerCfg)
    runner_cfg = cfg.to_runner_dict()
    assert runner_cfg["models"]["policy"]["class"]
    assert runner_cfg["agent"]["class"]
    assert runner_cfg["trainer"]["class"]


def test_skrl_configclass_supports_hydra_overrides() -> None:
    _, cfg = resolve_task_config(
        "Isaac-Cartpole", "skrl_cfg_entry_point", overrides=["agent.agent.learning_rate=0.001"]
    )

    assert isinstance(cfg, SkrlRunnerCfg)
    assert cfg.agent.learning_rate == 0.001
