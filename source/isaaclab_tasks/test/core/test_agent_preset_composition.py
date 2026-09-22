# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture and integration tests for task-owned agent preset families."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.hydra import collect_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _get(cfg, path):
    for name in path:
        cfg = cfg[name] if isinstance(cfg, dict) else getattr(cfg, name)
    return cfg


@pytest.mark.parametrize(
    ("task", "entry_point", "preset", "env_path", "env_type", "agent_path", "agent_value"),
    [
        (
            "IsaacContrib-Cartpole-Showcase-Direct",
            "skrl_cfg_entry_point",
            "box_discrete",
            (),
            "BoxDiscreteEnvCfg",
            ("agent", "experiment", "directory"),
            "cartpole_direct_box_discrete",
        ),
        (
            "IsaacContrib-Cartpole-Camera-Showcase-Direct",
            "skrl_cfg_entry_point",
            "dict_discrete",
            (),
            "DictDiscreteEnvCfg",
            ("agent", "experiment", "directory"),
            "cartpole_camera_direct_dict_discrete",
        ),
        (
            "Isaac-Cartpole-Camera",
            "rsl_rl_cfg_entry_point",
            "resnet18",
            ("observations",),
            "ResNet18ObservationCfg",
            ("experiment_name",),
            "cartpole_features",
        ),
        (
            "Isaac-Cartpole-Camera",
            "rl_games_cfg_entry_point",
            "theia_tiny",
            ("observations",),
            "TheiaTinyObservationCfg",
            ("params", "config", "name"),
            "cartpole_features",
        ),
    ],
)
def test_shared_preset_resolves_matching_environment_and_agent(
    task, entry_point, preset, env_path, env_type, agent_path, agent_value
):
    env_cfg, agent_cfg = resolve_task_config(task, entry_point, overrides=[f"presets={preset}"])

    assert type(_get(env_cfg, env_path)).__name__ == env_type
    assert _get(agent_cfg, agent_path) == agent_value


@pytest.mark.parametrize(
    ("task", "entry_point", "expected"),
    [
        ("IsaacContrib-Cartpole-Showcase-Direct", "skrl_cfg_entry_point", None),
        ("IsaacContrib-Cartpole-Camera-Showcase-Direct", "skrl_cfg_entry_point", None),
        ("Isaac-Cartpole-Camera", "rsl_rl_cfg_entry_point", {"resnet18", "theia_tiny"}),
        ("Isaac-Cartpole-Camera", "rl_games_cfg_entry_point", {"resnet18", "theia_tiny"}),
    ],
)
def test_agent_roots_contain_only_variants_that_change_the_agent(task, entry_point, expected):
    env_options = set(collect_presets(load_cfg_from_registry(task, "env_cfg_entry_point"))[""])
    agent_options = collect_presets(load_cfg_from_registry(task, entry_point))[""]
    variants = set(agent_options) - {"default"}
    expected = env_options - {"default", "box_box"} if expected is None else expected

    assert variants == expected
    assert variants < env_options
    assert all(agent_options[name] != agent_options["default"] for name in variants)


def test_missing_agent_entrypoint_diagnostic_does_not_invent_algorithm_labels():
    with pytest.raises(ValueError) as exc_info:
        load_cfg_from_registry("IsaacContrib-Humanoid-AMP-Walk-Direct", "skrl_missing_cfg_entry_point")

    assert "skrl_cfg_entry_point" in str(exc_info.value)
    assert "skrl_amp_cfg_entry_point" in str(exc_info.value)
    assert "PPO" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("IsaacContrib-Cartpole-Showcase-Direct", {"env_cfg_entry_point", "skrl_cfg_entry_point"}),
        ("IsaacContrib-Cartpole-Camera-Showcase-Direct", {"env_cfg_entry_point", "skrl_cfg_entry_point"}),
        (
            "Isaac-Cartpole-Camera",
            {"env_cfg_entry_point", "rl_games_cfg_entry_point", "rsl_rl_cfg_entry_point"},
        ),
    ],
)
def test_preset_composed_tasks_expose_only_canonical_config_roots(task, expected):
    registered = {name for name in gym.spec(task).kwargs if name.endswith("_cfg_entry_point")}
    assert registered == expected
    assert "agent_preset_compatibility" not in gym.spec(task).kwargs
