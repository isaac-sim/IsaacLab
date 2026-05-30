# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
TEMPLATE_TOOL_DIR = ROOT_DIR / "tools" / "template"
sys.path.insert(0, str(TEMPLATE_TOOL_DIR))

import generator  # noqa: E402
from generator import generate, get_algorithms_per_rl_library  # noqa: E402

_SINGLE_AGENT_RL_LIBRARIES = [
    {"name": "rl_games", "algorithms": ["ppo"]},
    {"name": "rsl_rl", "algorithms": ["ppo"]},
    {"name": "skrl", "algorithms": ["amp", "ppo"]},
    {"name": "sb3", "algorithms": ["ppo"]},
]

_MULTI_AGENT_RL_LIBRARIES = [
    {"name": "skrl", "algorithms": ["ippo", "mappo"]},
]


def _task_name(project_name: str) -> str:
    """Return the generated task name stem for a project name."""
    return "-".join(item.capitalize() for item in project_name.split("_"))


def _task_folder(project_name: str, workflow_type: str) -> str:
    """Return the generated task folder name for a project name."""
    task_name = _task_name(project_name)
    if workflow_type == "multi-agent":
        task_name += "-Marl"
    return task_name.replace("-", "_").lower()


def _task_class(project_name: str, workflow_type: str) -> str:
    """Return the generated task class stem for a project name."""
    task_name = _task_name(project_name)
    if workflow_type == "multi-agent":
        task_name += "-Marl"
    return task_name.replace("-", "")


def _task_id(project_name: str, workflow_name: str, workflow_type: str, external: bool) -> str:
    """Return the generated Gym task id."""
    prefix = "Template" if external else "Isaac"
    task_name = _task_name(project_name)
    if workflow_type == "multi-agent":
        task_name += "-Marl"
    if workflow_name == "direct":
        return f"{prefix}-{task_name}-Direct-v0"
    return f"{prefix}-{task_name}-v0"


def _task_dir(root_dir: Path, project_name: str, workflow_name: str, workflow_type: str, external: bool) -> Path:
    """Return the generated task directory."""
    tasks_dir = root_dir
    if external:
        tasks_dir = root_dir / project_name / "source" / project_name / project_name / "tasks"
    return tasks_dir / workflow_name.replace("-", "_") / _task_folder(project_name, workflow_type)


def _load_registration_module(task_dir: Path, module_name: str) -> None:
    """Execute a generated task registration module without importing its parent package."""
    spec = importlib.util.spec_from_file_location(
        module_name, task_dir / "__init__.py", submodule_search_locations=[str(task_dir)]
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)


def _unregister(task_id: str) -> None:
    """Remove a generated Gym task id from the process-local registry."""
    gym.envs.registration.registry.pop(task_id, None)


@pytest.mark.parametrize(
    ("single_agent", "multi_agent", "expected"),
    [
        (
            True,
            False,
            {
                "rl_games": ["PPO"],
                "rsl_rl": ["PPO"],
                "skrl": ["AMP", "PPO"],
                "sb3": ["PPO"],
            },
        ),
        (
            False,
            True,
            {
                "rl_games": [],
                "rsl_rl": [],
                "skrl": ["IPPO", "MAPPO"],
                "sb3": [],
            },
        ),
        (
            True,
            True,
            {
                "rl_games": ["PPO"],
                "rsl_rl": ["PPO"],
                "skrl": ["AMP", "IPPO", "MAPPO", "PPO"],
                "sb3": ["PPO"],
            },
        ),
    ],
)
def test_get_algorithms_per_rl_library_filters_by_workflow_type(single_agent, multi_agent, expected):
    """Check that the CLI-facing algorithm discovery matches supported workflow types."""
    assert get_algorithms_per_rl_library(single_agent=single_agent, multi_agent=multi_agent) == expected


@pytest.mark.parametrize("external", [False, True])
def test_generator_registers_single_agent_rl_config_entry_points_for_all_libraries(tmp_path, monkeypatch, external):
    """Generate single-agent tasks and verify every RL library gets the registry key its train script expects."""
    project_name = f"template_single_{'external' if external else 'internal'}"
    root_dir = tmp_path / ("external_root" if external else "internal_tasks")
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    if not external:
        monkeypatch.setattr(generator, "TASKS_DIR", str(root_dir))

    specification = {
        "external": external,
        "name": project_name,
        "workflows": [
            {"name": "manager-based", "type": "single-agent"},
            {"name": "direct", "type": "single-agent"},
        ],
        "rl_libraries": _SINGLE_AGENT_RL_LIBRARIES,
    }
    if external:
        specification["path"] = str(root_dir)

    generate(specification)

    for workflow_name in ["manager-based", "direct"]:
        workflow_type = "single-agent"
        task_id = _task_id(project_name, workflow_name, workflow_type, external)
        task_dir = _task_dir(root_dir, project_name, workflow_name, workflow_type, external)
        module_name = f"_template_test_{project_name}_{workflow_name.replace('-', '_')}"
        _unregister(task_id)
        _load_registration_module(task_dir, module_name)

        spec = gym.spec(task_id)
        task_folder = _task_folder(project_name, workflow_type)
        task_class = _task_class(project_name, workflow_type)
        agents_module = f"{module_name}.agents"

        if workflow_name == "direct":
            assert spec.entry_point == f"{module_name}.{task_folder}_env:{task_class}Env"
        else:
            assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"

        assert spec.kwargs["env_cfg_entry_point"] == f"{module_name}.{task_folder}_env_cfg:{task_class}EnvCfg"
        assert spec.kwargs["rl_games_cfg_entry_point"] == f"{agents_module}:rl_games_ppo_cfg.yaml"
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == f"{agents_module}.rsl_rl_ppo_cfg:PPORunnerCfg"
        assert spec.kwargs["skrl_amp_cfg_entry_point"] == f"{agents_module}:skrl_amp_cfg.yaml"
        assert spec.kwargs["skrl_cfg_entry_point"] == f"{agents_module}:skrl_ppo_cfg.yaml"
        assert spec.kwargs["sb3_cfg_entry_point"] == f"{agents_module}:sb3_ppo_cfg.yaml"
        assert "skrl_ppo_cfg_entry_point" not in spec.kwargs

        _unregister(task_id)


@pytest.mark.parametrize("external", [False, True])
def test_generator_registers_multi_agent_skrl_config_entry_points(tmp_path, monkeypatch, external):
    """Generate a multi-agent task and verify skrl IPPO/MAPPO registry keys."""
    project_name = f"template_multi_{'external' if external else 'internal'}"
    root_dir = tmp_path / ("external_root" if external else "internal_tasks")
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    if not external:
        monkeypatch.setattr(generator, "TASKS_DIR", str(root_dir))

    specification = {
        "external": external,
        "name": project_name,
        "workflows": [{"name": "direct", "type": "multi-agent"}],
        "rl_libraries": _MULTI_AGENT_RL_LIBRARIES,
    }
    if external:
        specification["path"] = str(root_dir)

    generate(specification)

    task_id = _task_id(project_name, "direct", "multi-agent", external)
    task_dir = _task_dir(root_dir, project_name, "direct", "multi-agent", external)
    module_name = f"_template_test_{project_name}_direct"
    _unregister(task_id)
    _load_registration_module(task_dir, module_name)

    spec = gym.spec(task_id)
    task_folder = _task_folder(project_name, "multi-agent")
    task_class = _task_class(project_name, "multi-agent")
    agents_module = f"{module_name}.agents"

    assert spec.entry_point == f"{module_name}.{task_folder}_env:{task_class}Env"
    assert spec.kwargs["env_cfg_entry_point"] == f"{module_name}.{task_folder}_env_cfg:{task_class}EnvCfg"
    assert spec.kwargs["skrl_ippo_cfg_entry_point"] == f"{agents_module}:skrl_ippo_cfg.yaml"
    assert spec.kwargs["skrl_mappo_cfg_entry_point"] == f"{agents_module}:skrl_mappo_cfg.yaml"
    assert "skrl_cfg_entry_point" not in spec.kwargs

    _unregister(task_id)


def test_external_launch_configs_pass_skrl_algorithm_for_every_generated_skrl_agent(tmp_path, monkeypatch):
    """Verify generated VS Code launch configs select the matching skrl algorithm."""
    project_name = "template_launch_external"
    root_dir = tmp_path / "external_root"
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)

    generate(
        {
            "external": True,
            "path": str(root_dir),
            "name": project_name,
            "workflows": [
                {"name": "direct", "type": "single-agent"},
                {"name": "direct", "type": "multi-agent"},
            ],
            "rl_libraries": [
                {"name": "skrl", "algorithms": ["amp", "ppo", "ippo", "mappo"]},
            ],
        }
    )

    launch_config = (root_dir / project_name / ".vscode" / "tools" / "launch.template.json").read_text()
    for algorithm in ["AMP", "PPO", "IPPO", "MAPPO"]:
        assert f'"--algorithm", "{algorithm}"' in launch_config
