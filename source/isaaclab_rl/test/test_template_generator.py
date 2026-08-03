# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import importlib.util
import pkgutil
import subprocess
import sys
import textwrap
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


def _all_libraries() -> list[dict]:
    """Every rl library with every algorithm it supports across both workflow types."""
    algorithms = get_algorithms_per_rl_library(single_agent=True, multi_agent=True)
    return [{"name": name, "algorithms": [a.lower() for a in algos]} for name, algos in algorithms.items() if algos]


# Backend runtime modules that must not be imported while a config is loaded (before SimulationApp); see PR #5826.
_FORBIDDEN_RUNTIME_MODULES = {"pxr", "omni", "carb", "isaacsim", "usdrt"}


def _top_level_imported_roots(source: str) -> set[str]:
    """Return the root package names imported UNCONDITIONALLY at module scope.

    Only statements directly in the module body are inspected, so deferred (in-function) and
    ``if TYPE_CHECKING:`` imports -- the sanctioned way to reference backend modules -- are ignored.
    """
    roots: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("external", [True, False])
def test_generated_env_modules_have_no_forbidden_top_level_imports(tmp_path, monkeypatch, external):
    """Generated ``*_env.py`` / ``*_env_cfg.py`` must not import backend runtime modules at module scope.

    Config loading runs before ``SimulationApp``; a top-level ``pxr``/``omni``/``carb``/``isaacsim`` import there
    causes the pre-SimulationApp crashes (e.g. ``free(): invalid pointer``, ``TfNotice`` wrapper not created). This
    catches such an import being (re)introduced into the env templates; deferred / ``TYPE_CHECKING`` imports are fine.
    """
    project_name = f"template_imports_{'external' if external else 'internal'}"
    root_dir = tmp_path / ("external_root" if external else "internal_tasks")
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    if not external:
        monkeypatch.setattr(generator, "TASKS_DIR", str(root_dir))

    specification = {
        "external": external,
        "name": project_name,
        "workflows": [
            {"name": "direct", "type": "single-agent"},
            {"name": "manager-based", "type": "single-agent"},
            {"name": "direct", "type": "multi-agent"},
        ],
        "rl_libraries": _all_libraries(),
    }
    if external:
        specification["path"] = str(root_dir)

    generate(specification)

    env_modules = list(root_dir.rglob("*_env.py")) + list(root_dir.rglob("*_env_cfg.py"))
    assert env_modules, "generator produced no env modules"
    for module_file in env_modules:
        forbidden = _top_level_imported_roots(module_file.read_text()) & _FORBIDDEN_RUNTIME_MODULES
        assert not forbidden, (
            f"{module_file} imports {sorted(forbidden)} at module top level (load before SimulationApp)"
        )


def test_each_requested_agent_cfg_file_is_generated(tmp_path, monkeypatch):
    """Every requested (RL library, algorithm) agent config is generated; a missing template raises, not skips."""
    project_name = "template_agents"
    root_dir = tmp_path / "external_root"
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)

    workflows = [
        {"name": "direct", "type": "single-agent"},
        {"name": "manager-based", "type": "single-agent"},
        {"name": "direct", "type": "multi-agent"},
    ]
    generate(
        {
            "external": True,
            "path": str(root_dir),
            "name": project_name,
            "workflows": workflows,
            "rl_libraries": _all_libraries(),
        }
    )

    single_libraries = {lib["name"]: lib["algorithms"] for lib in _SINGLE_AGENT_RL_LIBRARIES}
    multi_libraries = {lib["name"]: lib["algorithms"] for lib in _MULTI_AGENT_RL_LIBRARIES}
    for workflow in workflows:
        libraries = multi_libraries if workflow["type"] == "multi-agent" else single_libraries
        agents_dir = _task_dir(root_dir, project_name, workflow["name"], workflow["type"], external=True) / "agents"
        for library, algorithms in libraries.items():
            for algorithm in algorithms:
                extension = ".py" if library == "rsl_rl" else ".yaml"
                cfg_file = agents_dir / f"{library}_{algorithm}_cfg{extension}"
                assert cfg_file.exists(), f"missing generated agent config: {cfg_file}"


@pytest.mark.parametrize("external", [True, False])
def test_generated_python_is_syntactically_valid(tmp_path, monkeypatch, external):
    """Every generated ``.py`` must parse and compile, catching template edits that emit broken Python."""
    project_name = f"template_syntax_{'external' if external else 'internal'}"
    root_dir = tmp_path / ("external_root" if external else "internal_tasks")
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    if not external:
        monkeypatch.setattr(generator, "TASKS_DIR", str(root_dir))

    specification = {
        "external": external,
        "name": project_name,
        "workflows": [
            {"name": "direct", "type": "single-agent"},
            {"name": "manager-based", "type": "single-agent"},
            {"name": "direct", "type": "multi-agent"},
        ],
        "rl_libraries": _all_libraries(),
    }
    if external:
        specification["path"] = str(root_dir)

    generate(specification)

    python_files = list(root_dir.rglob("*.py"))
    assert python_files, "generator produced no Python files"
    for python_file in python_files:
        source = python_file.read_text()
        ast.parse(source, filename=str(python_file))
        compile(source, str(python_file), "exec")


def test_extension_toml_registers_ui_extension_as_separate_module(tmp_path, monkeypatch):
    """The UI extension must stay loadable by Kit via its own ``[[python.module]]`` entry now that the package
    ``__init__`` no longer imports it (so ``import <project>`` stays omni-free). Locks both halves of the fix."""
    project_name = "template_ui_module"
    root_dir = tmp_path / "external_root"
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    generate(
        {
            "external": True,
            "path": str(root_dir),
            "name": project_name,
            "workflows": [{"name": "direct", "type": "single-agent"}],
            "rl_libraries": [{"name": "skrl", "algorithms": ["ppo"]}],
        }
    )
    package_dir = root_dir / project_name / "source" / project_name / project_name
    extension_toml = (package_dir.parent / "config" / "extension.toml").read_text()
    assert f'name = "{project_name}"' in extension_toml
    assert f'name = "{project_name}.ui_extension_example"' in extension_toml
    # the package __init__ must not eagerly import the omni-dependent UI module (a comment mentioning it is fine)
    init_tree = ast.parse((package_dir / "__init__.py").read_text())
    imported = [node.module for node in ast.walk(init_tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(init_tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(module and "ui_extension_example" in module for module in imported)


def test_generated_package_import_is_omni_and_pxr_free(tmp_path, monkeypatch):
    """NVBug 6251247: importing a generated project must not pull ``omni``/``pxr`` (headless / pre-SimulationApp).

    The generated package ``__init__`` runs Gym registration (``import_packages``) on import; if it eagerly
    imports the example UI extension (``omni.ext``) or any pxr-loading module, ``import <project>`` crashes
    before Isaac Sim starts. We import the generated package in a subprocess with ``omni``/``pxr`` blocked.
    """
    project_name = "template_import_safe"
    root_dir = tmp_path / "external_root"
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    generate(
        {
            "external": True,
            "path": str(root_dir),
            "name": project_name,
            "workflows": [
                {"name": "direct", "type": "single-agent"},
                {"name": "manager-based", "type": "single-agent"},
                {"name": "direct", "type": "multi-agent"},
            ],
            "rl_libraries": _all_libraries(),
        }
    )
    source_dir = root_dir / project_name / "source" / project_name

    program = textwrap.dedent(
        f"""
        import sys

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("omni", "pxr"):
                    raise ImportError(f"BLOCKED eager import of {{name!r}} at package import time")
                return None

        sys.path.insert(0, {str(source_dir)!r})
        sys.meta_path.insert(0, _Blocker())
        import {project_name}  # noqa: F401
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"importing the generated package eagerly pulled omni/pxr:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_generated_manager_based_env_cfg_resolution_is_omni_and_pxr_free(tmp_path, monkeypatch):
    """NVBug 6251258/6257159: resolving a generated env config must not pull ``omni``/``pxr``.

    The training scripts use the deferred-launch pattern: ``resolve_task_config`` imports and
    instantiates the env config *before* ``launch_simulation`` starts Kit. If the env config (or an
    MDP term it references) eagerly loads ``pxr``, the USD plugins initialize out of order and Kit
    aborts with ``TfNotice ... has not been created yet``. We resolve the generated manager-based env
    config in a subprocess with ``omni``/``pxr`` blocked.
    """
    project_name = "template_env_safe"
    root_dir = tmp_path / "external_root"
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    generate(
        {
            "external": True,
            "path": str(root_dir),
            "name": project_name,
            "workflows": [{"name": "manager-based", "type": "single-agent"}],
            "rl_libraries": [{"name": "skrl", "algorithms": ["ppo"]}],
        }
    )
    source_dir = root_dir / project_name / "source" / project_name
    task_folder = _task_folder(project_name, "single-agent")
    env_cfg_module = f"{project_name}.tasks.manager_based.{task_folder}.{task_folder}_env_cfg"
    env_cfg_class = f"{_task_class(project_name, 'single-agent')}EnvCfg"

    program = textwrap.dedent(
        f"""
        import sys

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("omni", "pxr"):
                    raise ImportError(f"BLOCKED eager import of {{name!r}} before launch_simulation")
                return None

        sys.path.insert(0, {str(source_dir)!r})
        sys.meta_path.insert(0, _Blocker())

        import importlib

        cfg_module = importlib.import_module({env_cfg_module!r})
        getattr(cfg_module, {env_cfg_class!r})()  # instantiate to evaluate scene/mdp field defaults
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"generated env config eagerly pulled omni/pxr:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_generated_internal_task_workflow_dirs_are_importable_packages(tmp_path, monkeypatch):
    """An internal task must register out of the box: its workflow dirs must be importable packages.

    After the core/contrib split (#5891), ``isaaclab_tasks/{direct,manager_based}/`` are namespace dirs
    with no ``__init__.py``. ``import_packages`` uses ``pkgutil.iter_modules``, which skips namespace
    packages, so a generated internal task placed under them would never be discovered/registered. The
    generator must therefore create the per-workflow ``__init__.py``.
    """
    tasks_dir = tmp_path / "isaaclab_tasks"
    tasks_dir.mkdir()
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    monkeypatch.setattr(generator, "TASKS_DIR", str(tasks_dir))
    generate(
        {
            "external": False,
            "name": "template_internal_reg",
            "workflows": [
                {"name": "manager-based", "type": "single-agent"},
                {"name": "direct", "type": "single-agent"},
            ],
            "rl_libraries": [{"name": "skrl", "algorithms": ["ppo"]}],
        }
    )
    # pkgutil.iter_modules (what import_packages walks) lists a directory only if it is a regular package
    discovered = {info.name for info in pkgutil.iter_modules([str(tasks_dir)]) if info.ispkg}
    assert {"direct", "manager_based"} <= discovered, (
        "generated internal task workflow dirs are not importable packages, so import_packages would"
        f" skip them and the task would never register; discovered packages: {sorted(discovered)}"
    )


def test_generated_external_project_registers_tasks_on_import(tmp_path, monkeypatch):
    """A freshly generated external project must register all its tasks on a plain ``import`` — no manual steps.

    End-to-end check of the "generate then it just works" promise: import the generated package the normal
    way (its ``__init__`` runs Gym registration) and assert every workflow's task id is in the registry.
    """
    project_name = "template_reg_import"
    root_dir = tmp_path / "external_root"
    monkeypatch.setattr(generator, "_setup_git_repo", lambda project_dir: None)
    generate(
        {
            "external": True,
            "path": str(root_dir),
            "name": project_name,
            "workflows": [
                {"name": "manager-based", "type": "single-agent"},
                {"name": "direct", "type": "single-agent"},
                {"name": "direct", "type": "multi-agent"},
            ],
            "rl_libraries": [{"name": "skrl", "algorithms": ["ppo", "ippo", "mappo"]}],
        }
    )
    source_dir = root_dir / project_name / "source" / project_name
    expected = sorted(
        {
            _task_id(project_name, "manager-based", "single-agent", external=True),
            _task_id(project_name, "direct", "single-agent", external=True),
            _task_id(project_name, "direct", "multi-agent", external=True),
        }
    )
    program = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {str(source_dir)!r})
        import {project_name}  # noqa: F401  (registration runs on import, with no manual step)
        import gymnasium as gym

        want = {expected!r}
        missing = [task_id for task_id in want if task_id not in gym.registry]
        assert not missing, f"not registered on import: {{missing}}"
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"external project did not register tasks on import:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
