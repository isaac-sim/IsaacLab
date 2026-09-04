# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import shutil
import subprocess

import jinja2
from common import MULTI_AGENT_ALGORITHMS, ROOT_DIR, SINGLE_AGENT_ALGORITHMS, TASKS_DIR, TEMPLATE_DIR

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def _setup_git_repo(project_dir: str) -> None:
    """Setup the git repository.

    Args:
        project_dir: The directory of the project.
    """
    commands = [
        ["git", "init"],
        ["git", "add", "-f", "."],
        ["git", "commit", "-q", "-m", "Initial commit"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, cwd=project_dir)
        for line in result.stdout.splitlines():
            print(f"  |  {line}")


def _write_file(dst: str, content: str) -> None:
    """Write the content to a file.

    Args:
        dst: The path to the file.
        content: The content to write to the file.
    """
    with open(dst, "w") as file:
        file.write(content)


def _generate_task_per_workflow(task_dir: str, specification: dict) -> None:
    """Generate the task files for a single workflow.

    Args:
        task_dir: The directory where the task files will be generated.
        specification: The specification of the project/task.
    """
    task_spec = specification["task"]
    agents_dir = os.path.join(task_dir, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    template = jinja_env.get_template("tasks/__init__task")
    _write_file(os.path.join(task_dir, "__init__.py"), content=template.render(**specification))
    template = jinja_env.get_template("tasks/__init__agents")
    _write_file(os.path.join(agents_dir, "__init__.py"), content=template.render(**specification))
    for rl_library in specification["rl_libraries"]:
        rl_library_name = rl_library["name"]
        for algorithm in rl_library.get("algorithms", []):
            file_name = f"{rl_library_name}_{algorithm.lower()}_cfg"
            file_ext = ".py" if rl_library_name == "rsl_rl" else ".yaml"
            try:
                template = jinja_env.get_template(f"agents/{file_name}")
            except jinja2.exceptions.TemplateNotFound as exc:
                raise FileNotFoundError(
                    f"No agent config template 'agents/{file_name}' for the requested '{rl_library_name}'"
                    f" algorithm '{algorithm}'. Add the template or drop the algorithm from the selection."
                ) from exc
            _write_file(os.path.join(agents_dir, file_name + file_ext), content=template.render(**specification))
    if task_spec["workflow"]["name"] == "direct":
        template = jinja_env.get_template(f"tasks/direct_{task_spec['workflow']['type']}/env_cfg")
        _write_file(
            os.path.join(task_dir, f"{task_spec['env_cfg_filename']}.py"), content=template.render(**specification)
        )
        template = jinja_env.get_template(f"tasks/direct_{task_spec['workflow']['type']}/env")
        _write_file(os.path.join(task_dir, f"{task_spec['env_filename']}.py"), content=template.render(**specification))
    elif task_spec["workflow"]["name"] == "manager-based":
        template = jinja_env.get_template(f"tasks/manager-based_{task_spec['workflow']['type']}/env_cfg")
        _write_file(
            os.path.join(task_dir, f"{task_spec['env_cfg_filename']}.py"), content=template.render(**specification)
        )
        shutil.copytree(
            os.path.join(TEMPLATE_DIR, "tasks", f"manager-based_{task_spec['workflow']['type']}", "mdp"),
            os.path.join(task_spec["family_dir"], "mdp"),
            dirs_exist_ok=True,
        )


def _generate_tasks(specification: dict, task_dir: str) -> list[dict]:
    """Generate the task files for an external project or an internal task.

    Args:
        specification: The specification of the project/task.
        task_dir: The directory where the tasks will be generated.

    Returns:
        A list of specifications for the tasks.
    """
    specifications = []
    task_family_name = specification.get("task_name", specification["name"])
    robot_name = specification.get("robot_name", "cartpole")
    general_task_name = "-".join(item.capitalize() for item in task_family_name.split("_"))
    project_name = specification.get(
        "task_id_prefix", "".join(item.capitalize() for item in specification["name"].split("_"))
    )
    robot_task_name = "-".join(
        item.upper() if any(character.isdigit() for character in item) else item.capitalize()
        for item in robot_name.split("_")
    )
    for workflow in specification["workflows"]:
        task_name = general_task_name + ("-Marl" if workflow["type"] == "multi-agent" else "")
        filename = task_name.replace("-", "_").lower()
        family_name = f"{filename}_direct" if workflow["name"] == "direct" else filename
        family_dir = os.path.join(task_dir, family_name)
        task_id = f"{project_name}-{task_name}-{robot_task_name}" if specification["external"] else f"Isaac-{task_name}"
        if workflow["name"] == "direct":
            task_id += "-Direct"
        task = {
            "workflow": workflow,
            "filename": filename,
            "classname": task_name.replace("-", ""),
            "family_name": family_name,
            "family_dir": family_dir,
            "dir": os.path.join(family_dir, "config", robot_name),
            "env_cfg_filename": "env_cfg" if specification["external"] else f"{filename}_env_cfg",
            "env_filename": "env" if specification["external"] else f"{filename}_env",
            "id": task_id,
        }
        print(f"  |    |-- Generating '{task['id']}' task...")
        for package_dir in (family_dir, os.path.join(family_dir, "config")):
            os.makedirs(package_dir, exist_ok=True)
            package_init = os.path.join(package_dir, "__init__.py")
            if not os.path.exists(package_init):
                shutil.copyfile(os.path.join(TEMPLATE_DIR, "extension", "__init__task_family"), package_init)
        task_specification = {**specification, "task": task}
        _generate_task_per_workflow(task["dir"], task_specification)
        specifications.append(task_specification)
    return specifications


def _external(specification: dict) -> None:
    """Generate an external project.

    Args:
        specification: The specification of the project/task.
    """
    name = specification["name"]
    project_dir = os.path.join(specification["path"], name)
    os.makedirs(project_dir, exist_ok=True)
    print("  |-- Copying repo files...")
    for filename in [".gitattributes", ".gitignore", ".pre-commit-config.yaml", "LICENSE"]:
        shutil.copyfile(os.path.join(TEMPLATE_DIR, "external", filename), os.path.join(project_dir, filename))
    template = jinja_env.get_template("external/pyproject.toml")
    _write_file(os.path.join(project_dir, "pyproject.toml"), content=template.render(**specification))
    print("  |-- Copying utility scripts...")
    scripts_dir = os.path.join(project_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    template = jinja_env.get_template("external/list_envs.py")
    _write_file(os.path.join(scripts_dir, "list_envs.py"), content=template.render(**specification))

    print("  |-- Generating tasks...")
    module_dir = os.path.join(project_dir, "src", name)
    tasks_dir = os.path.join(module_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    specifications = _generate_tasks(specification, tasks_dir)
    shutil.copyfile(os.path.join(TEMPLATE_DIR, "extension", "__init__tasks"), os.path.join(tasks_dir, "__init__.py"))
    template = jinja_env.get_template("external/__init__package")
    _write_file(os.path.join(module_dir, "__init__.py"), content=template.render(**specification))
    template = jinja_env.get_template("external/README.md")
    _write_file(
        os.path.join(project_dir, "README.md"), content=template.render(specifications=specifications, **specification)
    )

    print("  |-- Generating tests...")
    tests_dir = os.path.join(project_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    template = jinja_env.get_template("external/test_registration")
    _write_file(
        os.path.join(tests_dir, "test_registration.py"),
        content=template.render(specifications=specifications, **specification),
    )

    if specification.get("include_ui_extension", False):
        print("  |-- Copying Isaac Sim UI extension files...")
        config_dir = os.path.join(project_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        template = jinja_env.get_template("extension/config/extension.toml")
        _write_file(os.path.join(config_dir, "extension.toml"), content=template.render(**specification))
        template = jinja_env.get_template("extension/ui_extension_example.py")
        _write_file(os.path.join(module_dir, "ui_extension_example.py"), content=template.render(**specification))

    print("  |-- Copying vscode files...")
    vscode_dir = os.path.join(project_dir, ".vscode")
    shutil.copytree(os.path.join(TEMPLATE_DIR, "external", ".vscode"), vscode_dir, dirs_exist_ok=True)
    shutil.copyfile(
        os.path.join(ROOT_DIR, ".vscode", "tools", "setup_vscode.py"),
        os.path.join(vscode_dir, "tools", "setup_vscode.py"),
    )
    template = jinja_env.get_template("external/.vscode/tasks.json")
    _write_file(os.path.join(vscode_dir, "tasks.json"), content=template.render(**specification))
    template = jinja_env.get_template("external/.vscode/tools/launch.template.json")
    _write_file(
        os.path.join(vscode_dir, "tools", "launch.template.json"),
        content=template.render(specifications=specifications),
    )

    print(f"Setting up git repo in {project_dir} path...")
    _setup_git_repo(project_dir)
    print("\n" + "-" * 80)
    print(f"Project '{name}' generated successfully in {project_dir} path.")
    print(f"See {project_dir}/README.md to get started!")
    print("-" * 80)


def get_algorithms_per_rl_library(single_agent: bool = True, multi_agent: bool = True):
    assert single_agent or multi_agent, "At least one of 'single_agent' or 'multi_agent' must be True"
    data = {"rsl_rl": [], "rl_games": [], "skrl": [], "sb3": []}
    algorithm_order = SINGLE_AGENT_ALGORITHMS + MULTI_AGENT_ALGORITHMS
    for file in glob.glob(os.path.join(TEMPLATE_DIR, "agents", "*_cfg")):
        for rl_library in data.keys():
            basename = os.path.basename(file).replace("_cfg", "")
            if basename.startswith(f"{rl_library}_"):
                algorithm = basename.replace(f"{rl_library}_", "").upper()
                assert algorithm in SINGLE_AGENT_ALGORITHMS or algorithm in MULTI_AGENT_ALGORITHMS, (
                    f"{algorithm} algorithm is not listed in the supported algorithms"
                )
                if single_agent and algorithm in SINGLE_AGENT_ALGORITHMS:
                    data[rl_library].append(algorithm)
                if multi_agent and algorithm in MULTI_AGENT_ALGORITHMS:
                    data[rl_library].append(algorithm)
    for rl_library in data.keys():
        data[rl_library] = sorted(set(data[rl_library]), key=algorithm_order.index)
    return data


def generate(specification: dict) -> None:
    """Generate the project/task.

    Args:
        specification: The specification of the project/task.
    """
    print("\nValidating specification...")
    specification = specification.copy()
    assert "external" in specification, "External flag is required"
    assert specification.get("name", "").isidentifier(), "Name must be a valid identifier"
    if specification["external"]:
        specification.setdefault("task_name", "balance")
        specification.setdefault("robot_name", "cartpole")
        specification.setdefault("include_ui_extension", False)
        specification["task_id_prefix"] = "".join(item.capitalize() for item in specification["name"].split("_"))
        assert specification["task_name"].isidentifier(), "Task family name must be a valid identifier"
        assert specification["robot_name"].isidentifier(), "Robot/config name must be a valid identifier"
    for workflow in specification["workflows"]:
        assert workflow["name"] in ["direct", "manager-based"], f"Invalid workflow: {workflow}"
        assert workflow["type"] in ["single-agent", "multi-agent"], f"Invalid workflow type: {workflow}"
    if specification["external"]:
        assert "path" in specification, "Path is required for external projects"
    if specification["external"]:
        print("Generating external project...")
        _external(specification)
    else:
        print("Generating internal task...")
        print("  |-- Generating tasks...")
        _generate_tasks(specification, TASKS_DIR)
