# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime

import jinja2
from common import MULTI_AGENT_ALGORITHMS, ROOT_DIR, SINGLE_AGENT_ALGORITHMS, TASKS_DIR, TEMPLATE_DIR

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
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


def _replace_in_file(replacements: list[tuple[str, str]], src: str, dst: str | None = None) -> None:
    """Replace the placeholders in the file.

    Args:
        replacements: The replacements to make.
        src: The source file.
        dst: The destination file. If not provided, the source file will be overwritten.
    """
    with open(src) as file:
        content = file.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(src if dst is None else dst, "w") as file:
        file.write(content)


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
    # common content
    # - task/__init__.py
    template = jinja_env.get_template("tasks/__init__task")
    _write_file(os.path.join(task_dir, "__init__.py"), content=template.render(**specification))
    # - task/agents/__init__.py
    template = jinja_env.get_template("tasks/__init__agents")
    _write_file(os.path.join(agents_dir, "__init__.py"), content=template.render(**specification))
    # - task/agents/*cfg*
    for rl_library in specification["rl_libraries"]:
        rl_library_name = rl_library["name"]
        for algorithm in rl_library.get("algorithms", []):
            file_name = f"{rl_library_name}_{algorithm.lower()}_cfg"
            file_ext = ".py" if rl_library_name == "rsl_rl" else ".yaml"
            try:
                template = jinja_env.get_template(f"agents/{file_name}")
            except jinja2.exceptions.TemplateNotFound:
                print(f"Template not found: agents/{file_name}")
                continue
            _write_file(os.path.join(agents_dir, file_name + file_ext), content=template.render(**specification))
    # workflow-specific content
    if task_spec["workflow"]["name"] == "direct":
        # - task/*env_cfg.py
        template = jinja_env.get_template(f'tasks/direct_{task_spec["workflow"]["type"]}/env_cfg')
        _write_file(
            os.path.join(task_dir, f'{task_spec["filename"]}_env_cfg.py'), content=template.render(**specification)
        )
        # - task/*env.py
        template = jinja_env.get_template(f'tasks/direct_{task_spec["workflow"]["type"]}/env')
        _write_file(os.path.join(task_dir, f'{task_spec["filename"]}_env.py'), content=template.render(**specification))
    elif task_spec["workflow"]["name"] == "manager-based":
        # - task/*env_cfg.py
        template = jinja_env.get_template(f'tasks/manager-based_{task_spec["workflow"]["type"]}/env_cfg')
        _write_file(
            os.path.join(task_dir, f'{task_spec["filename"]}_env_cfg.py'), content=template.render(**specification)
        )
        # - task/mdp folder
        shutil.copytree(
            os.path.join(TEMPLATE_DIR, "tasks", f'manager-based_{task_spec["workflow"]["type"]}', "mdp"),
            os.path.join(task_dir, "mdp"),
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
    task_name_prefix = "Template" if specification["external"] else "Isaac"
    general_task_name = "-".join([item.capitalize() for item in specification["name"].split("_")])
    for workflow in specification["workflows"]:
        task_name = general_task_name + ("-Marl" if workflow["type"] == "multi-agent" else "")
        filename = task_name.replace("-", "_").lower()
        task = {
            "workflow": workflow,
            "filename": filename,
            "classname": task_name.replace("-", ""),
            "dir": os.path.join(task_dir, workflow["name"].replace("-", "_"), filename),
        }
        if task["workflow"]["name"] == "direct":
            task["id"] = f"{task_name_prefix}-{task_name}-Direct-v0"
        elif task["workflow"]["name"] == "manager-based":
            task["id"] = f"{task_name_prefix}-{task_name}-v0"
        print(f"  |    |-- Generating '{task['id']}' task...")
        _generate_task_per_workflow(task["dir"], {**specification, "task": task})
        specifications.append({**specification, "task": task})
    return specifications


def _external(specification: dict) -> None:
    """Generate an external project.

    Args:
        specification: The specification of the project/task.
    """
    name = specification["name"]
    project_dir = os.path.join(specification["path"], name)
    os.makedirs(project_dir, exist_ok=True)
    # repo files
    print("  |-- Copying repo files...")
    shutil.copyfile(os.path.join(ROOT_DIR, ".dockerignore"), os.path.join(project_dir, ".dockerignore"))
    shutil.copyfile(os.path.join(ROOT_DIR, ".flake8"), os.path.join(project_dir, ".flake8"))
    shutil.copyfile(os.path.join(ROOT_DIR, ".gitattributes"), os.path.join(project_dir, ".gitattributes"))
    if os.path.exists(os.path.join(ROOT_DIR, ".gitignore")):
        shutil.copyfile(os.path.join(ROOT_DIR, ".gitignore"), os.path.join(project_dir, ".gitignore"))
    shutil.copyfile(
        os.path.join(ROOT_DIR, ".pre-commit-config.yaml"), os.path.join(project_dir, ".pre-commit-config.yaml")
    )
    template = jinja_env.get_template("external/README.md")
    _write_file(os.path.join(project_dir, "README.md"), content=template.render(**specification))
    # scripts
    print("  |-- Copying scripts...")
    # reinforcement learning libraries
    dir = os.path.join(project_dir, "scripts")
    os.makedirs(dir, exist_ok=True)
    for rl_library in specification["rl_libraries"]:
        shutil.copytree(
            os.path.join(ROOT_DIR, "scripts", "reinforcement_learning", rl_library["name"]),
            os.path.join(dir, rl_library["name"]),
            dirs_exist_ok=True,
        )
        # replace placeholder in scripts
        for file in glob.glob(os.path.join(dir, rl_library["name"], "*.py")):
            _replace_in_file(
                [(
                    "# PLACEHOLDER: Extension template (do not remove this comment)",
                    f"import {name}.tasks  # noqa: F401",
                )],
                src=file,
            )
    # - other scripts
    _replace_in_file(
        [("import isaaclab_tasks", f"import {name}.tasks"), ("isaaclab_tasks", name), ('"Isaac"', '"Template-"')],
        src=os.path.join(ROOT_DIR, "scripts", "environments", "list_envs.py"),
        dst=os.path.join(dir, "list_envs.py"),
    )
    for script in ["zero_agent.py", "random_agent.py"]:
        _replace_in_file(
            [(
                "# PLACEHOLDER: Extension template (do not remove this comment)",
                f"import {name}.tasks  # noqa: F401",
            )],
            src=os.path.join(ROOT_DIR, "scripts", "environments", script),
            dst=os.path.join(dir, script),
        )
    # # docker files
    # print("  |-- Copying docker files...")
    # dir = os.path.join(project_dir, "docker")
    # os.makedirs(dir, exist_ok=True)
    # template = jinja_env.get_template("external/docker/.env.base")
    # _write_file(os.path.join(dir, ".env.base"), content=template.render(**specification))
    # template = jinja_env.get_template("external/docker/docker-compose.yaml")
    # _write_file(os.path.join(dir, "docker-compose.yaml"), content=template.render(**specification))
    # template = jinja_env.get_template("external/docker/Dockerfile")
    # _write_file(os.path.join(dir, "Dockerfile"), content=template.render(**specification))
    # extension files
    print("  |-- Copying extension files...")
    # - config/extension.toml
    dir = os.path.join(project_dir, "source", name, "config")
    os.makedirs(dir, exist_ok=True)
    template = jinja_env.get_template("extension/config/extension.toml")
    _write_file(os.path.join(dir, "extension.toml"), content=template.render(**specification))
    # - docs/CHANGELOG.rst
    dir = os.path.join(project_dir, "source", name, "docs")
    os.makedirs(dir, exist_ok=True)
    template = jinja_env.get_template("extension/docs/CHANGELOG.rst")
    _write_file(
        os.path.join(dir, "CHANGELOG.rst"), content=template.render({"date": datetime.now().strftime("%Y-%m-%d")})
    )
    # - setup.py and pyproject.toml
    dir = os.path.join(project_dir, "source", name)
    template = jinja_env.get_template("extension/setup.py")
    _write_file(os.path.join(dir, "setup.py"), content=template.render(**specification))
    shutil.copyfile(os.path.join(TEMPLATE_DIR, "extension", "pyproject.toml"), os.path.join(dir, "pyproject.toml"))
    # - tasks
    print("  |-- Generating tasks...")
    dir = os.path.join(project_dir, "source", name, name, "tasks")
    os.makedirs(dir, exist_ok=True)
    specifications = _generate_tasks(specification, dir)
    shutil.copyfile(os.path.join(TEMPLATE_DIR, "extension", "__init__tasks"), os.path.join(dir, "__init__.py"))
    for workflow in specification["workflows"]:
        shutil.copyfile(
            os.path.join(TEMPLATE_DIR, "extension", "__init__workflow"),
            os.path.join(dir, workflow["name"].replace("-", "_"), "__init__.py"),
        )
    # - other files
    dir = os.path.join(project_dir, "source", name, name)
    template = jinja_env.get_template("extension/ui_extension_example.py")
    _write_file(os.path.join(dir, "ui_extension_example.py"), content=template.render(**specification))
    shutil.copyfile(os.path.join(TEMPLATE_DIR, "extension", "__init__ext"), os.path.join(dir, "__init__.py"))
    # .vscode files
    print("  |-- Copying vscode files...")
    dir = os.path.join(project_dir, ".vscode")
    shutil.copytree(os.path.join(TEMPLATE_DIR, "external", ".vscode"), dir, dirs_exist_ok=True)
    template = jinja_env.get_template("external/.vscode/tasks.json")
    _write_file(os.path.join(dir, "tasks.json"), content=template.render(**specification))
    template = jinja_env.get_template("external/.vscode/tools/launch.template.json")
    _write_file(
        os.path.join(dir, "tools", "launch.template.json"), content=template.render(specifications=specifications)
    )
    # setup git repo
    print(f"Setting up git repo in {project_dir} path...")
    _setup_git_repo(project_dir)
    # show end message
    print("\n" + "-" * 80)
    print(f"Project '{name}' generated successfully in {project_dir} path.")
    print(f"See {project_dir}/README.md to get started!")
    print("-" * 80)


def get_algorithms_per_rl_library(single_agent: bool = True, multi_agent: bool = True):
    assert single_agent or multi_agent, "At least one of 'single_agent' or 'multi_agent' must be True"
    data = {"rl_games": [], "rsl_rl": [], "skrl": [], "sb3": []}
    # get algorithms
    for file in glob.glob(os.path.join(TEMPLATE_DIR, "agents", "*_cfg")):
        for rl_library in data.keys():
            basename = os.path.basename(file).replace("_cfg", "")
            if basename.startswith(f"{rl_library}_"):
                algorithm = basename.replace(f"{rl_library}_", "").upper()
                assert (
                    algorithm in SINGLE_AGENT_ALGORITHMS or algorithm in MULTI_AGENT_ALGORITHMS
                ), f"{algorithm} algorithm is not listed in the supported algorithms"
                if single_agent and algorithm in SINGLE_AGENT_ALGORITHMS:
                    data[rl_library].append(algorithm)
                if multi_agent and algorithm in MULTI_AGENT_ALGORITHMS:
                    data[rl_library].append(algorithm)
    # remove duplicates and sort
    for rl_library in data.keys():
        data[rl_library] = sorted(list(set(data[rl_library])))
    return data


def generate(specification: dict) -> None:
    """Generate the project/task.

    Args:
        specification: The specification of the project/task.
    """
    # validate specification
    print("\nValidating specification...")
    assert "external" in specification, "External flag is required"
    assert specification.get("name", "").isidentifier(), "Name must be a valid identifier"
    for workflow in specification["workflows"]:
        assert workflow["name"] in ["direct", "manager-based"], f"Invalid workflow: {workflow}"
        assert workflow["type"] in ["single-agent", "multi-agent"], f"Invalid workflow type: {workflow}"
    if specification["external"]:
        assert "path" in specification, "Path is required for external projects"
    # add other information to specification
    specification["platform"] = sys.platform
    # generate project/task
    if specification["external"]:
        print("Generating external project...")
        _external(specification)
    else:
        print("Generating internal task...")
        print("  |-- Generating tasks...")
        _generate_tasks(specification, TASKS_DIR)
