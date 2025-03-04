# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil

import jinja2
from common import TASKS_DIR, TEMPLATE_DIR

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _write_file(dst: str, content: str):
    with open(dst, "w") as file:
        file.write(content)


def _generate_task(task_dir: str, specification: dict):
    task_spec = specification["task"]
    agents_dir = os.path.join(task_dir, "agents")
    os.makedirs(agents_dir, exist_ok=True)

    # common content
    # - task/__init__.py
    template = jinja_env.get_template("task__init__")
    _write_file(os.path.join(task_dir, "__init__.py"), content=template.render(**specification))
    # - task/agents/__init__.py
    template = jinja_env.get_template("agents__init__")
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


def _internal(specification: dict):
    task_name = "-".join([item.capitalize() for item in specification["name"].split("_")])
    for workflow in specification["workflows"]:
        filename = specification["name"] + ("_marl" if workflow["type"] == "multi-agent" else "")
        task = {
            "workflow": workflow,
            "filename": filename,
            "classname": task_name.replace("-", "") + ("MARL" if workflow["type"] == "multi-agent" else ""),
            "dir": os.path.join(TASKS_DIR, workflow["name"].replace("-", "_"), filename),
        }
        if task["workflow"]["name"] == "direct":
            task["id"] = f"Isaac-{task_name}-Direct-v0"
        elif task["workflow"]["name"] == "manager-based":
            task["id"] = f"Isaac-{task_name}-v0"
        _generate_task(task["dir"], {**specification, "task": task})


def _external(specification: dict):
    return False


def generate(specification: dict):
    # validate specification
    assert len(specification.get("name", "")), "Name is required"
    # if workflow["name"] not in ["direct", "manager-based"]:
    #     raise ValueError(f"Invalid workflow: {workflow}")
    # generate project/task
    (_external if specification.get("external", False) else _internal)(specification)


if __name__ == "__main__":
    spec = {
        "external": False,
        "path": None,
        "name": "lorem_ipsum",
        "workflows": [
            {"name": "direct", "type": "single-agent"},
            {"name": "direct", "type": "multi-agent"},
            {"name": "manager-based", "type": "single-agent"},
        ],
        "rl_libraries": [
            {"name": "rl_games", "algorithms": ["ppo"]},
            {"name": "rsl_rl", "algorithms": ["ppo"]},
            {"name": "skrl", "algorithms": ["amp", "ppo", "ippo", "mappo"]},
            {"name": "sb3", "algorithms": ["ppo"]},
        ],
    }
    generate(spec)
