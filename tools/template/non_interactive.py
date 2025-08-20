# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import os
from typing import List, Dict

from common import ROOT_DIR
from generator import generate, get_algorithms_per_rl_library


def _parse_workflow_arg(item: str) -> Dict[str, str]:
    raw = item.strip().lower()
    # Enforce strict underscore format: "<name>_<type>"
    if "_" not in raw or any(sep in raw for sep in ("|", ":", " ")):
        raise ValueError(
            "Invalid workflow format. Use underscore format like 'direct_single_agent' or 'manager-based_single_agent'"
        )
    name_token, type_token_raw = raw.split("_", 1)
    type_token = type_token_raw.replace("_", "-")  # normalize to single-agent / multi-agent

    if name_token not in {"direct", "manager-based"}:
        raise ValueError(
            f"Invalid workflow name: {name_token}. Allowed: 'direct' or 'manager-based'"
        )
    if type_token not in {"single-agent", "multi-agent"}:
        raise ValueError(
            f"Invalid workflow type: {type_token}. Allowed: 'single-agent' or 'multi-agent'"
        )

    return {"name": name_token, "type": type_token}


def _validate_external_path(path: str) -> None:
    if os.path.abspath(path).startswith(os.path.abspath(ROOT_DIR)):
        raise ValueError("External project path cannot be within the Isaac Lab project")


def main(argv: List[str] | None = None) -> None:
    '''
    Non-interactive entrypoint for the template generator workflow.

    Parses command-line flags, builds the specification dict, and calls generate().
    This avoids any interactive prompts or dependencies on Inquirer-based flow.
    '''
    
    parser = argparse.ArgumentParser(add_help=False)
    supported_workflows = [
        "direct_single_agent",
        "direct_multi_agent",
        "manager-based_single_agent",
    ]
    supported_rl_libraries = ["rl_games", "rsl_rl", "skrl", "sb3"]
    # All known algorithms across libraries (lowercase for consistent CLI input)
    _all_algos_map = get_algorithms_per_rl_library(True, True)
    rl_algo_choices = sorted(
        {algo.lower() for algos in _all_algos_map.values() for algo in algos}
    )

    parser.add_argument("--task-type", "--task_type", type=str, required=True, choices=["External", "Internal"])
    parser.add_argument("--project-path", "--project_path", type=str)
    parser.add_argument("--project-name", "--project_name", type=str, required=True)
    parser.add_argument(
        "--workflow",
        action="append",
        required=True,
        type=str.lower,
        choices=[w.lower() for w in supported_workflows],
    )
    parser.add_argument("--rl-library", "--rl_library", type=str.lower, required=True, choices=supported_rl_libraries)
    parser.add_argument("--rl-algorithm", "--rl_algorithm", type=str.lower, required=True, choices=rl_algo_choices)

    args, _ = parser.parse_known_args(argv)

    is_external = args.task_type.lower() == "external"
    if is_external:
        if not args.project_path:
            raise ValueError("--project-path is required for External task type")
        _validate_external_path(args.project_path)
        project_path = args.project_path
    else:
        project_path = None

    if not args.project_name.isidentifier():
        raise ValueError("--project-name must be a valid identifier (letters, numbers, underscores)")

    workflows = [_parse_workflow_arg(item) for item in args.workflow]
    single_agent = any(wf["type"] == "single-agent" for wf in workflows)
    multi_agent = any(wf["type"] == "multi-agent" for wf in workflows)

    # Filter allowed algorithms per RL library under given workflow capabilities
    algos_map = get_algorithms_per_rl_library(single_agent, multi_agent)
    lib = args.rl_library.strip().lower()
    algo = args.rl_algorithm.strip().lower()
    supported_algos = [a.lower() for a in algos_map.get(lib, [])]
    if algo not in supported_algos:
        allowed = ", ".join(supported_algos) if supported_algos else "none"
        raise ValueError(
            f"Algorithm '{args.rl_algorithm}' is not supported for {lib} under selected workflows. Allowed: {allowed}"
        )

    specification = {
        "external": is_external,
        "path": project_path,
        "name": args.project_name,
        "workflows": workflows,
        "rl_libraries": [{"name": lib, "algorithms": [algo]}],
    }

    generate(specification)


if __name__ == "__main__":
    main()


