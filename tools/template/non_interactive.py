# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import os

from common import ROOT_DIR
from generator import generate, get_algorithms_per_rl_library


def _parse_workflow_arg(item: str) -> dict[str, str]:
    raw = item.strip().lower()
    # Enforce strict underscore format: "<name>_<type>"
    if "_" not in raw or any(sep in raw for sep in ("|", ":", " ")):
        raise ValueError(
            "Invalid workflow format. Use underscore format like 'direct_single_agent' or 'manager-based_single_agent'"
        )
    name_token, type_token_raw = raw.split("_", 1)
    type_token = type_token_raw.replace("_", "-")  # normalize to single-agent / multi-agent

    if name_token not in {"direct", "manager-based"}:
        raise ValueError(f"Invalid workflow name: {name_token}. Allowed: 'direct' or 'manager-based'")
    if type_token not in {"single-agent", "multi-agent"}:
        raise ValueError(f"Invalid workflow type: {type_token}. Allowed: 'single-agent' or 'multi-agent'")

    return {"name": name_token, "type": type_token}


def _validate_external_path(path: str) -> None:
    if os.path.abspath(path).startswith(os.path.abspath(ROOT_DIR)):
        raise ValueError("External project path cannot be within the Isaac Lab project")


def main(argv: list[str] | None = None) -> None:
    """
    Non-interactive entrypoint for the template generator workflow.

    Parses command-line flags, builds the specification dict, and calls generate().
    This avoids any interactive prompts or dependencies on Inquirer-based flow.
    """

    parser = argparse.ArgumentParser(add_help=False)
    supported_workflows = [
        "direct_single_agent",
        "direct_multi_agent",
        "manager-based_single_agent",
    ]
    supported_rl_libraries = ["rl_games", "rsl_rl", "skrl", "sb3"]
    # All known algorithms across libraries (lowercase for consistent CLI input)
    _all_algos_map = get_algorithms_per_rl_library(True, True)
    rl_algo_choices = sorted({algo.lower() for algos in _all_algos_map.values() for algo in algos})

    parser.add_argument("--task-type", "--task_type", type=str, required=True, choices=["External", "Internal"])
    parser.add_argument("--project-path", "--project_path", type=str)
    parser.add_argument("--project-name", "--project_name", type=str, required=True)
    parser.add_argument(
        "--workflow",
        action="append",
        required=True,
        type=str.lower,
        choices=[*([w.lower() for w in supported_workflows]), "all"],
    )
    parser.add_argument(
        "--rl-library",
        "--rl_library",
        type=str.lower,
        required=True,
        choices=[*supported_rl_libraries, "all"],
    )
    parser.add_argument(
        "--rl-algorithm",
        "--rl_algorithm",
        type=str.lower,
        required=False,
        default=None,
        choices=[*rl_algo_choices, "all"],
        help=(
            "RL algorithm to use. If omitted, the tool auto-selects when exactly one algorithm "
            "is valid for the chosen workflows and library."
        ),
    )

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

    # Expand workflows: allow "all" to mean all supported workflows
    if any(item == "all" for item in args.workflow):
        workflows = [_parse_workflow_arg(item) for item in supported_workflows]
    else:
        workflows = [_parse_workflow_arg(item) for item in args.workflow]
    single_agent = any(wf["type"] == "single-agent" for wf in workflows)
    multi_agent = any(wf["type"] == "multi-agent" for wf in workflows)

    # Filter allowed algorithms per RL library under given workflow capabilities
    algos_map = get_algorithms_per_rl_library(single_agent, multi_agent)

    # Expand RL libraries: allow "all" to mean all libraries that have at least one supported algorithm
    rl_lib_input = args.rl_library.strip().lower()
    if rl_lib_input == "all":
        selected_libs = [lib for lib, algos in algos_map.items() if len(algos) > 0]
        if not selected_libs:
            raise ValueError(
                "No RL libraries are supported under the selected workflows. Please choose different workflows."
            )
    else:
        selected_libs = [rl_lib_input]
        if rl_lib_input not in algos_map:
            raise ValueError(f"Unknown RL library: {rl_lib_input}")
    # Pre-compute supported algorithms per selected library (lowercased)
    supported_algos_per_lib = {lib: [a.lower() for a in algos_map.get(lib, [])] for lib in selected_libs}

    # Auto-select algorithm if not provided
    rl_algo_input = args.rl_algorithm.strip().lower() if args.rl_algorithm is not None else None

    rl_libraries_spec = []
    if rl_algo_input is None:
        # If a single library is selected, preserve previous behavior
        if len(selected_libs) == 1:
            lib = selected_libs[0]
            supported_algos = supported_algos_per_lib.get(lib, [])
            if len(supported_algos) == 0:
                raise ValueError(
                    f"No algorithms are supported for {lib} under the selected workflows. "
                    "Please choose a different combination."
                )
            if len(supported_algos) > 1:
                allowed = ", ".join(supported_algos)
                raise ValueError(
                    "Multiple algorithms are valid for the selected workflows and library. "
                    f"Please specify one using --rl-algorithm or use --rl-algorithm all. Allowed: {allowed}"
                )
            rl_libraries_spec.append({"name": lib, "algorithms": [supported_algos[0]]})
        else:
            # Multiple libraries selected. If each has exactly one algorithm, auto-select; otherwise require explicit choice.
            libs_with_multi = [lib for lib, algos in supported_algos_per_lib.items() if len(algos) > 1]
            if libs_with_multi:
                details = "; ".join(f"{lib}: {', '.join(supported_algos_per_lib[lib])}" for lib in libs_with_multi)
                raise ValueError(
                    "Multiple algorithms are valid for one or more libraries under the selected workflows. "
                    "Please specify --rl-algorithm or use --rl-algorithm all. Details: "
                    + details
                )
            for lib, algos in supported_algos_per_lib.items():
                if not algos:
                    continue
                rl_libraries_spec.append({"name": lib, "algorithms": [algos[0]]})
    elif rl_algo_input == "all":
        # Include all supported algorithms per selected library
        for lib, algos in supported_algos_per_lib.items():
            if not algos:
                continue
            rl_libraries_spec.append({"name": lib, "algorithms": algos})
        if not rl_libraries_spec:
            raise ValueError("No algorithms are supported under the selected workflows.")
    else:
        # Specific algorithm requested: include only libraries that support it
        matching_libs = []
        for lib, algos in supported_algos_per_lib.items():
            if rl_algo_input in algos:
                matching_libs.append(lib)
                rl_libraries_spec.append({"name": lib, "algorithms": [rl_algo_input]})
        if not matching_libs:
            allowed_desc = {lib: algos for lib, algos in supported_algos_per_lib.items() if algos}
            raise ValueError(
                f"Algorithm '{args.rl_algorithm}' is not supported under the selected workflows for the chosen"
                f" libraries. Supported per library: {allowed_desc}"
            )

    specification = {
        "external": is_external,
        "path": project_path,
        "name": args.project_name,
        "workflows": workflows,
        "rl_libraries": rl_libraries_spec,
    }

    generate(specification)


if __name__ == "__main__":
    main()
