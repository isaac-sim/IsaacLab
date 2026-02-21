# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import enum
import importlib
import os
from collections.abc import Callable

import rich.console
import rich.table
from common import ROOT_DIR
from generator import generate, get_algorithms_per_rl_library
from InquirerPy import inquirer, separator


class CLIHandler:
    """CLI handler for the Isaac Lab template."""

    def __init__(self):
        self.console = rich.console.Console()

    @staticmethod
    def get_choices(choices: list[str], default: list[str]) -> list[str]:
        return default if "all" in choices or "both" in choices else choices

    def output_table(self, table: rich.table.Table, new_line_start: bool = True) -> None:
        """Print a rich table to the console.

        Args:
            table: The table to print.
            new_line_start: Whether to print a new line before the table.
        """
        self.console.print(table, new_line_start=new_line_start)

    def input_select(
        self, message: str, choices: list[str], default: str | None = None, long_instruction: str = ""
    ) -> str:
        """Prompt the user to select an option from a list of choices.

        Args:
            message: The message to display to the user.
            choices: The list of choices to display to the user.
            default: The default choice.
            long_instruction: The long instruction to display to the user.

        Returns:
            str: The selected choice.
        """
        return inquirer.select(
            message=message,
            choices=choices,
            cycle=True,
            default=default,
            style=None,
            wrap_lines=True,
            long_instruction=long_instruction,
        ).execute()

    def input_checkbox(self, message: str, choices: list[str], default: str | None = None) -> list[str]:
        """Prompt the user to select one or more options from a list of choices.

        Args:
            message: The message to display to the user.
            choices: The list of choices to display to the user.
            default: The default choice.

        Returns:
            The selected choices.
        """

        def transformer(result: list[str]) -> str:
            if "all" in result or "both" in result:
                token = "all" if "all" in result else "both"
                return f"{token} ({', '.join(choices[: choices.index('---')])})"
            return ", ".join(result)

        return inquirer.checkbox(
            message=message,
            choices=[separator.Separator() if "---" in item else item for item in choices],
            cycle=True,
            default=default,
            style=None,
            wrap_lines=True,
            validate=lambda result: len(result) >= 1,
            invalid_message="No option selected (SPACE: select/deselect an option, ENTER: confirm selection)",
            transformer=transformer,
        ).execute()

    def input_path(
        self,
        message: str,
        default: str | None = None,
        validate: Callable[[str], bool] | None = None,
        invalid_message: str = "",
    ) -> str:
        """Prompt the user to input a path.

        Args:
            message: The message to display to the user.
            default: The default path.
            validate: A callable to validate the path.
            invalid_message: The message to display to the user if the path is invalid.

        Returns:
            The input path.
        """
        return inquirer.filepath(
            message=message,
            default=default if default is not None else "",
            validate=validate,
            invalid_message=invalid_message,
        ).execute()

    def input_text(
        self,
        message: str,
        default: str | None = None,
        validate: Callable[[str], bool] | None = None,
        invalid_message: str = "",
    ) -> str:
        """Prompt the user to input a text.

        Args:
            message: The message to display to the user.
            default: The default text.
            validate: A callable to validate the text.
            invalid_message: The message to display to the user if the text is invalid.

        Returns:
            The input text.
        """
        return inquirer.text(
            message=message,
            default=default if default is not None else "",
            validate=validate,
            invalid_message=invalid_message,
        ).execute()


class State(str, enum.Enum):
    Yes = "[green]yes[/green]"
    No = "[red]no[/red]"


def main() -> None:
    """Main function to run template generation from CLI."""
    cli_handler = CLIHandler()

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-n", "--non-interactive", action="store_true")
    parser.add_argument("--rl-library", dest="rl_library", type=str, default=None)
    parser.add_argument("--rl-algorithm", dest="rl_algorithm", type=str, default=None)
    parser.add_argument(
        "--task-type",
        dest="task_type",
        type=str,
        choices=["External", "Internal", "external", "internal"],
        default=None,
    )
    parser.add_argument("--project-path", dest="project_path", type=str, default=None)
    parser.add_argument("--project-name", dest="project_name", type=str, default=None)
    parser.add_argument("--workflow", dest="workflow", type=str, default=None)
    args = parser.parse_args()

    lab_module = importlib.import_module("isaaclab")
    lab_path = os.path.realpath(getattr(lab_module, "__file__", "") or (getattr(lab_module, "__path__", [""])[0]))
    is_lab_pip_installed = ("site-packages" in lab_path) or ("dist-packages" in lab_path)

    if not is_lab_pip_installed:
        # project type
        if args.non_interactive:
            if args.task_type is not None:
                is_external_project = args.task_type.lower() == "external"
            else:
                is_external_project = True
        else:
            is_external_project = (
                cli_handler.input_select(
                    "Task type:",
                    choices=["External", "Internal"],
                    long_instruction=(
                        "External (recommended): task/project is in its own folder/repo outside the Isaac Lab"
                        " project.\nInternal: the task is implemented within the Isaac Lab project (in"
                        " source/isaaclab_tasks)."
                    ),
                ).lower()
                == "external"
            )
    else:
        is_external_project = True

    # project path (if 'external')
    project_path = None
    if is_external_project:
        if args.non_interactive:
            project_path = args.project_path
            if project_path is None:
                raise SystemExit("In non-interactive mode, --project_path is required for External task type.")
            if os.path.abspath(project_path).startswith(os.path.abspath(ROOT_DIR)):
                raise SystemExit("External project path cannot be within the Isaac Lab project")
        else:
            project_path = cli_handler.input_path(
                "Project path:",
                default=os.path.dirname(ROOT_DIR) + os.sep,
                validate=lambda path: not os.path.abspath(path).startswith(os.path.abspath(ROOT_DIR)),
                invalid_message="External project path cannot be within the Isaac Lab project",
            )

    # project/task name
    if args.non_interactive:
        project_name = args.project_name
        if project_name is None or not project_name.isidentifier():
            raise SystemExit("In non-interactive mode, --project_name is required and must be a valid identifier")
    else:
        project_name = cli_handler.input_text(
            "Project name:" if is_external_project else "Task's folder name:",
            validate=lambda name: name.isidentifier(),
            invalid_message=(
                "Project/task name must be a valid identifier (Letters, numbers and underscores only. No spaces, etc.)"
            ),
        )

    # Isaac Lab workflow
    # - show supported workflows and features
    workflow_table = rich.table.Table(title="RL environment features support according to Isaac Lab workflows")
    workflow_table.add_column("Environment feature", no_wrap=True)
    workflow_table.add_column("Direct", justify="center")
    workflow_table.add_column("Manager-based", justify="center")
    workflow_table.add_row("Single-agent", State.Yes, State.Yes)
    workflow_table.add_row("Multi-agent", State.Yes, State.No)
    workflow_table.add_row("Fundamental/composite spaces (apart from 'Box')", State.Yes, State.No)
    cli_handler.output_table(workflow_table)
    # - prompt for workflows
    supported_workflows = ["Direct | single-agent", "Direct | multi-agent", "Manager-based | single-agent"]
    if args.non_interactive:
        if args.workflow is not None:
            selected_workflows = [item.strip() for item in args.workflow.split(",") if item.strip()]
            if any(item.lower() == "all" for item in selected_workflows):
                workflow = supported_workflows
            else:
                selected_workflows = [item for item in selected_workflows if item in supported_workflows]
                if not selected_workflows:
                    raise SystemExit("No valid --workflow provided for the selected workflows")
                workflow = selected_workflows
        else:
            workflow = supported_workflows
    else:
        workflow = cli_handler.get_choices(
            cli_handler.input_checkbox("Isaac Lab workflow:", choices=[*supported_workflows, "---", "all"]),
            default=supported_workflows,
        )
    workflow = [{"name": item.split(" | ")[0].lower(), "type": item.split(" | ")[1].lower()} for item in workflow]
    single_agent_workflow = [item for item in workflow if item["type"] == "single-agent"]
    multi_agent_workflow = [item for item in workflow if item["type"] == "multi-agent"]

    # RL library
    rl_library_algorithms = []
    algorithms_per_rl_library = get_algorithms_per_rl_library()
    # - show supported RL libraries and features
    rl_library_table = rich.table.Table(title="Supported RL libraries")
    rl_library_table.add_column("RL/training feature", no_wrap=True)
    rl_library_table.add_column("rl_games")
    rl_library_table.add_column("rsl_rl")
    rl_library_table.add_column("skrl")
    rl_library_table.add_column("sb3")
    rl_library_table.add_row("ML frameworks", "PyTorch", "PyTorch", "PyTorch, JAX", "PyTorch")
    rl_library_table.add_row("Relative performance", "~1X", "~1X", "~1X", "~0.03X")
    rl_library_table.add_row(
        "Algorithms",
        ", ".join(algorithms_per_rl_library.get("rl_games", [])),
        ", ".join(algorithms_per_rl_library.get("rsl_rl", [])),
        ", ".join(algorithms_per_rl_library.get("skrl", [])),
        ", ".join(algorithms_per_rl_library.get("sb3", [])),
    )
    rl_library_table.add_row("Multi-agent support", State.No, State.No, State.Yes, State.No)
    rl_library_table.add_row("Distributed training", State.Yes, State.No, State.Yes, State.No)
    rl_library_table.add_row("Vectorized training", State.Yes, State.Yes, State.Yes, State.No)
    rl_library_table.add_row("Fundamental/composite spaces", State.No, State.No, State.Yes, State.No)
    cli_handler.output_table(rl_library_table)
    # - prompt for RL libraries
    supported_rl_libraries = ["rl_games", "rsl_rl", "skrl", "sb3"] if len(single_agent_workflow) else ["skrl"]
    if args.non_interactive:
        if args.rl_library is not None:
            selected_rl_libraries_raw = [item.strip() for item in args.rl_library.split(",") if item.strip()]
            if any(item.lower() == "all" for item in selected_rl_libraries_raw):
                selected_rl_libraries = supported_rl_libraries
            else:
                selected_rl_libraries = [item for item in selected_rl_libraries_raw if item in supported_rl_libraries]
                if not selected_rl_libraries:
                    raise SystemExit("No valid --rl_library provided for the selected workflows")
        else:
            selected_rl_libraries = supported_rl_libraries
    else:
        selected_rl_libraries = cli_handler.get_choices(
            cli_handler.input_checkbox("RL library:", choices=[*supported_rl_libraries, "---", "all"]),
            default=supported_rl_libraries,
        )
    # - prompt for algorithms per RL library
    algorithms_per_rl_library = get_algorithms_per_rl_library(
        bool(len(single_agent_workflow)), bool(len(multi_agent_workflow))
    )
    for rl_library in selected_rl_libraries:
        algorithms = algorithms_per_rl_library.get(rl_library, [])
        if args.non_interactive:
            if args.rl_algorithm is not None:
                provided_algorithms = [item.strip().lower() for item in args.rl_algorithm.split(",") if item.strip()]
                if "all" in provided_algorithms:
                    selected_algorithms = [item.lower() for item in algorithms]
                else:
                    valid_algorithms = [item for item in provided_algorithms if item in [a.lower() for a in algorithms]]
                    if not valid_algorithms:
                        raise SystemExit(f"No valid --rl_algorithm provided for library '{rl_library}'")
                    selected_algorithms = valid_algorithms
            else:
                selected_algorithms = [item.lower() for item in algorithms]
        else:
            if len(algorithms) > 1:
                algorithms = cli_handler.get_choices(
                    cli_handler.input_checkbox(f"RL algorithms for {rl_library}:", choices=[*algorithms, "---", "all"]),
                    default=algorithms,
                )
            selected_algorithms = [item.lower() for item in algorithms]
        rl_library_algorithms.append({"name": rl_library, "algorithms": selected_algorithms})

    specification = {
        "external": is_external_project,
        "path": project_path,
        "name": project_name,
        "workflows": workflow,
        "rl_libraries": rl_library_algorithms,
    }
    generate(specification)


if __name__ == "__main__":
    main()
