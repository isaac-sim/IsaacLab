# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
                return f'{token} ({", ".join(choices[:choices.index("---")])})'
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

    lab_module = importlib.import_module("isaaclab")
    lab_path = os.path.realpath(getattr(lab_module, "__file__", "") or (getattr(lab_module, "__path__", [""])[0]))
    is_lab_pip_installed = ("site-packages" in lab_path) or ("dist-packages" in lab_path)

    if not is_lab_pip_installed:
        # project type
        is_external_project = (
            cli_handler.input_select(
                "Task type:",
                choices=["External", "Internal"],
                long_instruction=(
                    "External (recommended): task/project is in its own folder/repo outside the Isaac Lab project.\n"
                    "Internal: the task is implemented within the Isaac Lab project (in source/isaaclab_tasks)."
                ),
            ).lower()
            == "external"
        )
    else:
        is_external_project = True

    # project path (if 'external')
    project_path = None
    if is_external_project:
        project_path = cli_handler.input_path(
            "Project path:",
            default=os.path.dirname(ROOT_DIR) + os.sep,
            validate=lambda path: not os.path.abspath(path).startswith(os.path.abspath(ROOT_DIR)),
            invalid_message="External project path cannot be within the Isaac Lab project",
        )

    # project/task name
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
    selected_rl_libraries = cli_handler.get_choices(
        cli_handler.input_checkbox("RL library:", choices=[*supported_rl_libraries, "---", "all"]),
        default=supported_rl_libraries,
    )
    # - prompt for algorithms per RL library
    algorithms_per_rl_library = get_algorithms_per_rl_library(len(single_agent_workflow), len(multi_agent_workflow))
    for rl_library in selected_rl_libraries:
        algorithms = algorithms_per_rl_library.get(rl_library, [])
        if len(algorithms) > 1:
            algorithms = cli_handler.get_choices(
                cli_handler.input_checkbox(f"RL algorithms for {rl_library}:", choices=[*algorithms, "---", "all"]),
                default=algorithms,
            )
        rl_library_algorithms.append({"name": rl_library, "algorithms": [item.lower() for item in algorithms]})

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
