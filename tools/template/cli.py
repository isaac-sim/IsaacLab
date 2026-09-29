# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import enum
import importlib
import os
from collections.abc import Callable
from pathlib import Path
from textwrap import fill

import rich.console
import rich.table
from common import MULTI_AGENT_ALGORITHMS, ROOT_DIR, SINGLE_AGENT_ALGORITHMS
from generator import generate, get_algorithms_per_rl_library
from rich.prompt import Prompt

_SUPPORTED_WORKFLOWS = ["Manager-based | single-agent", "Direct | single-agent", "Direct | multi-agent"]
_SINGLE_AGENT_RL_LIBRARIES = ["rsl_rl", "rl_games", "skrl", "sb3"]
_NON_INTERACTIVE_WORKFLOWS = {
    "manager-based:single-agent": {"name": "manager-based", "type": "single-agent"},
    "direct:single-agent": {"name": "direct", "type": "single-agent"},
    "direct:multi-agent": {"name": "direct", "type": "multi-agent"},
}


def _is_external_path(path: str) -> bool:
    return not Path(path).resolve().is_relative_to(Path(ROOT_DIR).resolve())


class CLIHandler:
    """CLI handler for the Isaac Lab template."""

    def __init__(self, console: rich.console.Console | None = None):
        self.console = console if console is not None else rich.console.Console()

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
        if long_instruction:
            self.console.print(long_instruction, markup=False)
        return self._ask(message, choices=choices, default=default)

    def input_checkbox(self, message: str, choices: list[str], default: str | None = None) -> list[str]:
        """Prompt the user to select one or more options from a list of choices.

        Args:
            message: The message to display to the user.
            choices: The list of choices to display to the user.
            default: The default choice.

        Returns:
            The selected choices.
        """

        selectable_choices = [choice for choice in choices if choice != "---"]
        for index, choice in enumerate(selectable_choices, start=1):
            self.console.print(f"  [cyan]{index}[/cyan].", choice)

        default_index = None
        if default is not None and default in selectable_choices:
            default_index = str(selectable_choices.index(default) + 1)

        while True:
            response = self._ask(
                f"{message} Enter comma-separated numbers",
                default=default_index,
            )
            try:
                indices = [int(token.strip()) for token in response.split(",")]
            except ValueError:
                indices = []
            if indices and all(1 <= index <= len(selectable_choices) for index in indices):
                return list(dict.fromkeys(selectable_choices[index - 1] for index in indices))
            self.console.print("Enter one or more valid numbers separated by commas.", style="red")

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
        return self._input_value(message, default, validate, invalid_message)

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
        return self._input_value(message, default, validate, invalid_message)

    def _ask(
        self,
        message: str,
        choices: list[str] | None = None,
        default: str | None = None,
    ) -> str:
        """Prompt for a string with optional choices and default value."""
        kwargs = {"console": self.console, "choices": choices, "case_sensitive": False}
        if default is not None:
            kwargs["default"] = default
        return Prompt.ask(message.removesuffix(":"), **kwargs)

    def _input_value(
        self,
        message: str,
        default: str | None,
        validate: Callable[[str], bool] | None,
        invalid_message: str,
    ) -> str:
        """Prompt until the entered value passes validation."""
        while True:
            value = self._ask(message, default=default)
            if validate is None or validate(value):
                return value
            self.console.print(invalid_message or "Invalid input.", style="red")


class State(str, enum.Enum):
    Yes = "[green]yes[/green]"
    No = "[red]no[/red]"


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create the template generator argument parser."""
    parser = argparse.ArgumentParser(description="Create an Isaac Lab project or task from a template.")
    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="Generate without prompts using command-line arguments.",
    )
    parser.add_argument("--task_type", choices=("external", "internal"), help="Where to create the task.")
    parser.add_argument("--project_path", help="Parent directory for an external project.")
    parser.add_argument("--name", "--project_name", dest="name", help="Project name or internal task folder name.")
    parser.add_argument(
        "--author",
        action="append",
        help="Project author. Repeat this option to specify multiple authors.",
    )
    parser.add_argument("--initial_content", choices=("blank", "cartpole"), help="Initial external project content.")
    parser.add_argument("--task_name", help="Task family name for the Cartpole example.")
    parser.add_argument("--robot_name", help="Robot/config name for the Cartpole example.")
    parser.add_argument(
        "--include_ui_extension",
        action="store_true",
        default=None,
        help="Include an Isaac Sim UI extension in an external project.",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        choices=tuple(_NON_INTERACTIVE_WORKFLOWS),
        help="Task workflow. Repeat this option to generate several workflows.",
    )
    parser.add_argument(
        "--rl_library",
        action="append",
        choices=tuple(_SINGLE_AGENT_RL_LIBRARIES),
        help="RL library. Repeat this option to include several libraries.",
    )
    parser.add_argument(
        "--rl_algorithm",
        action="append",
        help=(
            "RL algorithm, such as 'ppo', or a library-qualified value, such as 'skrl:ippo'. "
            "Repeat this option to include several algorithms."
        ),
    )
    return parser


def _isaaclab_installation() -> tuple[object, bool]:
    """Return the imported Isaac Lab module and whether it came from an installed wheel."""
    lab_module = importlib.import_module("isaaclab")
    lab_path = os.path.realpath(getattr(lab_module, "__file__", "") or (getattr(lab_module, "__path__", [""])[0]))
    return lab_module, ("site-packages" in lab_path) or ("dist-packages" in lab_path)


def _collect_interactive_specification(lab_module: object, is_lab_pip_installed: bool) -> dict:
    """Collect a project specification through interactive prompts."""
    cli_handler = CLIHandler()

    if not is_lab_pip_installed:
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

    project_path = None
    if is_external_project:
        project_path = cli_handler.input_path(
            "Project path:",
            default=os.path.dirname(ROOT_DIR) + os.sep,
            validate=_is_external_path,
            invalid_message="External project path cannot be within the Isaac Lab project",
        )

    project_name = cli_handler.input_text(
        "Project name:" if is_external_project else "Task's folder name:",
        validate=lambda name: name.isascii() and name.isidentifier(),
        invalid_message=(
            "Project/task name must be a valid identifier (Letters, numbers and underscores only. No spaces, etc.)"
        ),
    )
    if is_external_project:
        authors = [
            author.strip()
            for author in cli_handler.input_text(
                "Author name(s), comma-separated:",
                validate=lambda value: bool(value.strip()) and all(part.strip() for part in value.split(",")),
                invalid_message="Enter at least one author name; separate multiple names with commas.",
            ).split(",")
        ]
        initial_content = cli_handler.input_select(
            "Initial project content:",
            choices=["Cartpole", "Blank"],
            default="Cartpole",
            long_instruction=(
                "Cartpole creates a runnable example task. Blank creates only the project structure and tooling."
            ),
        ).lower()
        if initial_content == "cartpole":
            task_name = cli_handler.input_text(
                "Task family name:",
                default="balance",
                validate=lambda name: name.isascii() and name.isidentifier(),
                invalid_message="Task family name must be a valid Python identifier.",
            )
            robot_name = cli_handler.input_text(
                "Robot/config name:",
                default="cartpole",
                validate=lambda name: name.isascii() and name.isidentifier(),
                invalid_message="Robot/config name must be a valid Python identifier.",
            )
        else:
            task_name = "balance"
            robot_name = "cartpole"
        include_ui_extension = (
            cli_handler.input_select(
                "Include Isaac Sim UI extension:",
                choices=["No", "Yes"],
                default="No",
                long_instruction=(
                    "Choose Yes only if this project needs an extension loaded through the Isaac Sim Extension Manager."
                ),
            ).lower()
            == "yes"
        )
    else:
        authors = []
        initial_content = "cartpole"
        task_name = project_name
        robot_name = "cartpole"
        include_ui_extension = False

    workflow = []
    rl_library_algorithms = []
    if initial_content == "cartpole":
        workflow_table = rich.table.Table(title="RL environment features support according to Isaac Lab workflows")
        workflow_table.add_column("Environment feature", no_wrap=True)
        workflow_table.add_column("Manager-based", justify="center")
        workflow_table.add_column("Direct", justify="center")
        workflow_table.add_row("Single-agent", State.Yes, State.Yes)
        workflow_table.add_row("Multi-agent", State.No, State.Yes)
        workflow_table.add_row("Fundamental/composite spaces (apart from 'Box')", State.No, State.Yes)
        cli_handler.output_table(workflow_table)
        workflow = cli_handler.get_choices(
            cli_handler.input_checkbox("Isaac Lab workflow:", choices=[*_SUPPORTED_WORKFLOWS, "---", "all"]),
            default=_SUPPORTED_WORKFLOWS,
        )
        workflow = [{"name": item.split(" | ")[0].lower(), "type": item.split(" | ")[1].lower()} for item in workflow]
        single_agent_workflow = [item for item in workflow if item["type"] == "single-agent"]
        multi_agent_workflow = [item for item in workflow if item["type"] == "multi-agent"]

        algorithms_per_rl_library = get_algorithms_per_rl_library()
        rl_library_table = rich.table.Table(title="Supported RL libraries")
        rl_library_table.add_column("RL/training feature", no_wrap=True)
        rl_library_table.add_column("rsl_rl", overflow="fold")
        rl_library_table.add_column("rl_games", overflow="fold")
        rl_library_table.add_column("skrl", overflow="fold")
        rl_library_table.add_column("sb3", overflow="fold")
        rl_library_table.add_row("ML frameworks", "PyTorch", "PyTorch", "PyTorch, JAX", "PyTorch")
        rl_library_table.add_row("Relative performance", "~1X", "~1X", "~1X", "~0.03X")
        rl_library_table.add_row(
            "Algorithms",
            fill(", ".join(algorithms_per_rl_library.get("rsl_rl", [])), width=12, break_long_words=False),
            fill(", ".join(algorithms_per_rl_library.get("rl_games", [])), width=12, break_long_words=False),
            fill(", ".join(algorithms_per_rl_library.get("skrl", [])), width=12, break_long_words=False),
            fill(", ".join(algorithms_per_rl_library.get("sb3", [])), width=12, break_long_words=False),
        )
        rl_library_table.add_row("Multi-agent support", State.No, State.No, State.Yes, State.No)
        rl_library_table.add_row("Distributed training", State.Yes, State.Yes, State.Yes, State.No)
        rl_library_table.add_row("Vectorized training", State.Yes, State.Yes, State.Yes, State.No)
        rl_library_table.add_row("Fundamental/composite spaces", State.No, State.No, State.Yes, State.No)
        cli_handler.output_table(rl_library_table)
        supported_rl_libraries = _SINGLE_AGENT_RL_LIBRARIES if len(single_agent_workflow) else ["skrl"]
        selected_rl_libraries = cli_handler.get_choices(
            cli_handler.input_checkbox("RL library:", choices=[*supported_rl_libraries, "---", "all"]),
            default=supported_rl_libraries,
        )
        algorithms_per_rl_library = get_algorithms_per_rl_library(len(single_agent_workflow), len(multi_agent_workflow))
        for rl_library in selected_rl_libraries:
            algorithms = algorithms_per_rl_library.get(rl_library, [])
            if len(algorithms) > 1:
                algorithms = cli_handler.get_choices(
                    cli_handler.input_checkbox(f"RL algorithms for {rl_library}:", choices=[*algorithms, "---", "all"]),
                    default=algorithms,
                )
            rl_library_algorithms.append({"name": rl_library, "algorithms": [item.lower() for item in algorithms]})

    return {
        "external": is_external_project,
        "path": project_path,
        "name": project_name,
        "authors": authors,
        "initial_content": initial_content,
        "isaaclab_version": getattr(lab_module, "__version__"),
        "isaaclab_source_path": ROOT_DIR if not is_lab_pip_installed else None,
        "task_name": task_name,
        "robot_name": robot_name,
        "include_ui_extension": include_ui_extension,
        "workflows": workflow,
        "rl_libraries": rl_library_algorithms,
    }


def _collect_non_interactive_specification(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    lab_module: object,
    is_lab_pip_installed: bool,
) -> dict:
    """Build and validate a project specification from command-line arguments."""
    task_type = args.task_type or "external"
    is_external_project = task_type == "external"
    if not is_external_project and is_lab_pip_installed:
        parser.error("--task_type internal requires an Isaac Lab source checkout")
    if not args.name:
        parser.error("--name is required in non-interactive mode")
    if not args.name.isascii() or not args.name.isidentifier():
        parser.error("--name must be an ASCII Python identifier")

    if is_external_project:
        if not args.project_path:
            parser.error("--project_path is required for an external project in non-interactive mode")
        if not _is_external_path(args.project_path):
            parser.error("--project_path for an external project cannot be within the Isaac Lab project")
        if not args.author or any(not author.strip() for author in args.author):
            parser.error("at least one non-empty --author is required for an external project")
        authors = [author.strip() for author in args.author]
        initial_content = args.initial_content or "cartpole"
        include_ui_extension = bool(args.include_ui_extension)
        task_name = args.task_name or "balance"
        robot_name = args.robot_name or "cartpole"
    else:
        invalid_external_options = [
            option
            for option, value in (
                ("--project_path", args.project_path),
                ("--author", args.author),
                ("--initial_content", args.initial_content),
                ("--task_name", args.task_name),
                ("--include_ui_extension", args.include_ui_extension),
            )
            if value is not None
        ]
        if invalid_external_options:
            parser.error(f"{'/'.join(invalid_external_options)} only applies to external projects")
        authors = []
        initial_content = "cartpole"
        include_ui_extension = False
        task_name = args.name
        robot_name = args.robot_name or "cartpole"

    if not task_name.isascii() or not task_name.isidentifier():
        parser.error("--task_name must be an ASCII Python identifier")
    if not robot_name.isascii() or not robot_name.isidentifier():
        parser.error("--robot_name must be an ASCII Python identifier")

    if initial_content == "blank":
        incompatible_options = [
            option
            for option, value in (
                ("--task_name", args.task_name),
                ("--robot_name", args.robot_name),
                ("--workflow", args.workflow),
                ("--rl_library", args.rl_library),
                ("--rl_algorithm", args.rl_algorithm),
            )
            if value is not None
        ]
        if incompatible_options:
            parser.error(f"{'/'.join(incompatible_options)} cannot be used with --initial_content blank")
        workflows = []
        rl_libraries = []
    else:
        workflow_names = list(dict.fromkeys(args.workflow or ["manager-based:single-agent"]))
        workflows = [_NON_INTERACTIVE_WORKFLOWS[name].copy() for name in workflow_names]
        has_single_agent = any(workflow["type"] == "single-agent" for workflow in workflows)
        has_multi_agent = any(workflow["type"] == "multi-agent" for workflow in workflows)
        supported_libraries = _SINGLE_AGENT_RL_LIBRARIES if has_single_agent else ["skrl"]
        default_libraries = ["skrl"] if has_multi_agent else ["rsl_rl"]
        selected_libraries = list(dict.fromkeys(args.rl_library or default_libraries))
        invalid_libraries = [library for library in selected_libraries if library not in supported_libraries]
        if invalid_libraries:
            parser.error(
                f"RL libraries {invalid_libraries} do not support the selected workflows; "
                f"choose from {supported_libraries}"
            )

        default_algorithms = (
            ["ppo", "ippo"] if has_single_agent and has_multi_agent else ["ppo" if has_single_agent else "ippo"]
        )
        requested_algorithms = args.rl_algorithm or default_algorithms
        shared_algorithms = []
        algorithms_by_library: dict[str, list[str]] = {library: [] for library in selected_libraries}
        for value in requested_algorithms:
            selector, separator, algorithm = value.lower().partition(":")
            if separator:
                if selector not in algorithms_by_library:
                    parser.error(f"--rl_algorithm {value!r} refers to an unselected RL library")
                if not algorithm:
                    parser.error(f"--rl_algorithm {value!r} must include an algorithm after ':'")
                algorithms_by_library[selector].append(algorithm)
            else:
                shared_algorithms.append(selector)

        supported_algorithms = get_algorithms_per_rl_library(has_single_agent, has_multi_agent)
        rl_libraries = []
        for library in selected_libraries:
            algorithms = list(dict.fromkeys([*shared_algorithms, *algorithms_by_library[library]]))
            if not algorithms:
                parser.error(f"specify at least one --rl_algorithm for {library}")
            invalid_algorithms = [
                algorithm for algorithm in algorithms if algorithm.upper() not in supported_algorithms[library]
            ]
            if invalid_algorithms:
                parser.error(
                    f"RL algorithms {invalid_algorithms} are not supported by {library}; "
                    f"choose from {[value.lower() for value in supported_algorithms[library]]}"
                )
            rl_libraries.append({"name": library, "algorithms": algorithms})

        selected_algorithms = {algorithm.upper() for library in rl_libraries for algorithm in library["algorithms"]}
        for workflow_type, required_algorithms in (
            ("single-agent", SINGLE_AGENT_ALGORITHMS if has_single_agent else []),
            ("multi-agent", MULTI_AGENT_ALGORITHMS if has_multi_agent else []),
        ):
            if required_algorithms and selected_algorithms.isdisjoint(required_algorithms):
                parser.error(f"select an RL algorithm compatible with the {workflow_type} workflow")

    return {
        "external": is_external_project,
        "path": args.project_path if is_external_project else None,
        "name": args.name,
        "authors": authors,
        "initial_content": initial_content,
        "isaaclab_version": getattr(lab_module, "__version__"),
        "isaaclab_source_path": ROOT_DIR if not is_lab_pip_installed else None,
        "task_name": task_name,
        "robot_name": robot_name,
        "include_ui_extension": include_ui_extension,
        "workflows": workflows,
        "rl_libraries": rl_libraries,
    }


def main(argv: list[str] | None = None) -> None:
    """Run template generation from the command line."""
    parser = _create_argument_parser()
    args = parser.parse_args(argv)
    has_generation_args = any(value is not None for name, value in vars(args).items() if name != "non_interactive")
    if not args.non_interactive and has_generation_args:
        parser.error("template arguments require --non_interactive")

    lab_module, is_lab_pip_installed = _isaaclab_installation()
    if args.non_interactive:
        specification = _collect_non_interactive_specification(
            args,
            parser,
            lab_module,
            is_lab_pip_installed,
        )
    else:
        specification = _collect_interactive_specification(lab_module, is_lab_pip_installed)
    generate(specification)


if __name__ == "__main__":
    main()
