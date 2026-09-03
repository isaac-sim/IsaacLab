# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project template interactive prompts."""

import importlib.util
import io
import sys
from pathlib import Path
from unittest import mock

import tomllib
from rich.console import Console

_TEMPLATE_DIR = Path(__file__).parent
_SPEC = importlib.util.spec_from_file_location("isaaclab_template_cli", _TEMPLATE_DIR / "cli.py")
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_TEMPLATE_DIR))
try:
    _SPEC.loader.exec_module(_MODULE)
finally:
    sys.path.pop(0)

CLIHandler = _MODULE.CLIHandler
_GENERATOR = sys.modules["generator"]


def _handler() -> tuple[CLIHandler, io.StringIO]:
    """Create a prompt handler whose output can be asserted."""
    output = io.StringIO()
    return CLIHandler(Console(file=output, force_terminal=False)), output


def test_select_uses_rich_prompt_and_displays_long_instruction():
    """Single selection must retain explanatory text and return the chosen value."""
    handler, output = _handler()

    with mock.patch.object(_MODULE.Prompt, "ask", return_value="External") as ask:
        result = handler.input_select(
            "Task type:",
            choices=["External", "Internal"],
            long_instruction="External projects live outside Isaac Lab.",
        )

    assert result == "External"
    assert "External projects live outside Isaac Lab." in output.getvalue()
    ask.assert_called_once_with(
        "Task type",
        console=handler.console,
        choices=["External", "Internal"],
        case_sensitive=False,
    )


def test_checkbox_parses_multiple_choices_and_reprompts_invalid_input():
    """Multi-selection must validate numbered input and preserve choice order."""
    handler, output = _handler()

    with mock.patch.object(_MODULE.Prompt, "ask", side_effect=["invalid", "1, 3, 1"]):
        result = handler.input_checkbox("Workflow:", ["Direct", "Manager-based", "---", "all"])

    assert result == ["Direct", "all"]
    assert "Enter one or more valid numbers" in output.getvalue()


def test_text_reprompts_until_validation_succeeds():
    """Text entry must surface the validation message and retry."""
    handler, output = _handler()

    with mock.patch.object(_MODULE.Prompt, "ask", side_effect=["not valid", "valid_name"]):
        result = handler.input_text(
            "Project name:",
            validate=str.isidentifier,
            invalid_message="Project name must be a valid identifier.",
        )

    assert result == "valid_name"
    assert "Project name must be a valid identifier." in output.getvalue()


def test_generated_project_metadata(tmp_path):
    """A generated project must declare optional backends and development tooling."""
    specification = {
        "external": True,
        "path": str(tmp_path),
        "name": "test_project",
        "workflows": [{"name": "manager-based", "type": "single-agent"}],
        "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
    }

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    with (project_dir / "pyproject.toml").open("rb") as file:
        development_config = tomllib.load(file)
    with (project_dir / "source" / "test_project" / "pyproject.toml").open("rb") as file:
        task_package = tomllib.load(file)["project"]

    development_project = development_config["project"]
    assert development_project["dependencies"] == ["test_project"]
    assert development_project["optional-dependencies"] == {
        "isaacsim": ["isaaclab[isaacsim]"],
        "ov": ["isaaclab[ov]"],
        "ovphysx": ["isaaclab[ovphysx]"],
        "ovrtx": ["isaaclab[ovrtx]"],
    }
    assert task_package["dependencies"] == ["isaaclab[rsl-rl]"]
    assert development_config["dependency-groups"]["dev"] == ["pre-commit", "pytest"]
    assert development_config["tool"]["pytest"]["ini_options"]["markers"] == [
        "unit: test exercises isolated logic and does not launch the simulator",
        "integration: test drives the simulator/scene/environment end-to-end",
        "smoke: tests for core installation, task, and RL functionality",
        "kitless: test must pass inside the Kit-less container, which has no Isaac Sim runtime",
    ]
    assert development_config["tool"]["ruff"]["lint"]["per-file-ignores"]["**/__init__.pyi"] == [
        "F401",
        "F403",
    ]
