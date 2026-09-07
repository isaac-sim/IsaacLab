# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project template interactive prompts."""

import importlib.util
import io
import sys
import types
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


def _external_specification(tmp_path: Path, include_ui_extension: bool = False) -> dict:
    """Create a canonical external-project specification."""
    return {
        "external": True,
        "path": str(tmp_path),
        "name": "test_project",
        "task_name": "place_vial",
        "robot_name": "so101",
        "include_ui_extension": include_ui_extension,
        "workflows": [{"name": "manager-based", "type": "single-agent"}],
        "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
    }


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


def test_main_collects_canonical_external_project_choices():
    """The prompts must collect project layers and list flagship choices first."""
    handler = mock.Mock(spec=CLIHandler)
    handler.input_select.side_effect = ["External", "No"]
    handler.input_path.return_value = "/tmp"
    handler.input_text.side_effect = ["test_project", "place_vial", "so101"]
    handler.input_checkbox.side_effect = lambda message, choices: [choices[0]]
    handler.get_choices.side_effect = CLIHandler.get_choices

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py")
    with (
        mock.patch.object(_MODULE, "CLIHandler", return_value=handler),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main()

    checkbox_calls = handler.input_checkbox.call_args_list
    assert checkbox_calls[0].kwargs["choices"][0] == "Manager-based | single-agent"
    assert checkbox_calls[1].kwargs["choices"][0] == "rsl_rl"
    assert checkbox_calls[2].kwargs["choices"][0] == "PPO"
    ui_prompt = handler.input_select.call_args_list[1]
    assert ui_prompt.kwargs["choices"] == ["No", "Yes"]
    assert ui_prompt.kwargs["default"] == "No"
    specification = generate.call_args.args[0]
    assert specification["task_name"] == "place_vial"
    assert specification["robot_name"] == "so101"
    assert specification["include_ui_extension"] is False


def test_generated_project_matches_canonical_uv_layout(tmp_path):
    """A generated project must use one uv project with a src package and tests."""
    specification = _external_specification(tmp_path)

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    with (project_dir / "pyproject.toml").open("rb") as file:
        project_config = tomllib.load(file)

    assert project_config["build-system"] == {
        "requires": ["uv_build>=0.12.6,<0.13"],
        "build-backend": "uv_build",
    }
    assert project_config["project"]["dependencies"] == ["isaaclab[rsl-rl]"]
    assert project_config["project"]["entry-points"]["isaaclab.tasks"] == {"test_project": "test_project.tasks"}
    assert project_config["project"]["optional-dependencies"] == {
        "isaacsim": ["isaaclab[isaacsim]"],
        "ov": ["isaaclab[ov]"],
        "ovphysx": ["isaaclab[ovphysx]"],
        "ovrtx": ["isaaclab[ovrtx]"],
    }
    assert project_config["dependency-groups"]["dev"] == [
        "codespell>=2.4",
        "pre-commit>=4.2",
        "pytest>=8.3",
        "ruff>=0.11",
    ]
    assert project_config["tool"]["uv"]["build-backend"]["module-name"] == "test_project"
    assert project_config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert project_config["tool"]["pytest"]["ini_options"]["markers"] == [
        "unit: test exercises isolated logic and does not launch the simulator",
        "integration: test drives the simulator/scene/environment end-to-end",
        "smoke: tests for core installation, task, and RL functionality",
        "kitless: test must pass inside the Kit-less container, which has no Isaac Sim runtime",
    ]

    module_dir = project_dir / "src" / "test_project"
    task_dir = module_dir / "tasks" / "place_vial" / "config" / "so101"
    assert (project_dir / "LICENSE").is_file()
    assert (project_dir / "tests" / "test_registration.py").is_file()
    assert 'default="TestProject-"' in (project_dir / "scripts" / "list_envs.py").read_text()
    assert (task_dir / "env_cfg.py").is_file()
    assert (task_dir / "agents" / "rsl_rl_ppo_cfg.py").is_file()
    assert not (project_dir / "source").exists()
    assert not (project_dir / "config" / "extension.toml").exists()
    assert not (module_dir / "ui_extension_example.py").exists()
    assert "from .tasks import" not in (module_dir / "__init__.py").read_text()


def test_generated_project_can_opt_into_ui_extension(tmp_path):
    """Direct tasks and UI extension files must follow the canonical project layout."""
    specification = _external_specification(tmp_path, include_ui_extension=True)
    specification["workflows"] = [{"name": "direct", "type": "single-agent"}]

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    task_dir = project_dir / "src" / "test_project" / "tasks" / "place_vial_direct" / "config" / "so101"
    registration_test = (project_dir / "tests" / "test_registration.py").read_text()
    with (project_dir / "config" / "extension.toml").open("rb") as file:
        extension_config = tomllib.load(file)
    assert (task_dir / "env.py").is_file()
    assert (task_dir / "env_cfg.py").is_file()
    assert "test_project.tasks.place_vial_direct.config.so101.env:PlaceVialEnv" in registration_test
    assert extension_config["python"]["module"] == [{"name": "test_project.ui_extension_example"}]
    assert (project_dir / "src" / "test_project" / "ui_extension_example.py").is_file()


def test_internal_task_keeps_repository_layout(tmp_path):
    """Aligning external projects must not change internal task filenames or layout."""
    specification = {
        "external": False,
        "name": "test_task",
        "workflows": [{"name": "manager-based", "type": "single-agent"}],
        "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
    }

    generated = _GENERATOR._generate_tasks(specification, str(tmp_path))

    task = generated[0]["task"]
    task_dir = tmp_path / "test_task" / "config" / "cartpole"
    assert task["id"] == "Isaac-Test-Task"
    assert (task_dir / "test_task_env_cfg.py").is_file()
