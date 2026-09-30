# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project template interactive prompts."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import tomllib

_TEMPLATE_DIR = Path(__file__).parents[4] / "tools" / "template"
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


def _external_specification(
    tmp_path: Path, include_ui_extension: bool = False, initial_content: str = "cartpole"
) -> dict:
    """Create a canonical external-project specification."""
    return {
        "external": True,
        "path": str(tmp_path),
        "name": "test_project",
        "authors": ["Test Author", "Example Organization"],
        "initial_content": initial_content,
        "isaaclab_version": "3.0.0",
        "isaaclab_optional_extras": ["all", "isaacsim", "ov", "ovphysx", "ovrtx", "rerun", "viser"],
        "task_name": "place_vial",
        "robot_name": "so101",
        "include_ui_extension": include_ui_extension,
        "workflows": [{"name": "manager-based", "type": "single-agent"}] if initial_content == "cartpole" else [],
        "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}] if initial_content == "cartpole" else [],
    }


def test_main_collects_canonical_external_project_choices():
    """The prompts must collect project layers and list flagship choices first."""
    handler = mock.Mock(spec=CLIHandler)
    handler.input_select.side_effect = ["External", "Cartpole", "No"]
    handler.input_path.return_value = "/tmp"
    handler.input_text.side_effect = ["test_project", "Test Author, Example Organization", "place_vial", "so101"]
    handler.input_checkbox.side_effect = lambda message, choices: [choices[0]]
    handler.get_choices.side_effect = CLIHandler.get_choices

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE, "CLIHandler", return_value=handler),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main([])

    checkbox_calls = handler.input_checkbox.call_args_list
    assert checkbox_calls[0].kwargs["choices"][0] == "Manager-based | single-agent"
    assert checkbox_calls[1].kwargs["choices"][0] == "rsl_rl"
    assert checkbox_calls[2].kwargs["choices"][0] == "PPO"
    content_prompt = handler.input_select.call_args_list[1]
    assert content_prompt.kwargs["choices"] == ["Cartpole", "Blank"]
    assert content_prompt.kwargs["default"] == "Cartpole"
    ui_prompt = handler.input_select.call_args_list[2]
    assert ui_prompt.kwargs["choices"] == ["No", "Yes"]
    assert ui_prompt.kwargs["default"] == "No"
    specification = generate.call_args.args[0]
    assert specification["task_name"] == "place_vial"
    assert specification["robot_name"] == "so101"
    assert specification["authors"] == ["Test Author", "Example Organization"]
    assert specification["initial_content"] == "cartpole"
    assert specification["include_ui_extension"] is False
    assert specification["isaaclab_version"] == "3.0.0"
    assert specification["isaaclab_source_path"] == _MODULE.ROOT_DIR


def test_main_skips_task_prompts_for_blank_project():
    """Blank projects must collect project metadata without asking for task or RL choices."""
    handler = mock.Mock(spec=CLIHandler)
    handler.input_select.side_effect = ["External", "Blank", "No"]
    handler.input_path.return_value = "/tmp"
    handler.input_text.side_effect = ["empty_project", "Test Author"]

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE, "CLIHandler", return_value=handler),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main([])

    handler.input_checkbox.assert_not_called()
    specification = generate.call_args.args[0]
    assert specification["initial_content"] == "blank"
    assert specification["workflows"] == []
    assert specification["rl_libraries"] == []


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            [],
            {
                "initial_content": "cartpole",
                "task_name": "balance",
                "robot_name": "cartpole",
                "workflows": [{"name": "manager-based", "type": "single-agent"}],
                "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
            },
        ),
        (["--initial_content", "blank"], {"workflows": [], "rl_libraries": []}),
        (
            [
                "--workflow",
                "direct:multi-agent",
                "--rl_library",
                "skrl",
                "--rl_algorithm",
                "skrl:ippo",
                "--include_ui_extension",
            ],
            {
                "workflows": [{"name": "direct", "type": "multi-agent"}],
                "rl_libraries": [{"name": "skrl", "algorithms": ["ippo"]}],
                "include_ui_extension": True,
            },
        ),
        (
            ["--workflow", "direct:single-agent", "--workflow", "direct:multi-agent"],
            {
                "workflows": [
                    {"name": "direct", "type": "single-agent"},
                    {"name": "direct", "type": "multi-agent"},
                ],
                "rl_libraries": [{"name": "skrl", "algorithms": ["ppo", "ippo"]}],
            },
        ),
    ],
)
def test_non_interactive_generation(options, expected, tmp_path):
    """Non-interactive mode must support defaults, Blank projects, and explicit selections."""
    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE, "CLIHandler", side_effect=AssertionError("unexpected prompt")),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main(
            [
                "--non_interactive",
                "--project_path",
                str(tmp_path),
                "--name",
                "automated_project",
                "--author",
                "Test Author",
                *options,
            ]
        )

    specification = generate.call_args.args[0]
    assert {key: specification[key] for key in expected} == expected


def test_generation_arguments_require_non_interactive_opt_in(capsys):
    """Automation arguments must not silently change the default interactive flow."""
    with pytest.raises(SystemExit, match="2"):
        _MODULE.main(["--name", "unexpected_project"])

    assert "template arguments require --non_interactive" in capsys.readouterr().err

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        pytest.raises(SystemExit, match="2"),
    ):
        _MODULE.main(
            [
                "--non_interactive",
                "--project_path",
                "/tmp",
                "--name",
                "blank_project",
                "--author",
                "Test Author",
                "--initial_content",
                "blank",
                "--workflow",
                "direct:single-agent",
            ]
        )

    assert "--workflow cannot be used with --initial_content blank" in capsys.readouterr().err


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
    assert project_config["project"]["dependencies"] == ["isaaclab[rsl-rl]==3.0.0"]
    assert project_config["project"]["authors"] == [
        {"name": "Test Author"},
        {"name": "Example Organization"},
    ]
    assert project_config["project"]["entry-points"]["isaaclab.tasks"] == {"test_project": "test_project.tasks"}
    assert project_config["project"]["optional-dependencies"] == {
        extra: [f"isaaclab[{extra}]==3.0.0"]
        for extra in ("all", "isaacsim", "ov", "ovphysx", "ovrtx", "rerun", "viser")
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
    assert "Test Author, Example Organization" in (project_dir / "LICENSE").read_text()
    assert (project_dir / "tests" / "test_registration.py").is_file()
    assert "command_list_envs" in (project_dir / "scripts" / "list_envs.py").read_text()
    assert "uv run isaaclab list_envs --show_presets" in (project_dir / "README.md").read_text()
    assert (task_dir / "env_cfg.py").is_file()
    assert not (task_dir / "env.py").exists()
    assert (task_dir / "agents" / "rsl_rl_ppo_cfg.py").is_file()
    assert not (project_dir / "source").exists()
    assert not (project_dir / "config" / "extension.toml").exists()
    assert not (module_dir / "ui_extension_example.py").exists()
    assert "from .tasks import" not in (module_dir / "__init__.py").read_text()
    assets_module = (module_dir / "assets" / "__init__.py").read_text()
    assert 'TEST_PROJECT_ASSETS_DIR = Path(__file__).resolve().parent / "data"' in assets_module
    assert (module_dir / "assets" / "data").is_dir()
    assert "from test_project.assets import TEST_PROJECT_ASSETS_DIR" in (project_dir / "README.md").read_text()
    pyproject_text = (project_dir / "pyproject.toml").read_text()
    assert '\n[test_project.entry-points."isaaclab.tasks"]' not in pyproject_text
    assert '\n[project.entry-points."isaaclab.tasks"]\ntest_project = "test_project.tasks"' in pyproject_text
    assert "RigidBodyPropertiesCfg" not in (task_dir / "env_cfg.py").read_text()
    assert "ArticulationRootPropertiesCfg" not in (task_dir / "env_cfg.py").read_text()


def test_generated_project_uses_active_source_checkout(tmp_path):
    """Source-generated projects must use the checkout instead of requiring an unpublished wheel."""
    specification = _external_specification(tmp_path)
    specification["isaaclab_source_path"] = _MODULE.ROOT_DIR

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    with (project_dir / "pyproject.toml").open("rb") as file:
        project_config = tomllib.load(file)

    assert project_config["project"]["dependencies"] == ["isaaclab-dev[rsl-rl]"]
    expected_root = Path(_MODULE.ROOT_DIR)
    with (expected_root / "pyproject.toml").open("rb") as file:
        source_config = tomllib.load(file)
    assert project_config["project"]["optional-dependencies"] == {
        extra: [f"isaaclab-dev[{extra}]"] for extra in source_config["project"]["optional-dependencies"]
    }
    sources = project_config["tool"]["uv"]["sources"]
    assert (project_dir / sources["isaaclab-dev"]["path"]).resolve() == expected_root.resolve()
    assert (project_dir / sources["isaaclab"]["path"]).resolve() == (expected_root / "source" / "isaaclab").resolve()
    assert sources["isaaclab-dev"]["editable"] is True
    assert sources["isaaclab"]["editable"] is True
    assert project_config["tool"]["uv"]["override-dependencies"] == source_config["tool"]["uv"]["override-dependencies"]
    assert project_config["tool"]["uv"]["environments"] == source_config["tool"]["uv"]["environments"]
    assert sources["torch"] == source_config["tool"]["uv"]["sources"]["torch"]
    assert "uses editable relative paths" in (project_dir / "README.md").read_text()


def test_installed_project_discovers_all_distribution_extras(tmp_path):
    """Wheel-generated projects must forward every extra declared in package metadata."""
    specification = _external_specification(tmp_path)
    specification.pop("isaaclab_optional_extras")
    distribution = mock.Mock()
    distribution.metadata.get_all.return_value = ["viser", "all", "rerun", "rerun"]

    with mock.patch.object(_GENERATOR.importlib.metadata, "distribution", return_value=distribution) as lookup:
        prepared = _GENERATOR._prepare_external_dependencies(specification, str(tmp_path / "test_project"))

    lookup.assert_called_once_with("isaaclab")
    assert prepared["isaaclab_optional_extras"] == ["all", "rerun", "viser"]


def test_generated_blank_project_contains_no_example_task(tmp_path):
    """A blank project must include packaging infrastructure without cart-pole task files."""
    specification = _external_specification(tmp_path, initial_content="blank")

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    tasks_dir = project_dir / "src" / "test_project" / "tasks"
    assert [path.name for path in tasks_dir.iterdir()] == ["__init__.py"]
    assert (project_dir / "src" / "test_project" / "assets" / "__init__.py").is_file()
    assert (project_dir / "src" / "test_project" / "assets" / "data" / ".gitkeep").is_file()
    assert "No tasks are registered yet" in (project_dir / "README.md").read_text()
    assert "cartpole" not in (project_dir / "README.md").read_text().lower()
    compile((project_dir / "tests" / "test_registration.py").read_text(), "test_registration.py", "exec")
    with (project_dir / "pyproject.toml").open("rb") as file:
        project_config = tomllib.load(file)
    assert project_config["project"]["dependencies"] == ["isaaclab==3.0.0"]


def test_source_path_falls_back_to_absolute_path_across_windows_drives(monkeypatch):
    """A project on another Windows drive must still receive a valid source path."""
    monkeypatch.setattr(_GENERATOR.os.path, "relpath", mock.Mock(side_effect=ValueError))
    monkeypatch.setattr(_GENERATOR.os.path, "realpath", lambda path: path)

    assert _GENERATOR._project_source_path(r"D:\IsaacLab", r"C:\projects\robot") == "D:/IsaacLab"


def test_generated_project_can_opt_into_ui_extension(tmp_path):
    """Direct tasks and UI extension files must follow the canonical project layout."""
    specification = _external_specification(tmp_path, include_ui_extension=True)
    specification["authors"] = ['Test "Quoted" Author']
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
    assert extension_config["package"]["author"] == 'Test "Quoted" Author'
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
