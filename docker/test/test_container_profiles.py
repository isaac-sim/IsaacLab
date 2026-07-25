# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from docker import container as container_cli
from docker.utils import ContainerInterface, volume_mounts

DOCKER_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def container_context(tmp_path: Path) -> Path:
    """Create the profile environment files needed by the container interface."""
    (tmp_path / ".env.base").write_text(
        "\n".join(
            (
                "ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim",
                "ISAACSIM_VERSION=6.0.0",
                "DOCKER_ISAACSIM_ROOT_PATH=/isaac-sim",
                "DOCKER_ISAACLAB_PATH=/workspace/isaaclab",
                "DOCKER_USER_HOME=/root",
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env.ros2").write_text("ROS2_APT_PACKAGE=ros-base\n", encoding="utf-8")
    (tmp_path / ".env.kitless").write_text(
        "\n".join(
            (
                "KITLESS_BASE_IMAGE=ubuntu:24.04",
                "DOCKER_ISAACLAB_PATH=/workspace/isaaclab",
                "DOCKER_USER_HOME=/home/isaaclab",
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / ".isaac-lab-docker-history").touch()
    return tmp_path


@pytest.fixture
def make_interface(container_context: Path) -> Callable[[str], ContainerInterface]:
    """Create container interfaces without persisting a state file."""

    def _make(profile: str) -> ContainerInterface:
        return ContainerInterface(context_dir=container_context, profile=profile, statefile=MagicMock())

    return _make


@pytest.mark.parametrize(
    ("profile", "expected_env_files"),
    (
        ("base", ["--env-file", ".env.base"]),
        ("ros2", ["--env-file", ".env.base", "--env-file", ".env.ros2"]),
        ("kitless", ["--env-file", ".env.kitless"]),
    ),
)
def test_profile_environment_inheritance(
    make_interface: Callable[[str], ContainerInterface], profile: str, expected_env_files: list[str]
):
    """Each profile loads only its intended environment-file chain."""
    interface = make_interface(profile)

    assert interface.add_env_files == expected_env_files
    if profile == "kitless":
        assert "ISAACSIM_BASE_IMAGE" not in interface.dot_vars
        assert interface.dot_vars["DOCKER_USER_HOME"] == "/home/isaaclab"


def test_profile_capabilities(make_interface: Callable[[str], ContainerInterface]):
    """The kit-less profile is standalone and headless while existing profiles retain their behavior."""
    base = make_interface("base")
    ros2 = make_interface("ros2")
    kitless = make_interface("kitless")

    assert not base.requires_base_image
    assert base.supports_x11
    assert ros2.requires_base_image
    assert ros2.supports_x11
    assert not kitless.requires_base_image
    assert not kitless.supports_x11


@pytest.mark.parametrize(
    ("profile", "expected_commands"),
    (
        (
            "base",
            [
                [
                    "docker",
                    "compose",
                    "--file",
                    "docker-compose.yaml",
                    "--profile",
                    "base",
                    "--env-file",
                    ".env.base",
                    "build",
                    "isaac-lab-base",
                ]
            ],
        ),
        (
            "ros2",
            [
                [
                    "docker",
                    "compose",
                    "--file",
                    "docker-compose.yaml",
                    "--profile",
                    "base",
                    "--env-file",
                    ".env.base",
                    "build",
                    "isaac-lab-base",
                ],
                [
                    "docker",
                    "compose",
                    "--file",
                    "docker-compose.yaml",
                    "--profile",
                    "ros2",
                    "--env-file",
                    ".env.base",
                    "--env-file",
                    ".env.ros2",
                    "build",
                    "isaac-lab-ros2",
                ],
            ],
        ),
        (
            "kitless",
            [
                [
                    "docker",
                    "compose",
                    "--file",
                    "docker-compose.yaml",
                    "--profile",
                    "kitless",
                    "--env-file",
                    ".env.kitless",
                    "build",
                    "isaac-lab-kitless",
                ]
            ],
        ),
    ),
)
def test_build_uses_profile_dependency_chain(
    make_interface: Callable[[str], ContainerInterface],
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected_commands: list[list[str]],
):
    """Build only the services required by the selected profile."""
    run = MagicMock()
    monkeypatch.setattr("docker.utils.container_interface.subprocess.run", run)

    make_interface(profile).build()

    assert [call.args[0] for call in run.call_args_list] == expected_commands


def test_kitless_start_does_not_build_base(
    make_interface: Callable[[str], ContainerInterface], monkeypatch: pytest.MonkeyPatch
):
    """Starting kit-less invokes only its standalone Compose profile."""
    run = MagicMock()
    monkeypatch.setattr("docker.utils.container_interface.subprocess.run", run)

    make_interface("kitless").start()

    assert [call.args[0] for call in run.call_args_list] == [
        [
            "docker",
            "compose",
            "--file",
            "docker-compose.yaml",
            "--profile",
            "kitless",
            "--env-file",
            ".env.kitless",
            "up",
            "--detach",
            "--build",
            "--remove-orphans",
        ]
    ]


def test_kitless_enter_and_stop_target_profile_container(
    make_interface: Callable[[str], ContainerInterface], monkeypatch: pytest.MonkeyPatch
):
    """Enter and stop use the kit-less service name and environment."""
    interface = make_interface("kitless")
    run = MagicMock()
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("docker.utils.container_interface.subprocess.run", run)
    monkeypatch.setattr(interface, "is_container_running", MagicMock(return_value=True))

    interface.enter()
    interface.stop()

    assert [call.args[0] for call in run.call_args_list] == [
        ["docker", "exec", "--interactive", "--tty", "isaac-lab-kitless", "bash"],
        [
            "docker",
            "compose",
            "--file",
            "docker-compose.yaml",
            "--profile",
            "kitless",
            "--env-file",
            ".env.kitless",
            "down",
            "--volumes",
        ],
    ]


@pytest.mark.parametrize("command", ("build", "start", "enter", "stop"))
def test_kitless_cli_skips_x11(monkeypatch: pytest.MonkeyPatch, command: str):
    """Kit-less CLI commands do not inspect or mutate the shared X11 state."""
    interface = MagicMock()
    interface.profile = "kitless"
    interface.supports_x11 = False
    interface.add_yamls = ["--file", "docker-compose.yaml"]
    interface.environ = {}
    interface_factory = MagicMock(return_value=interface)
    monkeypatch.setattr(container_cli, "ContainerInterface", interface_factory)
    monkeypatch.setattr(container_cli.shutil, "which", MagicMock(return_value="/usr/bin/docker"))
    x11_check = MagicMock()
    x11_refresh = MagicMock()
    x11_cleanup = MagicMock()
    monkeypatch.setattr(container_cli.x11_utils, "x11_check", x11_check)
    monkeypatch.setattr(container_cli.x11_utils, "x11_refresh", x11_refresh)
    monkeypatch.setattr(container_cli.x11_utils, "x11_cleanup", x11_cleanup)
    args = Namespace(
        command=command,
        profile="kitless",
        files=None,
        env_files=None,
        suffix=None,
        info=False,
    )

    container_cli.main(args)

    getattr(interface, command).assert_called_once_with()
    x11_check.assert_not_called()
    x11_refresh.assert_not_called()
    x11_cleanup.assert_not_called()


def test_existing_profile_cli_keeps_x11(monkeypatch: pytest.MonkeyPatch):
    """Existing profiles still merge the X11 overlay before starting."""
    interface = MagicMock()
    interface.profile = "base"
    interface.supports_x11 = True
    interface.add_yamls = ["--file", "docker-compose.yaml"]
    interface.environ = {}
    monkeypatch.setattr(container_cli, "ContainerInterface", MagicMock(return_value=interface))
    monkeypatch.setattr(container_cli.shutil, "which", MagicMock(return_value="/usr/bin/docker"))
    monkeypatch.setattr(
        container_cli.x11_utils,
        "x11_check",
        MagicMock(return_value=(["--file", "x11.yaml"], {"DISPLAY": ":0"})),
    )
    args = Namespace(
        command="start",
        profile="base",
        files=None,
        env_files=None,
        suffix=None,
        info=False,
    )

    container_cli.main(args)

    assert interface.add_yamls == ["--file", "docker-compose.yaml", "--file", "x11.yaml"]
    assert interface.environ["DISPLAY"] == ":0"
    interface.start.assert_called_once_with()


def test_kitless_compose_service_has_no_isaac_sim_mounts():
    """The kit-less Compose service uses only standalone paths and settings."""
    compose = yaml.safe_load((DOCKER_DIR / "docker-compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["isaac-lab-kitless"]
    mounts = compose["x-kitless-isaac-lab-volumes"]

    assert service["profiles"] == ["kitless"]
    assert service["env_file"] == ".env.kitless"
    assert service["build"]["dockerfile"] == "docker/Dockerfile.kitless"
    assert service["image"] == "isaac-lab-kitless${DOCKER_NAME_SUFFIX-}"
    assert service["container_name"] == "isaac-lab-kitless${DOCKER_NAME_SUFFIX-}"
    assert "environment" not in service
    assert service["volumes"] == mounts

    forbidden_sources = {"isaac-cache-kit", "isaac-data-kit", "isaac-carb-logs"}
    assert forbidden_sources.isdisjoint(mount.get("source") for mount in mounts)
    assert all("DOCKER_ISAACSIM" not in mount["target"] for mount in mounts)
    assert all("/kit/" not in mount["target"].lower() for mount in mounts)


def test_kitless_volume_key_resolves_owned_image_paths(monkeypatch: pytest.MonkeyPatch):
    """The explicit kit-less volume key resolves the paths prepared by its Dockerfile."""
    monkeypatch.setenv("DOCKER_ISAACLAB_PATH", "/workspace/isaaclab")
    monkeypatch.setenv("DOCKER_USER_HOME", "/home/isaaclab")

    targets = volume_mounts.resolved_targets(DOCKER_DIR / "docker-compose.yaml", "x-kitless-isaac-lab-volumes")

    assert targets == [
        "/home/isaaclab/.cache/uv",
        "/home/isaaclab/.cache/warp",
        "/workspace/isaaclab/docs/_build",
        "/workspace/isaaclab/logs",
        "/workspace/isaaclab/data_storage",
    ]


def test_default_volume_key_resolves_compose_defaults(monkeypatch: pytest.MonkeyPatch):
    """Compose defaults keep base volume targets resolvable without loading ``.env.base``."""
    monkeypatch.delenv("DOCKER_ISAACSIM_ROOT_PATH", raising=False)
    monkeypatch.setenv("DOCKER_ISAACLAB_PATH", "/workspace/isaaclab")
    monkeypatch.setenv("DOCKER_USER_HOME", "/root")

    targets = volume_mounts.resolved_targets(DOCKER_DIR / "docker-compose.yaml")

    assert "/isaac-sim/kit/cache" in targets
    assert all("${" not in target for target in targets)


def test_kitless_volume_key_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """The Dockerfile-facing CLI accepts the explicit kit-less volume key."""
    monkeypatch.setenv("DOCKER_ISAACLAB_PATH", "/workspace/isaaclab")
    monkeypatch.setenv("DOCKER_USER_HOME", "/home/isaaclab")
    monkeypatch.setattr(
        "sys.argv",
        ["volume_mounts.py", "--volumes_key", "x-kitless-isaac-lab-volumes"],
    )

    assert volume_mounts.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "/home/isaaclab/.cache/uv",
        "/home/isaaclab/.cache/warp",
        "/workspace/isaaclab/docs/_build",
        "/workspace/isaaclab/logs",
        "/workspace/isaaclab/data_storage",
    ]
