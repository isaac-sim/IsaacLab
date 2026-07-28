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
    """Only profiles derived from the Isaac Sim base image require it to be built first."""
    assert not make_interface("base").requires_base_image
    assert make_interface("ros2").requires_base_image
    assert not make_interface("kitless").requires_base_image


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
    """Enter and stop use the kit-less service name, environment, and display."""
    interface = make_interface("kitless")
    run = MagicMock()
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("docker.utils.container_interface.subprocess.run", run)
    monkeypatch.setattr(interface, "is_container_running", MagicMock(return_value=True))

    interface.enter()
    interface.stop()

    assert [call.args[0] for call in run.call_args_list] == [
        ["docker", "exec", "--interactive", "--tty", "-e", "DISPLAY=:99", "isaac-lab-kitless", "bash"],
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


def test_x11_overlay_covers_every_profile():
    """Compose merges the X11 override by service name, so each profile needs an entry."""
    overlay = yaml.safe_load((DOCKER_DIR / "x11.yaml").read_text(encoding="utf-8"))

    assert set(overlay["services"]) == {"isaac-lab-base", "isaac-lab-ros2", "isaac-lab-kitless"}
    for name, service in overlay["services"].items():
        assert "DISPLAY" in service["environment"], name
        assert any("X11-unix" in mount["source"] for mount in service["volumes"]), name


def test_cli_merges_x11_overlay_before_start(monkeypatch: pytest.MonkeyPatch):
    """The CLI merges the X11 overlay before starting a container."""
    interface = MagicMock()
    interface.profile = "base"
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
