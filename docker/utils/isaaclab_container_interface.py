# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from utils.statefile import Statefile


class IsaacLabContainerInterface:
    """
    Interface for managing Isaac Lab containers.
    """

    def __init__(self, context_dir: Path, profile: str = "base", statefile: None | Statefile = None):
        """
        Initialize the IsaacLabContainerInterface with the given parameters.

        Args:
            context_dir : The context directory for Docker operations.
            statefile : An instance of the Statefile class to manage state variables. If not provided, initializes a Statefile(path=self.context_dir/.container.yaml).
            profile : The profile name for the container. Defaults to "base".
        """
        self.context_dir = context_dir
        if statefile is None:
            self.statefile = Statefile(path=self.context_dir / ".container.cfg")
        else:
            self.statefile = statefile
        self.profile = profile
        if self.profile == "isaaclab":
            # Silently correct from isaaclab to base,
            # because isaaclab is a commonly passed arg
            # but not a real profile
            self.profile = "base"
        self.container_name = f"isaac-lab-{self.profile}"
        self.image_name = f"isaac-lab-{self.profile}:latest"
        self.environ = os.environ
        self.resolve_image_extension()
        self.load_dot_vars()

    def resolve_image_extension(self):
        """
        Resolve the image extension by setting up YAML files, profiles, and environment files for the Docker compose command.
        """
        self.add_yamls = ["--file", "docker-compose.yaml"]
        self.add_profiles = ["--profile", f"{self.profile}"]
        self.add_env_files = ["--env-file", ".env.base"]
        if self.profile != "base":
            self.add_env_files += ["--env-file", f".env.{self.profile}"]

    def load_dot_vars(self):
        """
        Load environment variables from .env files into a dictionary.

        The environment variables are read in order and overwritten if there are name conflicts,
        mimicking the behavior of Docker compose.
        """
        self.dot_vars: dict[str, Any] = {}
        if len(self.add_env_files) % 2 != 0:
            raise RuntimeError(
                "The parameter self.add_env_files is configured incorrectly. There should be an even number of"
                " arguments."
            )
        for i in range(1, len(self.add_env_files), 2):
            with open(self.context_dir / self.add_env_files[i]) as f:
                self.dot_vars.update(dict(line.strip().split("=", 1) for line in f if "=" in line))

    def is_container_running(self) -> bool:
        """
        Check if the container is running.

        If the container is not running, return False.

        Returns:
            bool: True if the container is running, False otherwise.
        """
        status = subprocess.run(
            ["docker", "container", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return status == "running"

    def does_image_exist(self) -> bool:
        """
        Check if the Docker image exists.

        If the image does not exist, return False.

        Returns:
            bool: True if the image exists, False otherwise.
        """
        result = subprocess.run(["docker", "image", "inspect", self.image_name], capture_output=True, text=True)
        return result.returncode == 0

    def start(self):
        """
        Build and start the Docker container using the Docker compose command.
        """
        subprocess.run(
            [
                "docker",
                "compose",
                "--file",
                "docker-compose.yaml",
                "--env-file",
                ".env.base",
                "build",
                "isaac-lab-base",
            ],
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )
        subprocess.run(
            ["docker", "compose"]
            + self.add_yamls
            + self.add_profiles
            + self.add_env_files
            + ["up", "--detach", "--build", "--remove-orphans"],
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )

    def enter(self):
        """
        Enter the running container by executing a bash shell.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            subprocess.run([
                "docker",
                "exec",
                "--interactive",
                "--tty",
                "-e",
                f"DISPLAY={os.environ['DISPLAY']}",
                f"{self.container_name}",
                "bash",
            ])
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running")

    def stop(self):
        """
        Stop the running container using the Docker compose command.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            subprocess.run(
                ["docker", "compose", "--file", "docker-compose.yaml"]
                + self.add_profiles
                + self.add_env_files
                + ["down"],
                check=False,
                cwd=self.context_dir,
                env=self.environ,
            )
        else:
            raise RuntimeError(f"Can't stop container '{self.container_name}' as it is not running.")

    def copy(self, output_dir: Path | None = None):
        """
        Copy artifacts from the running container to the host machine.

        Args:
            output_dir: The directory to copy the artifacts to. Defaults to self.context_dir.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            if output_dir is None:
                output_dir = self.context_dir
            output_dir = output_dir.joinpath("artifacts")
            if not output_dir.exists():
                output_dir.mkdir()
            artifacts = {
                "logs": output_dir.joinpath("logs"),
                "docs/_build": output_dir.joinpath("docs"),
                "data_storage": output_dir.joinpath("data_storage"),
            }
            for container_path, host_path in artifacts.items():
                print(f"\t - /workspace/isaaclab/{container_path} -> {host_path}")
            for path in artifacts.values():
                shutil.rmtree(path, ignore_errors=True)
            for container_path, host_path in artifacts.items():
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"isaac-lab-{self.profile}:/workspace/isaaclab/{container_path}/",
                        f"{host_path}",
                    ],
                    check=False,
                )
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running")
