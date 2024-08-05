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

    def __init__(
        self,
        context_dir: Path,
        profile: str = "base",
        statefile: None | Statefile = None,
        yamls: list[str] | None = None,
        envs: list[str] | None = None,
    ):
        """
        Initialize the IsaacLabContainerInterface with the given parameters.

        Args:
            context_dir : The context directory for Docker operations.
            statefile : An instance of the Statefile class to manage state variables. If not provided, initializes a Statefile(path=self.context_dir/.container.yaml).
            profile : The profile name for the container. Defaults to "base".
            yamls : A list of yamls to extend docker-compose.yaml. They will be extended in the order they are provided.
            envs : A list of envs to extend .env.base. They will be extended in the order they are provided.
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
        self.resolve_image_extension(yamls, envs)
        self.load_dot_vars()

    def resolve_image_extension(self, yamls: list[str] | None = None, envs: list[str] | None = None):
        """
        Resolve the image extension by setting up YAML files, profiles, and environment files for the Docker compose command.

        Args:
            yamls (List[str], optional): A list of yamls to extend docker-compose.yaml. They will be extended in the order they are provided.
            envs (List[str], optional): A list of envs to extend .env.base. They will be extended in the order they are provided.
        """
        self.add_yamls = ["--file", "docker-compose.yaml"]
        self.add_profiles = ["--profile", f"{self.profile}"]
        self.add_env_files = ["--env-file", ".env.base"]
        if self.profile != "base":
            self.add_env_files += ["--env-file", f".env.{self.profile}"]

        if yamls is not None:
            for yaml in yamls:
                self.add_yamls += ["--file", yaml]

        if envs is not None:
            for env in envs:
                self.add_env_files += ["--env-file", env]

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
        print(f"[INFO] Building the docker image and starting the container {self.container_name} in the background...")
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
            print(f"[INFO] Entering the existing {self.container_name} container in a bash session...")
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
            print(f"[INFO] Stopping the launched docker container {self.container_name}...")
            subprocess.run(
                ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["down"],
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
            print(f"[INFO] Copying artifacts from the 'isaac-lab-{self.container_name}' container...")
            if output_dir is None:
                output_dir = self.context_dir
            output_dir = output_dir.joinpath("artifacts")
            if not output_dir.is_dir():
                output_dir.mkdir()
            artifacts = {
                Path(self.dot_vars["DOCKER_ISAACLAB_PATH"]).joinpath("logs"): output_dir.joinpath("logs"),
                Path(self.dot_vars["DOCKER_ISAACLAB_PATH"]).joinpath("docs/_build"): output_dir.joinpath("docs"),
                Path(self.dot_vars["DOCKER_ISAACLAB_PATH"]).joinpath("data_storage"): output_dir.joinpath(
                    "data_storage"
                ),
            }
            for container_path, host_path in artifacts.items():
                print(f"\t -{container_path} -> {host_path}")
            for path in artifacts.values():
                shutil.rmtree(path, ignore_errors=True)
            for container_path, host_path in artifacts.items():
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"isaac-lab-{self.profile}:{container_path}/",
                        f"{host_path}",
                    ],
                    check=False,
                )
            print("\n[INFO] Finished copying the artifacts from the container.")
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running")

    def config(self, output_yaml: Path | None = None) -> None:
        """
        Generate a docker-compose.yaml from the passed yamls, .envs, and either print to the
        terminal or create a yaml at output_yaml

        Args:
            output_yaml (Path, optional): The absolute path of the yaml file to write the output to, if any. Defaults
            to None, and simply prints to the terminal
        """
        print("[INFO] Configuring the passed options into a yaml...")
        if output_yaml is not None:
            output = ["--output", output_yaml]
        else:
            output = []
        subprocess.run(
            ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["config"] + output,
            check=False,
            cwd=self.context_dir,
            env=self.environ,
        )
