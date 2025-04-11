# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import unittest
from pathlib import Path


class TestDocker(unittest.TestCase):
    """Test starting and stopping of the docker container with both currently supported profiles and with and without
    a suffix.  This assumes that docker is installed and configured correctly so that the user can use the docker
    commands from the current shell."""

    def start_stop_docker(self, profile, suffix):
        """Test starting and stopping docker profile with suffix."""
        environ = os.environ
        context_dir = Path(__file__).resolve().parent.parent

        # generate parameters for the arguments
        if suffix != "":
            container_name = f"isaac-lab-{profile}-{suffix}"
            suffix_args = ["--suffix", suffix]
        else:
            container_name = f"isaac-lab-{profile}"
            suffix_args = []

        run_kwargs = {
            "check": False,
            "capture_output": True,
            "text": True,
            "cwd": context_dir,
            "env": environ,
        }

        # start the container
        docker_start = subprocess.run(["python", "container.py", "start", profile] + suffix_args, **run_kwargs)
        self.assertEqual(docker_start.returncode, 0)

        # verify that the container is running
        docker_running_true = subprocess.run(["docker", "ps"], **run_kwargs)
        self.assertEqual(docker_running_true.returncode, 0)
        self.assertIn(container_name, docker_running_true.stdout)

        # stop the container
        docker_stop = subprocess.run(["python", "container.py", "stop", profile] + suffix_args, **run_kwargs)
        self.assertEqual(docker_stop.returncode, 0)

        # verify that the container has stopped
        docker_running_false = subprocess.run(["docker", "ps"], **run_kwargs)
        self.assertEqual(docker_running_false.returncode, 0)
        self.assertNotIn(container_name, docker_running_false.stdout)

    def test_docker_base(self):
        """Test starting and stopping docker base."""
        self.start_stop_docker("base", "")

    def test_docker_base_suffix(self):
        """Test starting and stopping docker base with a test suffix."""
        self.start_stop_docker("base", "test")

    def test_docker_ros2(self):
        """Test starting and stopping docker ros2."""
        self.start_stop_docker("ros2", "")

    def test_docker_ros2_suffix(self):
        """Test starting and stopping docker ros2 with a test suffix."""
        self.start_stop_docker("ros2", "test")


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=True)
