# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test dataset generation for Isaac Lab Mimic workflow."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os
import subprocess
import tempfile
import unittest

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0")
NUCLEUS_DATASET_PATH = os.path.join(ISAACLAB_NUCLEUS_DIR, "Tests", "Mimic", "dataset.hdf5")


class TestGenerateDataset(unittest.TestCase):
    """Test the dataset generation behavior of the Isaac Lab Mimic workflow."""

    def setUp(self):
        """Set up the environment for testing."""
        # Create the datasets directory if it does not exist
        if not os.path.exists(DATASETS_DOWNLOAD_DIR):
            print("Creating directory : ", DATASETS_DOWNLOAD_DIR)
            os.makedirs(DATASETS_DOWNLOAD_DIR)
        # Try to download the dataset from Nucleus
        try:
            retrieve_file_path(NUCLEUS_DATASET_PATH, DATASETS_DOWNLOAD_DIR)
        except Exception as e:
            print(e)
            print("Could not download dataset from Nucleus")
            self.fail(
                "The dataset required for this test is currently unavailable. Dataset path: " + NUCLEUS_DATASET_PATH
            )

        # Set the environment variable PYTHONUNBUFFERED to 1 to get all text outputs in result.stdout
        self.pythonunbuffered_env_var_ = os.environ.get("PYTHONUNBUFFERED")
        os.environ["PYTHONUNBUFFERED"] = "1"

        # Automatically detect the workflow root (backtrack from current file location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_root = os.path.abspath(os.path.join(current_dir, "../../.."))

        # Run the command to generate core configs
        config_command = [
            workflow_root + "/isaaclab.sh",
            "-p",
            os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"),
            "--task",
            "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
            "--input_file",
            DATASETS_DOWNLOAD_DIR + "/dataset.hdf5",
            "--output_file",
            DATASETS_DOWNLOAD_DIR + "/annotated_dataset.hdf5",
            "--signals",
            "grasp_1",
            "stack_1",
            "grasp_2",
            "--auto",
            "--headless",
        ]
        print(config_command)

        # Execute the command and capture the result
        result = subprocess.run(config_command, capture_output=True, text=True)

        # Print the result for debugging purposes
        print("Config generation result:")
        print(result.stdout)  # Print standard output from the command
        print(result.stderr)  # Print standard error from the command

        # Check if the config generation was successful
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def tearDown(self):
        """Clean up after tests."""
        if self.pythonunbuffered_env_var_:
            os.environ["PYTHONUNBUFFERED"] = self.pythonunbuffered_env_var_
        else:
            del os.environ["PYTHONUNBUFFERED"]

    def test_generate_dataset(self):
        """Test the dataset generation script."""
        # Automatically detect the workflow root (backtrack from current file location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_root = os.path.abspath(os.path.join(current_dir, "../../.."))

        # Define the command to run the dataset generation script
        command = [
            workflow_root + "/isaaclab.sh",
            "-p",
            os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
            "--input_file",
            DATASETS_DOWNLOAD_DIR + "/annotated_dataset.hdf5",
            "--output_file",
            DATASETS_DOWNLOAD_DIR + "/generated_dataset.hdf5",
            "--generation_num_trials",
            "1",
            "--headless",
        ]

        # Call the script and capture output
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the result for debugging purposes
        print("Dataset generation result:")
        print(result.stdout)  # Print standard output from the command
        print(result.stderr)  # Print standard error from the command

        # Check if the script executed successfully
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        # Check for specific output
        expected_output = "successes/attempts. Exiting"
        self.assertIn(expected_output, result.stdout)


if __name__ == "__main__":
    unittest.main()
