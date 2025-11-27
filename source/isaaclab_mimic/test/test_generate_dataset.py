# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import pytest

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0")
NUCLEUS_DATASET_PATH = os.path.join(ISAACLAB_NUCLEUS_DIR, "Tests", "Mimic", "dataset.hdf5")
EXPECTED_SUCCESSFUL_ANNOTATIONS = 10


@pytest.fixture
def setup_test_environment():
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
        pytest.fail(
            "The dataset required for this test is currently unavailable. Dataset path: " + NUCLEUS_DATASET_PATH
        )

    # Set the environment variable PYTHONUNBUFFERED to 1 to get all text outputs in result.stdout
    pythonunbuffered_env_var_ = os.environ.get("PYTHONUNBUFFERED")
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
        "--auto",
        "--headless",
    ]
    print(config_command)

    # Execute the command and capture the result
    result = subprocess.run(config_command, capture_output=True, text=True)

    print(f"Annotate demos result: {result.returncode}\n\n\n\n\n\n\n\n\n\n\n\n")

    # Print the result for debugging purposes
    print("Config generation result:")
    print(result.stdout)  # Print standard output from the command
    print(result.stderr)  # Print standard error from the command

    # Check if the config generation was successful
    assert result.returncode == 0, result.stderr

    # Check that at least one task was completed successfully by parsing stdout
    # Look for the line that reports successful task completions
    success_line = None
    for line in result.stdout.split("\n"):
        if "Successful task completions:" in line:
            success_line = line
            break

    assert success_line is not None, "Could not find 'Successful task completions:' in output"

    # Extract the number from the line
    try:
        successful_count = int(success_line.split(":")[-1].strip())
        assert (
            successful_count == EXPECTED_SUCCESSFUL_ANNOTATIONS
        ), f"Expected 10 successful annotations but got {successful_count}"
    except (ValueError, IndexError) as e:
        pytest.fail(f"Could not parse successful task count from line: '{success_line}'. Error: {e}")

    # Yield the workflow root for use in tests
    yield workflow_root

    # Cleanup: restore the original environment variable
    if pythonunbuffered_env_var_:
        os.environ["PYTHONUNBUFFERED"] = pythonunbuffered_env_var_
    else:
        del os.environ["PYTHONUNBUFFERED"]


@pytest.mark.isaacsim_ci
def test_generate_dataset(setup_test_environment):
    """Test the dataset generation script."""
    workflow_root = setup_test_environment

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
    assert result.returncode == 0, result.stderr

    # Check for specific output
    expected_output = "successes/attempts. Exiting"
    assert expected_output in result.stdout
