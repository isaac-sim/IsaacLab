# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test dataset generation for Isaac Lab Mimic workflow."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os
import sys
import tempfile

import pytest
from mimic_test_utils import run_script

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0")
NUCLEUS_DATASET_PATH = os.path.join(ISAACLAB_NUCLEUS_DIR, "Tests", "Mimic", "dataset.hdf5")
EXPECTED_SUCCESSFUL_ANNOTATIONS = 10

_SUBPROCESS_TIMEOUT = 600


@pytest.fixture
def setup_test_environment():
    """Set up the environment for testing."""
    # Create the datasets directory if it does not exist
    if not os.path.exists(DATASETS_DOWNLOAD_DIR):
        print("Creating directory : ", DATASETS_DOWNLOAD_DIR)
        os.makedirs(DATASETS_DOWNLOAD_DIR)

    # Try to download the dataset from Nucleus.
    # retrieve_file_path mirrors the remote directory tree under the download
    # dir and returns the actual local path of the downloaded file.
    try:
        downloaded_dataset_path = retrieve_file_path(NUCLEUS_DATASET_PATH, DATASETS_DOWNLOAD_DIR)
    except Exception as e:
        print(e)
        print("Could not download dataset from Nucleus")
        pytest.fail(
            "The dataset required for this test is currently unavailable. Dataset path: " + NUCLEUS_DATASET_PATH
        )

    # Verify the downloaded file actually exists on disk
    assert os.path.isfile(downloaded_dataset_path), (
        f"retrieve_file_path returned '{downloaded_dataset_path}' but the file does not exist on disk."
    )

    # Set the environment variable PYTHONUNBUFFERED to 1 to get all text outputs in result.stdout
    pythonunbuffered_env_var_ = os.environ.get("PYTHONUNBUFFERED")
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Automatically detect the workflow root (backtrack from current file location)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_root = os.path.abspath(os.path.join(current_dir, "../../.."))

    annotated_output_path = os.path.join(DATASETS_DOWNLOAD_DIR, "annotated_dataset.hdf5")

    # Run the annotate_demos script directly (bypassing isaaclab.sh) so that
    # stdout is properly captured.  When launched through the CLI wrapper the
    # Omniverse/Kit runtime redirects OS-level file descriptors during
    # SimulationApp init, swallowing all print() output.
    config_command = [
        sys.executable,
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"),
        "--task",
        "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
        "--input_file",
        downloaded_dataset_path,
        "--output_file",
        annotated_output_path,
        "--auto",
        "--headless",
    ]
    print(config_command)

    result = run_script(config_command, timeout=_SUBPROCESS_TIMEOUT)

    print(f"Annotate demos result: {result.returncode}\n")

    # Print the result for debugging purposes
    print("Config generation result:")
    print(result.stdout)  # Print standard output from the command
    print(result.stderr)  # Print standard error from the command

    # Check that at least one task was completed successfully by parsing stdout.
    # Note: we cannot rely on the process exit code because simulation_app.close()
    # triggers Kit runtime cleanup that resets the exit code to 0 (or the process
    # may have been killed after a cleanup hang, yielding -SIGKILL).
    combined_output = result.stdout + "\n" + result.stderr
    success_line = None
    for line in combined_output.split("\n"):
        if "Successful task completions:" in line:
            success_line = line
            break

    assert success_line is not None, (
        f"Could not find 'Successful task completions:' in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Extract the number from the line
    try:
        successful_count = int(success_line.split(":")[-1].strip())
    except (ValueError, IndexError) as e:
        pytest.fail(f"Could not parse successful task count from line: '{success_line}'. Error: {e}")

    assert successful_count == EXPECTED_SUCCESSFUL_ANNOTATIONS, (
        f"Expected {EXPECTED_SUCCESSFUL_ANNOTATIONS} successful annotations but got {successful_count}"
    )

    # Also verify the annotated output file was created
    assert os.path.exists(annotated_output_path), f"Annotated dataset file was not created at {annotated_output_path}"

    # Yield the workflow root for use in tests
    yield workflow_root

    # Cleanup: restore the original environment variable
    if pythonunbuffered_env_var_:
        os.environ["PYTHONUNBUFFERED"] = pythonunbuffered_env_var_
    else:
        del os.environ["PYTHONUNBUFFERED"]


def _run_generation(workflow_root: str, input_file: str, output_file: str, num_envs: int):
    """Build the generation command, run it, and assert success."""
    command = [
        sys.executable,
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "--input_file",
        input_file,
        "--output_file",
        output_file,
        "--num_envs",
        str(num_envs),
        "--generation_num_trials",
        "1",
        "--headless",
    ]

    result = run_script(command, timeout=_SUBPROCESS_TIMEOUT)

    print(f"State-based dataset generation result (num_envs={num_envs}):")
    print(result.stdout)
    print(result.stderr)

    assert os.path.exists(output_file), (
        f"Generated dataset file was not created at {output_file}.\n"
        f"returncode: {result.returncode}\nstderr: {result.stderr}"
    )

    combined_output = result.stdout + "\n" + result.stderr
    expected_output = "successes/attempts. Exiting"
    assert expected_output in combined_output, (
        f"Could not find '{expected_output}' in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.isaacsim_ci
def test_generate_dataset_franka_state(setup_test_environment):
    """Test dataset generation for the state-based cube-stack environment (single env)."""
    workflow_root = setup_test_environment
    annotated_input_path = os.path.join(DATASETS_DOWNLOAD_DIR, "annotated_dataset.hdf5")
    generated_output_path = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset.hdf5")
    _run_generation(workflow_root, annotated_input_path, generated_output_path, num_envs=1)


@pytest.mark.isaacsim_ci
def test_generate_dataset_franka_state_multi_env(setup_test_environment):
    """Test dataset generation for the state-based cube-stack environment (5 envs)."""
    workflow_root = setup_test_environment
    annotated_input_path = os.path.join(DATASETS_DOWNLOAD_DIR, "annotated_dataset.hdf5")
    generated_output_path = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset_multi_env.hdf5")
    _run_generation(workflow_root, annotated_input_path, generated_output_path, num_envs=5)
