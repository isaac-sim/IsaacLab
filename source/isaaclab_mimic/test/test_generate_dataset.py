# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test dataset generation for Isaac Lab Mimic workflow."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os
import signal
import subprocess
import sys
import tempfile

import pytest

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0")
NUCLEUS_DATASET_PATH = os.path.join(ISAACLAB_NUCLEUS_DIR, "Tests", "Mimic", "dataset.hdf5")
EXPECTED_SUCCESSFUL_ANNOTATIONS = 10

# Timeout for subprocess execution (seconds).  The annotation / generation
# scripts run a full simulation loop which can take several minutes.  A second,
# shorter grace period is given after the timeout to allow cleanup before the
# process is forcefully killed.
_SUBPROCESS_TIMEOUT = 600
_SUBPROCESS_GRACE_PERIOD = 15


def _run_script(command: list[str]) -> subprocess.CompletedProcess:
    """Run a script in a subprocess and return a CompletedProcess.

    The Kit / Omniverse runtime's ``simulation_app.close()`` can hang
    indefinitely when another ``SimulationApp`` instance is alive in the parent
    test process (shared GPU / IPC resources).  To avoid blocking the test
    suite we use ``Popen`` with an explicit timeout:

    1. Wait up to ``_SUBPROCESS_TIMEOUT`` seconds for the process to finish.
    2. On timeout send ``SIGTERM`` and wait ``_SUBPROCESS_GRACE_PERIOD`` seconds.
    3. If still alive, ``SIGKILL`` and collect remaining output.

    The captured *stdout* / *stderr* are returned regardless of how the process
    terminated so that callers can validate the script's printed output.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired:
        # Script likely hung during simulation_app.close() – ask nicely first.
        process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=_SUBPROCESS_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )


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

    result = _run_script(config_command)

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


@pytest.mark.isaacsim_ci
def test_generate_dataset(setup_test_environment):
    """Test the dataset generation script."""
    workflow_root = setup_test_environment

    annotated_input_path = os.path.join(DATASETS_DOWNLOAD_DIR, "annotated_dataset.hdf5")
    generated_output_path = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset.hdf5")

    # Define the command to run the dataset generation script directly
    # (bypassing isaaclab.sh — see fixture comments for rationale).
    command = [
        sys.executable,
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "--input_file",
        annotated_input_path,
        "--output_file",
        generated_output_path,
        "--generation_num_trials",
        "1",
        "--headless",
    ]

    result = _run_script(command)

    # Print the result for debugging purposes
    print("Dataset generation result:")
    print(result.stdout)  # Print standard output from the command
    print(result.stderr)  # Print standard error from the command

    # Verify the generated dataset file was created.
    # Note: we cannot rely solely on the exit code because the Kit runtime may
    # reset it to 0 during cleanup, so we check the output file and stdout.
    assert os.path.exists(generated_output_path), (
        f"Generated dataset file was not created at {generated_output_path}.\n"
        f"returncode: {result.returncode}\nstderr: {result.stderr}"
    )

    # Check for the expected completion message in output
    combined_output = result.stdout + "\n" + result.stderr
    expected_output = "successes/attempts. Exiting"
    assert expected_output in combined_output, (
        f"Could not find '{expected_output}' in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
