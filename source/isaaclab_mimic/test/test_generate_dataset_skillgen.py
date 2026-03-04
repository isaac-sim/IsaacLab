# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test dataset generation with SkillGen for Isaac Lab Mimic workflow."""

from isaaclab.app import AppLauncher

# Launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os
import signal
import subprocess
import sys
import tempfile

import pytest

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0")
NUCLEUS_SKILLGEN_ANNOTATED_DATASET_PATH = os.path.join(
    ISAACLAB_NUCLEUS_DIR, "Mimic", "franka_stack_datasets", "annotated_dataset_skillgen.hdf5"
)

# Timeout for subprocess execution (seconds).
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
def setup_skillgen_test_environment():
    """Prepare environment for SkillGen dataset generation test."""
    # Create the datasets directory if it does not exist
    if not os.path.exists(DATASETS_DOWNLOAD_DIR):
        print("Creating directory : ", DATASETS_DOWNLOAD_DIR)
        os.makedirs(DATASETS_DOWNLOAD_DIR)

    # Download the SkillGen annotated dataset from Nucleus into DATASETS_DOWNLOAD_DIR.
    # retrieve_file_path mirrors the remote directory tree under the download
    # dir and returns the actual local path of the downloaded file.
    downloaded_dataset_path = retrieve_file_path(NUCLEUS_SKILLGEN_ANNOTATED_DATASET_PATH, DATASETS_DOWNLOAD_DIR)

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

    # Yield the workflow root and the actual downloaded file path for use in tests
    yield workflow_root, downloaded_dataset_path

    # Cleanup: restore the original environment variable
    if pythonunbuffered_env_var_:
        os.environ["PYTHONUNBUFFERED"] = pythonunbuffered_env_var_
    else:
        del os.environ["PYTHONUNBUFFERED"]


def test_generate_dataset_skillgen(setup_skillgen_test_environment):
    """Test dataset generation with SkillGen enabled."""
    workflow_root, input_file = setup_skillgen_test_environment

    output_file = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset_skillgen.hdf5")

    # Run the script directly (bypassing isaaclab.sh) so that stdout is
    # properly captured (see _run_script docstring for details).
    command = [
        sys.executable,
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "--device",
        "cpu",
        "--input_file",
        input_file,
        "--output_file",
        output_file,
        "--num_envs",
        "1",
        "--generation_num_trials",
        "1",
        "--use_skillgen",
        "--headless",
        "--task",
        "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    ]

    result = _run_script(command)

    print("SkillGen dataset generation result:")
    print(result.stdout)
    print(result.stderr)

    # Verify the generated dataset file was created.
    assert os.path.exists(output_file), (
        f"Generated SkillGen dataset file was not created at {output_file}.\n"
        f"returncode: {result.returncode}\nstderr: {result.stderr}"
    )

    # Check for the expected completion message in output
    combined_output = result.stdout + "\n" + result.stderr
    expected_output = "successes/attempts. Exiting"
    assert expected_output in combined_output, (
        f"Could not find '{expected_output}' in output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
