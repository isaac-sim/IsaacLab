# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test dataset generation for Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os
import sys
import tempfile

import pytest
from mimic_test_utils import run_script

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

_TASK_NAME = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0"
DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix=f"_{_TASK_NAME}")
NUCLEUS_ANNOTATED_DATASET_PATH = os.path.join(
    ISAACLAB_NUCLEUS_DIR, "Mimic", "Tests", "annotated_dataset_franka_visuomotor_test.hdf5"
)


@pytest.fixture
def setup_visuomotor_test_environment():
    """Download the pre-annotated dataset and prepare the test environment."""
    if not os.path.exists(DATASETS_DOWNLOAD_DIR):
        os.makedirs(DATASETS_DOWNLOAD_DIR)

    try:
        downloaded_dataset_path = retrieve_file_path(NUCLEUS_ANNOTATED_DATASET_PATH, DATASETS_DOWNLOAD_DIR)
    except Exception as e:
        print(e)
        pytest.fail(
            "The pre-annotated dataset required for this test is currently unavailable. "
            f"Dataset path: {NUCLEUS_ANNOTATED_DATASET_PATH}"
        )

    assert os.path.isfile(downloaded_dataset_path), (
        f"retrieve_file_path returned '{downloaded_dataset_path}' but the file does not exist on disk."
    )

    pythonunbuffered_env_var_ = os.environ.get("PYTHONUNBUFFERED")
    os.environ["PYTHONUNBUFFERED"] = "1"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_root = os.path.abspath(os.path.join(current_dir, "../../.."))

    yield workflow_root, downloaded_dataset_path

    if pythonunbuffered_env_var_:
        os.environ["PYTHONUNBUFFERED"] = pythonunbuffered_env_var_
    else:
        del os.environ["PYTHONUNBUFFERED"]


def _run_generation(workflow_root: str, input_file: str, output_file: str, num_envs: int):
    """Build the generation command, run it, and assert success."""
    command = [
        sys.executable,
        os.path.join(workflow_root, "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "--task",
        _TASK_NAME,
        "--input_file",
        input_file,
        "--output_file",
        output_file,
        "--num_envs",
        str(num_envs),
        "--generation_num_trials",
        "1",
        "--enable_cameras",
        "--headless",
    ]

    result = run_script(command)

    print(f"Visuomotor dataset generation result (num_envs={num_envs}):")
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


def test_generate_dataset_franka_visuomotor(setup_visuomotor_test_environment):
    """Test dataset generation for the visuomotor cube-stack environment (single env)."""
    workflow_root, input_file = setup_visuomotor_test_environment
    output_file = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset.hdf5")
    _run_generation(workflow_root, input_file, output_file, num_envs=1)


def test_generate_dataset_franka_visuomotor_multi_env(setup_visuomotor_test_environment):
    """Test dataset generation for the visuomotor cube-stack environment (5 envs)."""
    workflow_root, input_file = setup_visuomotor_test_environment
    output_file = os.path.join(DATASETS_DOWNLOAD_DIR, "generated_dataset_multi_env.hdf5")
    _run_generation(workflow_root, input_file, output_file, num_envs=5)
