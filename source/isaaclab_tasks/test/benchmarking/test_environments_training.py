# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# Launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import subprocess
import sys
import time

import carb
import env_benchmark_test_utils as utils
import pytest

from isaaclab.utils.pretrained_checkpoint import WORKFLOW_EXPERIMENT_NAME_VARIABLE, WORKFLOW_TRAINER


def setup_environment():
    """Setup environment for testing."""
    # Acquire all Isaac environments names
    registered_task_specs = []
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id and not task_spec.id.endswith("Play-v0"):
            registered_task_specs.append(task_spec)

    # Sort environments by name
    registered_task_specs.sort(key=lambda x: x.id)

    # This flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    return registered_task_specs


def train_job(workflow, task, env_config, num_gpus):
    """Train a single job for a given workflow, task, and configuration, and return the duration."""
    cmd = [
        sys.executable,
        WORKFLOW_TRAINER[workflow],
        "--task",
        task,
        "--enable_cameras",
        "--headless",
    ]

    # Add max iterations if specified
    max_iterations = env_config.get("max_iterations")
    if max_iterations is not None:
        cmd.extend(["--max_iterations", str(max_iterations)])

    if num_gpus > 1:
        cmd.append(f"--nnprod_per_node={num_gpus}")
        cmd.append("--distributed")

    # Add experiment name variable
    cmd.append(f"{WORKFLOW_EXPERIMENT_NAME_VARIABLE[workflow]}={task}")

    print("Running : " + " ".join(cmd))

    start_time = time.time()
    subprocess.run(cmd)
    duration = time.time() - start_time

    return duration


@pytest.mark.parametrize("task_spec", setup_environment())
def test_train_environments(workflow, task_spec, config_path, mode, num_gpus, kpi_store):
    """Train environments provided in the config file, save KPIs, and evaluate against thresholds"""
    # Skip if workflow not supported for this task
    if workflow + "_cfg_entry_point" not in task_spec.kwargs:
        pytest.skip(f"Workflow {workflow} not supported for task {task_spec.id}")

    # Load environment config
    task = task_spec.id
    if config_path.startswith("/"):
        full_config_path = config_path
    else:
        full_config_path = os.path.join(os.path.dirname(__file__), config_path)
    env_configs = utils.get_env_configs(full_config_path)
    env_config = utils.get_env_config(env_configs, mode, workflow, task)

    # Skip if config not found
    if env_config is None:
        pytest.skip(f"No config found for task {task} in {mode} mode")

    job_name = f"{workflow}:{task}"
    print(f">>> Training: {job_name}")

    # Train and capture duration
    duration = train_job(workflow, task, env_config, num_gpus)

    print(f">>> Evaluating trained: {job_name}")
    # Check if training logs were output and all thresholds passed
    kpi_payload = utils.evaluate_job(workflow, task, env_config, duration)

    success_flag = kpi_payload["success"]
    print(f">>> Trained {job_name} success flag: {success_flag}.")
    print("-" * 80)

    # Save KPI
    kpi_store[job_name] = kpi_payload

    # Verify job was successful
    if not kpi_payload["success"]:
        pytest.fail(f"Job {job_name} failed to meet success criteria")
