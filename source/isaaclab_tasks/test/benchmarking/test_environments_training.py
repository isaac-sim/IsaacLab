# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training benchmark tests (subprocess train.py per job; no Kit in the pytest parent process)."""

import contextlib
import os
import subprocess
import sys
import threading
import time

import env_benchmark_test_utils as utils
import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401 — register Isaac-* tasks in gym registry

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_rl.utils.pretrained_checkpoint import WORKFLOW_EXPERIMENT_NAME_VARIABLE, WORKFLOW_TRAINER


def _run_subprocess_tee(cmd, *, cwd):
    """Run ``cmd`` forwarding stdout/stderr live while retaining full output for diagnostics."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    out_chunks: list[str] = []
    err_chunks: list[str] = []

    def _pump(stream, chunks, write_to):
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                write_to.write(line)
                write_to.flush()
        finally:
            stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, out_chunks, sys.stdout), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, err_chunks, sys.stderr), daemon=True)
    t_out.start()
    t_err.start()
    ret = proc.wait()
    t_out.join()
    t_err.join()
    return subprocess.CompletedProcess(cmd, ret, "".join(out_chunks), "".join(err_chunks))


def setup_environment():
    """Setup environment for testing."""
    # Acquire all Isaac environments names
    registered_task_specs = []
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id and not task_spec.id.endswith("Play-v0"):
            registered_task_specs.append(task_spec)

    # Sort environments by name
    registered_task_specs.sort(key=lambda x: x.id)

    return registered_task_specs


def train_job(
    workflow,
    task,
    env_config,
    num_gpus,
    *,
    video: bool = False,
    video_length: int = 200,
    video_interval: int = 2000,
):
    """Train a single job; return a dict with duration, returncode, stdout, and stderr."""
    cmd = [
        sys.executable,
        WORKFLOW_TRAINER[workflow],
        "--rl_library",
        workflow,
        "--task",
        task,
    ]

    if video:
        cmd.extend(["--video", "--video_length", str(video_length), "--video_interval", str(video_interval)])

    # Add max iterations if specified
    max_iterations = env_config.get("max_iterations")
    if max_iterations is not None:
        cmd.extend(["--max_iterations", str(max_iterations)])

    # Optional per-job parallel env count (omit flag to use each train.py / task default)
    num_envs = env_config.get("num_envs")
    if num_envs is not None:
        cmd.extend(["--num_envs", str(int(num_envs))])

    if num_gpus > 1:
        cmd.append(f"--nnprod_per_node={num_gpus}")
        cmd.append("--distributed")

    # Add experiment name variable
    workflow_experiment_name_variable = WORKFLOW_EXPERIMENT_NAME_VARIABLE.get(workflow)
    if workflow_experiment_name_variable:
        cmd.append(f"{workflow_experiment_name_variable}={task}")

    print("Running : " + " ".join(cmd))

    repo_root = utils._get_repo_path()
    start_time = time.time()
    # Stream child output live (``capture_output`` would hide logs until the run finishes).
    quiet = os.environ.get("ISAACLAB_BENCHMARK_TRAIN_QUIET", "").strip().lower() in ("1", "true", "yes")
    if quiet:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    else:
        result = _run_subprocess_tee(cmd, cwd=repo_root)
    duration = time.time() - start_time

    out = result.stdout or ""
    err = result.stderr or ""
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        if err.strip():
            print(f"STDERR (tail): {err[-6000:]}")
        if out.strip():
            print(f"STDOUT (tail): {out[-4000:]}")

    return {
        "duration": duration,
        "returncode": result.returncode,
        "stdout": out,
        "stderr": err,
    }


@pytest.mark.parametrize("task_spec", setup_environment())
def test_train_environments(
    workflow, task_spec, config_path, mode, num_gpus, video, video_length, video_interval, kpi_store
):
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

    train_result = train_job(
        workflow,
        task,
        env_config,
        num_gpus,
        video=video,
        video_length=video_length,
        video_interval=video_interval,
    )

    print(f">>> Evaluating trained: {job_name}")
    # Check if training logs were output and all thresholds passed
    kpi_payload = utils.evaluate_job(workflow, task, env_config, train_result)

    success_flag = kpi_payload["success"]
    print(f">>> Trained {job_name} success flag: {success_flag}.")
    print("-" * 80)

    # Save KPI
    kpi_store[job_name] = kpi_payload

    # Verify job was successful
    if not kpi_payload["success"]:
        pytest.fail(f"Job {job_name} failed to meet success criteria")
