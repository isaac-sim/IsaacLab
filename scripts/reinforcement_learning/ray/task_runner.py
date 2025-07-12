# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import yaml

import ray
import util

"""
This script dispatches one or more user-defined Python tasks to workers in a Ray cluster.
Each task, with its resource requirements and execution parameters, is described in a YAML configuration file.
You may specify the desired number of CPUs, GPUs, and memory allocation for each task in the config file.

Key features:
- Flexible resource management per task via config fields (`num_gpus`, `num_cpus`, `memory`).
- Real-time output streaming (stdout/stderr) for each task.
- Parallel execution of multiple tasks across cluster resources.

Tasks are distributed and scheduled according to Ray’s built-in resource manager.

Typical usage:
---------------

.. code-block:: bash

    # Print help and argument details:
    python task_runner.py -h

    # Submit tasks defined in a YAML file to the Ray cluster (auto-detects Ray head address):
    python task_runner.py --task_cfg /path/to/tasks.yaml

YAML configuration example:
---------------------------
.. code-block:: yaml
    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    tasks:
      - name: "task1"
        py_args: "-m torch.distributed.run --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=localhost:29501 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --max_iterations 200 --headless --distributed"
        num_gpus: 2
        num_cpus: 10
        memory: 10737418240
      - name: "task2"
        py_args: "script.py --option arg"
        num_gpus: 0
        num_cpus: 1
        memory: 10*1024*1024*1024

- `pip`: List of pip packages to install.
- `py_args`: Arguments passed to the Python executable for this task.
- `num_gpus`, `num_cpus`: Number of GPUs/CPUs to allocate. Can be integer or a string like `"2*2"`.
- `memory`: Amount of memory (bytes) to allocate. Can be integer or a string like `"10*1024*1024*1024"`.

To stop all tasks early, press Ctrl+C; the script will cancel all running Ray tasks.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run tasks from a YAML config file.")
    parser.add_argument("--task_cfg", type=str, required=True, help="Path to the YAML task file.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run nvidia-smi test instead of the arbitrary job,"
            "can use as a sanity check prior to any jobs to check "
            "that GPU resources are correctly isolated."
        ),
    )
    return parser.parse_args()


def parse_task_opt(task):
    opts = {}
    if "num_gpus" in task:
        opts["num_gpus"] = eval(task["num_gpus"]) if isinstance(task["num_gpus"], str) else task["num_gpus"]
    if "num_cpus" in task:
        opts["num_cpus"] = eval(task["num_cpus"]) if isinstance(task["num_cpus"], str) else task["num_cpus"]
    if "memory" in task:
        opts["memory"] = eval(task["memory"]) if isinstance(task["memory"], str) else task["memory"]
    return opts


@ray.remote
def remote_execute_job(job_cmd: str, identifier_string: str, test_mode: bool) -> str | dict:
    return util.execute_job(
        job_cmd=job_cmd,
        identifier_string=identifier_string,
        test_mode=test_mode,
        log_all_output=True,  # make log_all_output=True to check output in real time
    )


def run_tasks(ray_address, pip, py_modules, tasks, test_mode=False):
    if not tasks:
        print("[WARNING]: no tasks to submit")
        return

    if not ray.is_initialized():
        try:
            ray.init(
                address=ray_address,
                log_to_driver=True,
                runtime_env={
                    "pip": pip,
                    "py_modules": py_modules,
                },
            )
        except Exception as e:
            raise RuntimeError(f"initialize ray failed: {str(e)}")
    task_results = []
    for task in tasks:
        opts = parse_task_opt(task)
        task_cmd = " ".join([sys.executable, *task["py_args"].split()])
        print(f"[INFO] submitting task {task['name']} with opts={opts}: {task_cmd}")
        task_results.append(remote_execute_job.options(**opts).remote(task_cmd, task["name"], test_mode))

    try:
        results = ray.get(task_results)
        for i, result in enumerate(results):
            print(f"[INFO]: Task {tasks[i]['name']} result: \n{result}")
        print("[INFO]: all tasks completed.")
    except KeyboardInterrupt:
        print("[INFO]: dealing with keyboard interrupt")
        for future in task_results:
            ray.cancel(future, force=True)
        print("[INFO]: all tasks cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR]: error while running tasks: {str(e)}")
        raise e


def main():
    args = parse_args()
    try:
        with open(args.task_cfg) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise SystemExit(f"error while loading task config: {str(e)}")
    tasks = config["tasks"]
    py_modules = config.get("py_modules")
    pip = config.get("pip")
    run_tasks(
        ray_address=args.ray_address,
        pip=pip,
        py_modules=py_modules,
        tasks=tasks,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
