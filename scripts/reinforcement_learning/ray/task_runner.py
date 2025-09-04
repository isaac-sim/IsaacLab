# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script dispatches one or more user-defined Python tasks to workers in a Ray cluster.
Each task, along with its resource requirements and execution parameters, is specified in a YAML configuration file.
Users may define the number of CPUs, GPUs, and the amount of memory to allocate per task via the config file.

Key features:
-------------
- Fine-grained, per-task resource management via config fields (`num_gpus`, `num_cpus`, `memory`).
- Parallel execution of multiple tasks using available resources across the Ray cluster.
- Option to specify node affinity for tasks, e.g., by hostname, node ID, or any node.
- Optional batch (simultaneous) or independent scheduling of tasks.

Task scheduling and distribution are handled via Ray’s built-in resource manager.

YAML configuration fields:
--------------------------
- `pip`: List of extra pip packages to install before running any tasks.
- `py_modules`: List of additional Python module paths (directories or files) to include in the runtime environment.
- `concurrent`: (bool) It determines task dispatch semantics:
    - If `concurrent: true`, **all tasks are scheduled as a batch**. The script waits until sufficient resources are available for every task in the batch, then launches all tasks together. If resources are insufficient, all tasks remain blocked until the cluster can support the full batch.
    - If `concurrent: false`, tasks are launched as soon as resources are available for each individual task, and Ray independently schedules them. This may result in non-simultaneous task start times.
- `tasks`: List of task specifications, each with:
    - `name`: String identifier for the task.
    - `py_args`: Arguments to the Python interpreter (e.g., script/module, flags, user arguments).
    - `num_gpus`: Number of GPUs to allocate (float or string arithmetic, e.g., "2*2").
    - `num_cpus`: Number of CPUs to allocate (float or string).
    - `memory`: Amount of RAM in bytes (int or string).
    - `node` (optional): Node placement constraints.
        - `specific` (str): Type of node placement, support `hostname`, `node_id`, or `any`.
            - `any`: Place the task on any available node.
            - `hostname`: Place the task on a specific hostname. `hostname` must be specified in the node field.
            - `node_id`: Place the task on a specific node ID. `node_id` must be specified in the node field.
        - `hostname` (str): Specific hostname to place the task on.
        - `node_id` (str): Specific node ID to place the task on.


Typical usage:
---------------

.. code-block:: bash

    # Print help and argument details:
    python task_runner.py -h

    # Submit tasks defined in a YAML file to the Ray cluster (auto-detects Ray head address):
    python task_runner.py --task_cfg /path/to/tasks.yaml

YAML configuration example-1:
---------------------------
.. code-block:: yaml

    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    concurrent: false
    tasks:
      - name: "Isaac-Cartpole-v0"
        py_args: "-m torch.distributed.run --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=localhost:29501 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --max_iterations 200 --headless --distributed"
        num_gpus: 2
        num_cpus: 10
        memory: 10737418240
      - name: "script need some dependencies"
        py_args: "script.py --option arg"
        num_gpus: 0
        num_cpus: 1
        memory: 10*1024*1024*1024

YAML configuration example-2:
---------------------------
.. code-block:: yaml

    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    concurrent: true
    tasks:
    - name: "Isaac-Cartpole-v0-multi-node-train-1"
        py_args: "-m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --max_iterations 1000"
        num_gpus: 1
        num_cpus: 10
        memory: 10*1024*1024*1024
        node:
          specific: "hostname"
          hostname: "xxx"
    - name: "Isaac-Cartpole-v0-multi-node-train-2"
        py_args: "-m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=x.x.x.x:5555 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --max_iterations 1000"
        num_gpus: 1
        num_cpus: 10
        memory: 10*1024*1024*1024
        node:
          specific: "hostname"
          hostname: "xxx"

To stop all tasks early, press Ctrl+C; the script will cancel all running Ray tasks.
"""

import argparse
import yaml
from datetime import datetime

import util


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Ray task runner.

    Returns:
        argparse.Namespace: The namespace containing parsed CLI arguments:
            - task_cfg (str): Path to the YAML task file.
            - ray_address (str): Ray cluster address.
            - test (bool): Whether to run a GPU resource isolation sanity check.
    """
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


def parse_task_resource(task: dict) -> util.JobResource:
    """
    Parse task resource requirements from the YAML configuration.

    Args:
        task (dict): Dictionary representing a single task's configuration.
            Keys may include `num_gpus`, `num_cpus`, and `memory`, each either
            as a number or evaluatable string expression.

    Returns:
        util.JobResource: Resource object with the parsed values.
    """
    resource = util.JobResource()
    if "num_gpus" in task:
        resource.num_gpus = eval(task["num_gpus"]) if isinstance(task["num_gpus"], str) else task["num_gpus"]
    if "num_cpus" in task:
        resource.num_cpus = eval(task["num_cpus"]) if isinstance(task["num_cpus"], str) else task["num_cpus"]
    if "memory" in task:
        resource.memory = eval(task["memory"]) if isinstance(task["memory"], str) else task["memory"]
    return resource


def run_tasks(
    tasks: list[dict], args: argparse.Namespace, runtime_env: dict | None = None, concurrent: bool = False
) -> None:
    """
    Submit tasks to the Ray cluster for execution.

    Args:
        tasks (list[dict]): A list of task configuration dictionaries.
        args (argparse.Namespace): Parsed command-line arguments.
        runtime_env (dict | None): Ray runtime environment configuration containing:
            - pip (list[str] | None): Additional pip packages to install.
            - py_modules (list[str] | None): Python modules to include in the environment.
        concurrent (bool): Whether to launch tasks simultaneously as a batch,
                           or independently as resources become available.

    Returns:
        None
    """
    job_objs = []
    util.ray_init(ray_address=args.ray_address, runtime_env=runtime_env, log_to_driver=False)
    for task in tasks:
        resource = parse_task_resource(task)
        print(f"[INFO] Creating job {task['name']} with resource={resource}")
        job = util.Job(
            name=task["name"],
            py_args=task["py_args"],
            resources=resource,
            node=util.JobNode(
                specific=task.get("node", {}).get("specific"),
                hostname=task.get("node", {}).get("hostname"),
                node_id=task.get("node", {}).get("node_id"),
            ),
        )
        job_objs.append(job)
    start = datetime.now()
    print(f"[INFO] Creating {len(job_objs)} jobs at {start.strftime('%H:%M:%S.%f')} with runtime env={runtime_env}")
    # submit jobs
    util.submit_wrapped_jobs(
        jobs=job_objs,
        test_mode=args.test,
        concurrent=concurrent,
    )
    end = datetime.now()
    print(
        f"[INFO] All jobs completed at {end.strftime('%H:%M:%S.%f')}, took {(end - start).total_seconds():.2f} seconds."
    )


def main() -> None:
    """
    Main entry point for the Ray task runner script.

    Reads the YAML task configuration file, parses CLI arguments,
    and dispatches tasks to the Ray cluster.

    Returns:
        None
    """
    args = parse_args()
    with open(args.task_cfg) as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]
    runtime_env = {
        "pip": None if not config.get("pip") else config["pip"],
        "py_modules": None if not config.get("py_modules") else config["py_modules"],
    }
    concurrent = config.get("concurrent", False)
    run_tasks(
        tasks=tasks,
        args=args,
        runtime_env=runtime_env,
        concurrent=concurrent,
    )


if __name__ == "__main__":
    main()
