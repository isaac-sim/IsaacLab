# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import ray
import util
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

"""
This script dispatches sub-job(s) (individual jobs, use :file:`tuner.py` for tuning jobs)
to worker(s) on GPU-enabled node(s) of a specific cluster as part of an resource-wrapped aggregate
job. If no desired compute resources for each sub-job are specified,
this script creates one worker per available node for each node with GPU(s) in the cluster.
If the desired resources for each sub-job is specified,
the maximum number of workers possible with the desired resources are created for each node
with GPU(s) in the cluster. It is also possible to split available node resources for each node
into the desired number of workers with the ``--num_workers`` flag, to be able to easily
parallelize sub-jobs on multi-GPU nodes. Due to Isaac Lab requiring a GPU,
this ignores all CPU only nodes such as loggers.

Sub-jobs are matched with node(s) in a cluster via the following relation:
sorted_nodes = Node sorted by descending GPUs, then descending CPUs, then descending RAM, then node ID
node_submitted_to = sorted_nodes[job_index % total_node_count]

To check the ordering of sorted nodes, supply the ``--test`` argument and run the script.

Sub-jobs are separated by the + delimiter. The ``--sub_jobs`` argument must be the last
argument supplied to the script.

If there is more than one available worker, and more than one sub-job,
sub-jobs will be executed in parallel. If there are more sub-jobs than workers, sub-jobs will
be dispatched to workers as they become available. There is no limit on the number
of sub-jobs that can be near-simultaneously submitted.

This script is meant to be executed on a Ray cluster head node as an aggregate cluster job.
To submit aggregate cluster jobs such as this script to one or more remote clusters,
see :file:`../submit_isaac_ray_job.py`.

KubeRay clusters on Google GKE can be created with :file:`../launch.py`

Usage:

.. code-block:: bash
    # **Ensure that sub-jobs are separated by the ``+`` delimiter.**
    # Generic Templates-----------------------------------
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py -h
    # No resource isolation; no parallelization:
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py
    --sub_jobs <JOB0>+<JOB1>+<JOB2>
    # Automatic Resource Isolation; Example A: needed for parallelization
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py \
    --num_workers <NUM_TO_DIVIDE_TOTAL_RESOURCES_BY> \
    --sub_jobs <JOB0>+<JOB1>
    # Manual Resource Isolation; Example B:  needed for parallelization
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py --num_cpu_per_worker <CPU> \
    --gpu_per_worker <GPU> --ram_gb_per_worker <RAM> --sub_jobs <JOB0>+<JOB1>
    # Manual Resource Isolation; Example C: Needed for parallelization, for heterogeneous workloads
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py --num_cpu_per_worker <CPU> \
    --gpu_per_worker <GPU1> <GPU2> --ram_gb_per_worker <RAM> --sub_jobs <JOB0>+<JOB1>
    # to see all arguments
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py -h
"""


def wrap_resources_to_jobs(jobs: list[str], args: argparse.Namespace) -> None:
    """
    Provided a list of jobs, dispatch jobs to one worker per available node,
    unless otherwise specified by resource constraints.

    Args:
        jobs: bash commands to execute on a Ray cluster
        args: The arguments for resource allocation

    """
    if not ray.is_initialized():
        ray.init(address=args.ray_address, log_to_driver=True)
    job_results = []
    gpu_node_resources = util.get_gpu_node_resources(include_id=True, include_gb_ram=True)

    if any([args.gpu_per_worker, args.cpu_per_worker, args.ram_gb_per_worker]) and args.num_workers:
        raise ValueError("Either specify only num_workers or only granular resources(GPU,CPU,RAM_GB).")

    num_nodes = len(gpu_node_resources)
    # Populate arguments
    formatted_node_resources = {
        "gpu_per_worker": [gpu_node_resources[i]["GPU"] for i in range(num_nodes)],
        "cpu_per_worker": [gpu_node_resources[i]["CPU"] for i in range(num_nodes)],
        "ram_gb_per_worker": [gpu_node_resources[i]["ram_gb"] for i in range(num_nodes)],
        "num_workers": args.num_workers,  # By default, 1 worker por node
    }
    args = util.fill_in_missing_resources(args, resources=formatted_node_resources, policy=min)
    print(f"[INFO]: Number of GPU nodes found: {num_nodes}")
    if args.test:
        jobs = ["nvidia-smi"] * num_nodes
    for i, job in enumerate(jobs):
        gpu_node = gpu_node_resources[i % num_nodes]
        print(f"[INFO]: Submitting job {i + 1} of {len(jobs)} with job '{job}' to node {gpu_node}")
        print(
            f"[INFO]: Resource parameters: GPU: {args.gpu_per_worker[i]}"
            f" CPU: {args.cpu_per_worker[i]} RAM {args.ram_gb_per_worker[i]}"
        )
        print(f"[INFO] For the node parameters, creating {args.num_workers[i]} workers")
        num_gpus = args.gpu_per_worker[i] / args.num_workers[i]
        num_cpus = args.cpu_per_worker[i] / args.num_workers[i]
        memory = (args.ram_gb_per_worker[i] * 1024**3) / args.num_workers[i]
        print(f"[INFO]: Requesting {num_gpus=} {num_cpus=} {memory=} id={gpu_node['id']}")
        job = util.remote_execute_job.options(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            memory=memory,
            scheduling_strategy=NodeAffinitySchedulingStrategy(gpu_node["id"], soft=False),
        ).remote(job, f"Job {i}", args.test)
        job_results.append(job)

    results = ray.get(job_results)
    for i, result in enumerate(results):
        print(f"[INFO]: Job {i} result: {result}")
    print("[INFO]: All jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    parser = util.add_resource_arguments(arg_parser=parser)
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
    parser.add_argument(
        "--sub_jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="This should be last wrapper argument. Jobs separated by the + delimiter to run on a cluster.",
    )
    args = parser.parse_args()
    if args.sub_jobs is not None:
        jobs = " ".join(args.sub_jobs)
        formatted_jobs = jobs.split("+")
    else:
        formatted_jobs = []
    print(f"[INFO]: Isaac Ray Wrapper received jobs {formatted_jobs=}")
    wrap_resources_to_jobs(jobs=formatted_jobs, args=args)
