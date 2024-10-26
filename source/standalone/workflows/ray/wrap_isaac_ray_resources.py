# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import isaac_ray_util
import ray

"""
This script dispatches sub-job(s) (either individual jobs or tuning aggregate jobs)
to worker(s) on GPU-enabled node(s) of a specific cluster as part of an resource-wrapped aggregate
job. If no desired compute resources for each sub-job are specified,
this script creates one worker per available node for each node with GPU(s) in the cluster.
If the desired resources for each sub-job is specified,
the maximum number of workers possible with the desired resources are created for each node
with GPU(s) in the cluster. It is also possible to split available node resources for each node
into the desired number of workers with the ``--num_workers_per_node`` flag, to be able to easily
parallelize sub-jobs on multi-GPU nodes. Due to Isaac Lab requiring a GPU,
this ignores all CPU only nodes such as loggers.

Sub-jobs are separated by the + delimiter. The ``--jobs`` argument must be the last
argument supplied to the script.

If there is more than one available worker, and more than one sub-job,
sub-jobs will be executed in parallel. If there are more sub-jobs than workers, sub-jobs will
be dispatched to workers as they become available. There is no limit on the number
of sub-jobs that can be near-simultaneously submitted.

This assumes that all workers in a cluster are homogeneous. For heterogeneous workloads,
create several heterogeneous clusters (with homogeneous nodes in each cluster),
then submit several overall-cluster jobs with :file:`../submit_isaac_ray_job.py`.
KubeRay clusters on Google GKE can be created with :file:`../launch.py`

Usage:

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py -h
"""


def wrap_resources_to_jobs(
    jobs: list[str],
    num_workers: int | None,
    num_gpus: float | None,
    num_cpus: float | None,
    ram_gb: float | None,
    test_mode: bool = False,
    ray_address: str = "auto",
) -> None:
    """
    Provided a list of jobs, dispatch jobs to one worker per available node,
    unless otherwise specified by resource constraints.

    Args:
        jobs: bash commands to execute on a Ray cluster
        num_workers: How many workers to split each node into. If None is ignored
        num_gpus: How many GPUs to allocate per worker. If None is ignored
        num_cpus: How many CPUs to allocate per worker. If None is ignored
        ram_gb: How many gigabytes of RAM to allocate per worker. If None is ignore
        test_mode: If set to true, ignore jobs, and try only nvidia-smi. Defaults to False.
        ray_address: What ray address to connect to. Defaults to 'auto'

    """
    if not ray.is_initialized():
        ray.init(address=ray_address, log_to_driver=True)
    print("[INFO]: Connected to Ray cluster.")
    print("[WARNING]: Assuming homogeneous worker cluster resources.")
    print("[WARNING]: Create more than one cluster for heterogeneous jobs.")

    # Helper function to format resource information
    def format_resources(resources):
        """
        Formats the resources dictionary by converting memory units to gigabytes.
        """
        formatted_resources = {}
        for key, value in resources.items():
            if "memory" in key.lower() or "object_store_memory" in key.lower():
                # Convert bytes to gigabytes (Ray reports memory in bytes)
                gb_value = value / 1024**3
                formatted_resources[key] = [f"{gb_value:.2f}", "GB"]
            else:
                formatted_resources[key] = value
        return formatted_resources

    detailed_node_info = ray.nodes()

    print("[INFO]: Cluster Resource Information")
    num_gpu_nodes = 0
    for node in detailed_node_info:
        resources = node.get("Resources", {})
        formatted_resources = format_resources(resources)
        # print(f"Node {node_ip} resources: {formatted_resources}")
        # If local, head node has all resources
        # If remote, want to ignore head node
        # Assuming remote workers nodes are spec'd more heavily than head node
        # Assume all worker nodes are homogeneous
        if "GPU" in formatted_resources:
            if num_gpus is None:
                num_gpus = formatted_resources["GPU"]
            num_gpu_nodes += 1
        if num_cpus is None or ("CPU" in formatted_resources and num_cpus < formatted_resources["CPU"]):
            num_cpus = formatted_resources["CPU"]
        if ram_gb is None or ("memory" in formatted_resources and ram_gb < float(formatted_resources["memory"][0])):
            ram_gb = float(formatted_resources["memory"][0])
    job_results = []

    if num_workers:
        print(f"[WARNING]: For each node, splitting cluster resources into {num_workers}")
        num_gpus /= num_workers
        num_cpus /= num_workers
        ram_gb /= num_workers

    print(f"[INFO]: Number of GPU nodes found: {num_gpu_nodes}")
    print(f"[INFO]: Requesting resources: {num_gpus = } {num_cpus = } {ram_gb = }")

    for i, job in enumerate(jobs):
        print(f"Submitting job {i + 1} of {len(jobs)} with job '{job}'")
        job = isaac_ray_util.remote_execute_job.options(
            num_gpus=num_gpus, num_cpus=num_cpus, memory=ram_gb * 1024
        ).remote(job, f"Job {i}", test_mode)
        job_results.append(job)

    results = ray.get(job_results)
    for i, result in enumerate(results):
        print(f"[INFO]: Job {i} result: {result}")
    print("[INFO]: All jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    parser.add_argument("--name", type=str, help="The name of the Ray Cluster to train on.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    parser.add_argument(
        "--num_gpu_per_job",
        type=float,
        help="The number of GPUS to use per on-cluster job.",
    )
    parser.add_argument(
        "--num_cpu_per_job",
        type=float,
        help="The number of CPUS to use per on-cluster job.",
    )
    parser.add_argument(
        "--gb_ram_per_job",
        type=float,
        default=None,
        help="The gigabytes of RAM to user per on-cluster job",
    )
    parser.add_argument(
        "--num_workers_per_node",
        type=int,
        help="Supply to split each node into num_workers evenly.",
    )
    parser.add_argument("--test", action="store_true", help="Run nvidia-smi test instead of the arbitrary job")
    parser.add_argument(
        "--sub_jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="This should be last wrapper argument. Jobs separated by the + delimiter to run on a cluster.",
    )
    args = parser.parse_args()

    jobs = " ".join(args.sub_jobs)
    formatted_jobs = jobs.split("+")
    print(f"[INFO]: Isaac Ray Wrapper received jobs {formatted_jobs = }")
    wrap_resources_to_jobs(
        jobs=formatted_jobs,
        num_workers=args.num_workers_per_node,
        num_gpus=args.num_gpu_per_job,
        num_cpus=args.num_cpu_per_job,
        ram_gb=args.gb_ram_per_job,
        test_mode=args.test,
        ray_address=args.ray_address,
    )
