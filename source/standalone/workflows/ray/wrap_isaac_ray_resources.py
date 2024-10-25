# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse

import isaac_ray_util
import ray


def wrap_resources_to_jobs(
    jobs: list[str],
    num_workers: int | None,
    num_gpus: float | None,
    num_cpus: float | None,
    ram_gb: float | None,
    test_mode: bool = False,
) -> None:
    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=True)
    print("Connected to Ray cluster.")
    print("[INFO]: Assuming homogeneous worker cluster resources.")
    print("[INFO]: Create more than one cluster for heterogeneous jobs.")

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
        print(f"Job {i} result: {result}")
    print("All jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    isaac_ray_util.add_cluster_args(parser)
    parser.add_argument("--test", action="store_true", help="Run nvidia-smi test instead of the arbitrary job")
    parser.add_argument(
        "--jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="This should be last wrapper argument. Jobs separated by the + delimiter to run on a cluster.",
    )
    args = parser.parse_args()

    jobs = " ".join(args.jobs)
    formatted_jobs = jobs.split("+")
    print(f"[INFO]: Isaac Ray Wrapper received jobs {formatted_jobs = }")
    wrap_resources_to_jobs(
        formatted_jobs,
        args.num_workers_per_node,
        args.num_gpu_per_job,
        args.num_cpu_per_job,
        args.gb_ram_per_job,
        args.test,
    )
