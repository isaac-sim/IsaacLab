# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ray import job_submission

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

script_directory = os.path.dirname(os.path.abspath(__file__))

# Consolidated configuration
CONFIG = {"working_dir": script_directory, "executable": "/workspace/isaaclab/isaaclab.sh -p"}


def read_cluster_spec(fn: str | None = None) -> list[dict]:
    if fn is None:
        cluster_spec_path = os.path.expanduser("~/.cluster_config")
    else:
        cluster_spec_path = os.path.expanduser(fn)

    if not os.path.exists(cluster_spec_path):
        raise FileNotFoundError(f"Cluster spec file not found at {cluster_spec_path}")

    clusters = []
    with open(cluster_spec_path) as f:
        for line in f:
            parts = line.strip().split(" ")
            http_address = parts[3]
            cluster_info = {
                "name": parts[1],
                "address": http_address,
                "num_cpu": float(parts[5]),
                "num_gpu": int(float(parts[7])),
                "ram_gb": float(parts[9]),
                "num_workers": int(float(parts[11])),
            }
            logging.info(f"[INFO] Setting {cluster_info['name']} with {cluster_info['num_gpu']} GPUs.")
            clusters.append(cluster_info)

    return clusters


def submit_job(cluster, job_command):
    """
    Submits a job to a single cluster, prints the final result and Ray dashboard URL at the end.

    Args:
        cluster (dict): Information about the cluster.
        job_command (str): The command to execute as a job on the cluster.

    Returns:
        None
    """
    address = cluster["address"]
    num_gpus = cluster["num_gpu"]  # Extract GPU info for this cluster
    cluster_name = cluster["name"]

    print(f"Submitting job to cluster '{cluster_name}' at {address} with {num_gpus} GPUs.")

    # Create a JobSubmissionClient for the cluster
    client = job_submission.JobSubmissionClient(address)

    # Define the runtime environment (using consolidated config)
    runtime_env = {"working_dir": CONFIG["working_dir"], "executable": CONFIG["executable"]}

    # Prepare the job entrypoint
    entrypoint = f"{CONFIG['executable']} {job_command} "
    entrypoint += f"--cluster_gpu_count {num_gpus} --cluster_cpu_count "
    entrypoint += f"{cluster['num_cpu']} --cluster_ram_gb {cluster['ram_gb']} --num_workers {cluster['num_workers']}"

    # Submit the job
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)

    # Monitor the job without printing logs
    status = client.get_job_status(job_id)
    while status in [job_submission.JobStatus.PENDING, job_submission.JobStatus.RUNNING]:
        time.sleep(5)  # Poll every 5 seconds to check the status
        status = client.get_job_status(job_id)

    # Print the final result once the job finishes
    final_logs = client.get_job_logs(job_id)
    print("----------------------------------------------------")
    print(f"Cluster {cluster_name} Logs: \n")
    print(final_logs)
    print("----------------------------------------------------")


def submit_jobs_to_clusters(jobs, clusters):
    if len(jobs) != len(clusters):
        if len(jobs) == 1:  # If only one job is provided, replicate it for all clusters
            jobs = jobs * len(clusters)
        else:
            raise ValueError("Number of jobs does not match the number of clusters.")
    for job, cluster in zip(jobs, clusters):
        print(f"{job = } {cluster = }")
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        [executor.submit(submit_job, cluster, jobs[idx]) for idx, cluster in enumerate(clusters)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs to multiple Ray clusters.")
    parser.add_argument("--jobs", nargs="+", help="Job commands to run on clusters, enclosed in quotes.")
    args, unknown = parser.parse_known_args()  # Accept unknown args

    # Read the cluster spec and submit the jobs
    clusters = read_cluster_spec()

    # Ensure there's at least one job command
    if not args.jobs:
        args.jobs = [" ".join(sys.argv[1:])]  # Take all arguments as a single job if --jobs isn't explicitly used

    if args.jobs:
        full_jobs = [" ".join([job] + unknown) for job in args.jobs]
        submit_jobs_to_clusters(full_jobs, clusters)
    else:
        logging.error("No jobs specified. Please provide at least one job command.")
