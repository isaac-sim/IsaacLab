# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from ray import job_submission

script_directory = os.path.dirname(os.path.abspath(__file__))

# Consolidated configuration
CONFIG = {"working_dir": script_directory, "executable": "/workspace/isaaclab/isaaclab.sh -p"}

WRAP_SCRIPT = "wrap_isaac_ray_resources.py"


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
            cluster_info = {"name": parts[1], "address": http_address}
            print(f"[INFO] Setting {cluster_info['name']}")  # with {cluster_info['num_gpu']} GPUs.")
            clusters.append(cluster_info)

    return clusters


def submit_job(cluster: dict, job_command: str, test_mode: bool):
    """
    Submits a job to a single cluster, prints the final result and Ray dashboard URL at the end.
    Adds optional test mode for GPU checking.
    """
    address = cluster["address"]
    cluster_name = cluster["name"]
    print(f"Submitting job to cluster '{cluster_name}' at {address}")  # with {num_gpus} GPUs.")
    client = job_submission.JobSubmissionClient(address)
    runtime_env = {"working_dir": CONFIG["working_dir"], "executable": CONFIG["executable"]}
    print(f"[INFO]: Checking contents of the directory: {CONFIG['working_dir']}")
    try:
        dir_contents = os.listdir(CONFIG["working_dir"])
        print(f"[INFO]: Directory contents: {dir_contents}")
    except Exception as e:
        print(f"[INFO] Failed to list directory contents: {str(e)}")

    wrapped_command = f"{WRAP_SCRIPT + ' --test' if test_mode else ''} {job_command if job_command else ' '}"

    entrypoint = f"{CONFIG['executable']} {wrapped_command}"
    print(f"[INFO]: Attempting entrypoint {entrypoint = } in cluster {cluster}")
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    status = client.get_job_status(job_id)
    while status in [job_submission.JobStatus.PENDING, job_submission.JobStatus.RUNNING]:
        time.sleep(5)
        status = client.get_job_status(job_id)

    final_logs = client.get_job_logs(job_id)
    print("----------------------------------------------------")
    print(f"[INFO]: Cluster {cluster_name} Logs: \n")
    print(final_logs)
    print("----------------------------------------------------")


def submit_jobs_to_clusters(jobs: list[str], clusters: list[dict], test_mode: bool):
    """
    Submit all jobs to their respective clusters, cycling through clusters if there are more jobs than clusters.
    """
    if not clusters:
        raise ValueError("No clusters available for job submission.")

    if test_mode:
        jobs = [""] * len(clusters)  # Test mode will populate the jobs correctly, want to submit to all clusters
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        for idx, job_command in enumerate(jobs):
            # Cycle through clusters using modulus to wrap around if there are more jobs than clusters
            cluster = clusters[idx % len(clusters)]
            executor.submit(submit_job, cluster, job_command, test_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs to multiple Ray clusters.")
    parser.add_argument("--test", action="store_true", help="Run with test mode enabled for all jobs.")
    parser.add_argument(
        "--jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help=(
            "This should be last argument. Jobs separated by the + delimiter to run on a cluster. "
            "For more than one cluster, separate cluster dispatches by the * delimiter. "
            "For more than one cluster, jobs are matched with the ~/.cluster_config in the order "
            "that they appear. If there are more jobs than clusters, they will be submitted in "
            "modulus order that they appear. (Say with clusters c1 and c2, and jobs j1 j2 j3 j4) "
            "jobs j1 and j3 will be submitted to cluster c1, and jobs j2 and j4 will be submitted to cluster c2. "
        ),
    )
    args = parser.parse_args()
    if args.jobs is not None:
        jobs = " ".join(args.jobs)
        formatted_jobs = jobs.split("*")
        if len(formatted_jobs) > 1:
            print("Warning; Split jobs by cluster with the * delimiter")
    else:
        formatted_jobs = []
    print(f"[INFO]: Isaac Ray Wrapper received jobs {formatted_jobs = }")
    # Read the cluster spec
    clusters = read_cluster_spec()
    # Submit the jobs to the clusters or run in test mode
    submit_jobs_to_clusters(formatted_jobs, clusters, args.test)
