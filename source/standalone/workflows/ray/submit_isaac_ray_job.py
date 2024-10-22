# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from ray import job_submission

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
            logging.info(f"[INFO] Setting {cluster_info['name']}")  # with {cluster_info['num_gpu']} GPUs.")
            clusters.append(cluster_info)

    return clusters


def submit_job(cluster, job_command, test_mode):
    """
    Submits a job to a single cluster, prints the final result and Ray dashboard URL at the end.
    Adds optional test mode for GPU checking.
    """
    address = cluster["address"]
    cluster_name = cluster["name"]
    print(f"Submitting job to cluster '{cluster_name}' at {address}")  # with {num_gpus} GPUs.")
    client = job_submission.JobSubmissionClient(address)
    runtime_env = {"working_dir": CONFIG["working_dir"], "executable": CONFIG["executable"]}
    logging.info(f"Checking contents of the directory: {CONFIG['working_dir']}")
    try:
        dir_contents = os.listdir(CONFIG["working_dir"])
        logging.info(f"Directory contents: {dir_contents}")
    except Exception as e:
        logging.error(f"Failed to list directory contents: {str(e)}")

    wrapped_command = f"{WRAP_SCRIPT} {'--test' if test_mode else ''} {job_command if job_command else ' '}"

    entrypoint = f"{CONFIG['executable']} {wrapped_command}"
    print(f"{entrypoint = }")
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    status = client.get_job_status(job_id)
    while status in [job_submission.JobStatus.PENDING, job_submission.JobStatus.RUNNING]:
        time.sleep(5)
        status = client.get_job_status(job_id)

    final_logs = client.get_job_logs(job_id)
    print("----------------------------------------------------")
    print(f"Cluster {cluster_name} Logs: \n")
    print(final_logs)
    print("----------------------------------------------------")


def submit_jobs_to_clusters(jobs, clusters, test_mode):
    """
    Submit all jobs to their respective clusters.
    """
    with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
        for idx, cluster in enumerate(clusters):
            job_command = (
                jobs[idx] if idx < len(jobs) else None
            )  # Allow for running test mode without specific commands
            executor.submit(submit_job, cluster, job_command, test_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs to multiple Ray clusters.")
    parser.add_argument(
        "--jobs", nargs="*", required=False, help="Job commands to run on clusters, enclosed in quotes."
    )
    parser.add_argument("--test", action="store_true", help="Run with test mode enabled for all jobs.")

    args = parser.parse_args()

    # Read the cluster spec
    clusters = read_cluster_spec()

    # Submit the jobs to the clusters or run in test mode
    submit_jobs_to_clusters(args.jobs if args.jobs else [], clusters, args.test)
