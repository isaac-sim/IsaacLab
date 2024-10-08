# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys

import ray
from ray import job_submission


# Read the ~/.ray_address file and filter out commented lines
def read_ray_addresses(file_path):
    # Expand the '~' to the user's home directory
    file_path = os.path.expanduser(file_path)

    with open(file_path) as file:
        addresses = [line.strip() for line in file if not line.startswith("#") and line.strip()]
    return addresses


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Submit jobs to Ray clusters with different addresses.")
    parser.add_argument(
        "--working-dir",
        default="source/standalone/workflows/ray",
        help="Working directory for the Ray job (default: source/standalone/workflows/ray)",
    )
    parser.add_argument(
        "--executable",
        default="/workspace/isaaclab/_isaac_sim/python.sh",
        help="Executable to run the jobs (default: /workspace/isaaclab/_isaac_sim/python.sh)",
    )
    parser.add_argument("jobs", nargs="+", help="Jobs to submit with their corresponding arguments.")

    return parser.parse_args()


# Function to submit a job using Ray's job submission API
def submit_job(job_with_args, address, working_dir, executable):
    # Connect to the Ray cluster
    job_submission_client = job_submission.JobSubmissionClient(f"http://{address.split(':')[0]}:8265")

    # Prepare the job command
    job_parts = job_with_args.split(" ")
    entrypoint = f"{executable} {' '.join(job_parts)}"

    print(f"Submitting job: {entrypoint} to cluster at address: {address}")

    # Submit the job
    job_id = job_submission_client.submit_job(
        entrypoint=entrypoint,
        runtime_env={"working_dir": working_dir},
    )

    # Print the dashboard URL for the user to track the job
    dashboard_address = f"http://{address.split(':')[0]}:8265"
    print(
        f"Job '{job_id}' submitted successfully.\nVisit the Ray dashboard at {dashboard_address} to view the status.\n"
    )


def main():
    # Parse arguments
    args = parse_arguments()

    # Read Ray cluster addresses
    ray_addresses = read_ray_addresses("~/.ray_address")

    # If only one job is provided, use it for all addresses
    if len(args.jobs) == 1:
        print(f"Only one job provided. Submitting the same job to all {len(ray_addresses)} Ray clusters.")
        args.jobs = args.jobs * len(ray_addresses)

    # Ensure there are enough addresses for the jobs
    if len(args.jobs) > len(ray_addresses):
        print("Error: More jobs than available Ray addresses.")
        exit(1)

    # Submit each job to the corresponding Ray cluster
    for job_with_args, address in zip(args.jobs, ray_addresses):
        submit_job(job_with_args, address, args.working_dir, args.executable)


if __name__ == "__main__":
    main()
