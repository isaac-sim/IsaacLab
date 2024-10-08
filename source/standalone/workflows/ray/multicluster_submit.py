# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

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
    parser.add_argument("job", help="Job to submit (e.g., script to run)")
    parser.add_argument("job_args", nargs=argparse.REMAINDER, help="Additional arguments for the job")

    return parser.parse_args()


# Function to submit a job using Ray's job submission API
def submit_job(job, job_args, address, working_dir, executable):
    # Connect to the Ray cluster
    job_submission_client = job_submission.JobSubmissionClient(f"http://{address.split(':')[0]}:8265")

    # Prepare the job command with any extra arguments
    entrypoint = f"{executable} {job} {' '.join(job_args)}"

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
    print(f"Submitting the job '{args.job}' with args '{args.job_args}' to all {len(ray_addresses)} Ray clusters.")

    # Submit each job to the corresponding Ray cluster
    for address in ray_addresses:
        submit_job(args.job, args.job_args, address, args.working_dir, args.executable)


if __name__ == "__main__":
    main()
