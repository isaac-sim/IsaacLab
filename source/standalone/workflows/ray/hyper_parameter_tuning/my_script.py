# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
import torch

import ray

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@ray.remote(num_gpus=1)  # Each job requests 1 GPU
def get_available_gpus():
    logging.info("Job started.")
    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of available GPUs visible to this job: {num_gpus}")
    return f"Job completed on GPU with {num_gpus} GPUs visible."


def main(num_workers):
    # Define the specific Python environment
    runtime_env = {
        "executable": "/workspace/isaaclab/isaaclab/python.sh -p",  # Path to your Python environment
        # "pip": ["torch"] # TODO: Bad patch doesn't rlly work with other isaac things
    }

    # Initialize Ray with the correct runtime environment
    ray.init(address="auto", runtime_env=runtime_env)

    logging.info(f"Submitting {num_workers} GPU jobs...")

    job_results = []

    # Submit the jobs
    for i in range(num_workers):
        logging.info(f"Submitting job {i+1}/{num_workers}")
        job_results.append(get_available_gpus.remote())

    # Wait for all jobs to complete
    results = ray.get(job_results)

    for i, result in enumerate(results, 1):
        logging.info(f"Job {i} result: {result}")

    logging.info("All jobs completed.")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of GPU jobs to submit")
    args = parser.parse_args()

    # Run the main function
    main(args.num_workers)
