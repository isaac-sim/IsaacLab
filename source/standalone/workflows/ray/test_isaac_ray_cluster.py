# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
import subprocess
import torch

import isaac_ray_util
import ray

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@ray.remote  # Modify the number of GPUs dynamically when submitting jobs
def get_available_gpus(num_gpus) -> str:
    """
    Check GPU availability with nvidia-smi without depending on PyNVML.

    Args:
        num_gpus (int): Number of GPUs assigned to this job.

    Returns:
        str: A description of GPUs on the system.
    """
    logging.info("Job started.")
    logging.info("Checking GPUS.")
    gpu_details = []

    for i in range(num_gpus):
        # Use subprocess to call nvidia-smi and gather GPU info
        try:
            # Get the full output of nvidia-smi for detailed GPU information
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.free,serial", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=True,
                text=True,
            )

            # Parse the output, which is in CSV format
            output = result.stdout.strip().split("\n")
            for gpu_info in output:
                name, memory_free, serial = gpu_info.split(", ")
                gpu_details.append({"Name": name, "Memory Available": f"{memory_free} MB", "Serial Number": serial})
        except subprocess.CalledProcessError as e:
            logging.error(f"Error calling nvidia-smi: {e.stderr}")
            gpu_details.append({"error": "Failed to retrieve GPU information"})

    logging.info(f"Number of GPUs allocated to this job: {num_gpus}")
    logging.info(f"GPU Details: {gpu_details}")
    logging.info("Attempting to read GPUS from torch....")
    num_gpus = torch.cuda.device_count()

    logging.info("Checking Isaac...")

    from omni.isaac.lab.app import AppLauncher  # noqa: F401  # check that you can import Isaac Lab

    app_launcher = AppLauncher(headlesss=True, enable_cameras=True)
    # app_launcher.app.close() # Ray wull clean this up alone.
    return f"Job completed with {num_gpus} GPUs, details: {gpu_details}. Torch sees {num_gpus} GPUs."


def main(num_jobs, gpus_per_job):
    # Initialize Ray with the correct runtime environment
    ray.init(address="auto")

    logging.info(f"Submitting {num_jobs} GPU jobs with {gpus_per_job} GPU(s) per job...")

    job_results = []

    # Submit the jobs with the specified number of GPUs per job
    for i in range(num_jobs):
        logging.info(f"Submitting job {i+1}/{num_jobs}")
        logging.info(f"Using {gpus_per_job}")
        job_results.append(get_available_gpus.options(num_gpus=gpus_per_job).remote(gpus_per_job))

    # Wait for all jobs to complete
    results = ray.get(job_results)

    for i, result in enumerate(results, 1):
        logging.info(f"Job {i} result: {result}")

    logging.info("All jobs completed.")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs.")
    parser.add_argument("--num_cycles", type=int, default=1, help="Number of times to resubmit GPU jobs")
    isaac_ray_util.add_cluster_args(parser)  # accept cluster info from submit isaac_ray_job.py

    # When this script is called, cluster_gpu_count will be passed. However, this is overall, not per node.
    args = parser.parse_args()

    # Run the main function
    # Assumes evenly distributed resources across workers.
    # For each worker, issue num_jobs for the number of GPUs available.
    main(args.num_cycles * args.num_workers, args.cluster_gpu_count // args.num_workers)
