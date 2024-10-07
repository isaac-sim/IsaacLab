# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
import torch
from omni.isaac.lab.app import AppLauncher # check that you can import Isaac Lab
import ray
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@ray.remote(num_gpus=1)  # Each job requests 1 GPU
def get_available_gpus() -> str:
    """
    Check GPU availability with nvidia-smi without depending on PyNVML.

    Returns:
        str: a description of GPUs on the system.
    """
    logging.info("Job started.")

    num_gpus = torch.cuda.device_count()
    gpu_details = []

    for i in range(num_gpus):
        # Use subprocess to call nvidia-smi and gather GPU info
        try:
            # Get the full output of nvidia-smi for detailed GPU information
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,serial', '--format=csv,noheader,nounits'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            
            # Parse the output, which is in CSV format
            output = result.stdout.strip().split('\n')
            for gpu_info in output:
                name, memory_free, serial = gpu_info.split(', ')
                gpu_details.append({
                    "Name": name,
                    "Memory Available": f"{memory_free} MB",
                    "Serial Number": serial
                })
        except subprocess.CalledProcessError as e:
            logging.error(f"Error calling nvidia-smi: {e.stderr}")
            gpu_details.append({"error": "Failed to retrieve GPU information"})

    logging.info(f"Number of available GPUs visible to this job: {num_gpus}")
    logging.info(f"GPU Details: {gpu_details}")
    return f"Job completed on GPU with {num_gpus} GPUs visible, details: {gpu_details}"

def main(num_workers):
    # Define the specific Python environment
    runtime_env = {
        # "pip": ["pynvml"], # warning: this will not work with the way Ray is configured.
        # Add your dependency to the dockerfile instead.
        "executable": "/workspace/isaaclab/_isaac_sim/python.sh",  # Path to your Python environment
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
    parser.add_argument("--num_jobs", type=int, default=2, help="Number of GPU jobs to submit")
    args = parser.parse_args()

    # Run the main function
    main(args.num_jobs)
