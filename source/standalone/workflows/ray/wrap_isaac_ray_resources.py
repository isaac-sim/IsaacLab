# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import logging
import subprocess
from datetime import datetime
import isaac_ray_util
import ray

@ray.remote
def execute_command(command: str, test_mode: bool = False) -> str:
    import torch

    print("Job started.")
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    result_details = []

    if test_mode:
        print("Checking GPUs.")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.free,serial", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=True,
                text=True,
            )
            output = result.stdout.strip().split("\n")
            for gpu_info in output:
                name, memory_free, serial = gpu_info.split(", ")
                result_details.append({
                    "Name": name,
                    "Memory Available": f"{memory_free} MB",
                    "Serial Number": serial
                })
        except subprocess.CalledProcessError as e:
            logger.error(f"Error calling nvidia-smi: {e.stderr}")
            result_details.append({"error": "Failed to retrieve GPU information"})
    else:
        try:
            process = subprocess.Popen(
                command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            for line in process.stdout:
                print(line, end="")
            stdout, stderr = process.communicate()
            if stderr:
                logger.error(f"Error executing command: {stderr}")
        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
    num_gpus_detected = torch.cuda.device_count()
    now = datetime.now().strftime("%H:%M:%S.%f")
    result_str = (
        f"Job Started at {start_time}, completed at {now} | "
        f"# Detected GPUs: {num_gpus_detected} | Result details: {result_details}"
    )
    print(result_str)
    return result_str

def main(num_workers, commands, num_gpus, num_cpus, ram_gb, test_mode):
    ray.init(address="auto", log_to_driver=False)
    print("Connected to Ray cluster.")
    print("Assuming homogenous worker cluster resources.")
    print("Create more than one cluster for heterogenous jobs.")
    # Helper function to format resource information
    def format_resources(resources):
        """
        Formats the resources dictionary by converting memory units to gigabytes.
        """
        formatted_resources = {}
        for key, value in resources.items():
            if 'memory' in key.lower() or 'object_store_memory' in key.lower():
                # Convert bytes to gigabytes (Ray reports memory in bytes)
                gb_value = value / (1024 ** 3)
                formatted_resources[key] = [f"{gb_value:.2f}", "GB"]
            else:
                formatted_resources[key] = value
        return formatted_resources

    detailed_node_info = ray.nodes()

    if num_workers is None:
        num_workers = len(detailed_node_info) - 1 # one head Node

    print("Cluster resources before dispatching jobs:")
    for node in detailed_node_info:
        resources = node.get('Resources', {})
        node_ip = node.get('NodeManagerAddress')
        formatted_resources = format_resources(resources)
        print(f"Node {node_ip} resources: {formatted_resources}")
        head_node_pred = node.get('Resources', {}).get('node:__internal_head__', 0) > 0
        if not head_node_pred:
            if num_gpus is None:
                num_gpus = formatted_resources["GPU"]
            if num_cpus is None:
                num_cpus = formatted_resources["CPU"]
            if ram_gb is None:
                ram_gb = float(formatted_resources["memory"][0])
    job_results = []

    for i, command in enumerate(commands):
        print(f"Submitting job {i + 1} of {len(commands)} with command '{command}'")
        job = execute_command.options(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            memory=ram_gb * 1024
        ).remote(command, test_mode)
        job_results.append(job)

    results = ray.get(job_results)
    for i, result in enumerate(results, 1):
        print(f"Job {i} result: {result}")
    print("All jobs completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    isaac_ray_util.add_cluster_args(parser)
    parser.add_argument(
        "jobs",
        nargs=argparse.REMAINDER,
        help="Commands and their arguments to execute on workers.",
    )
    parser.add_argument("--test", action="store_true", help="Run nvidia-smi test instead of the arbitrary command")
    args = parser.parse_args()
    commands = isaac_ray_util.split_args_by_proceeding_py(args.jobs)
    main(args.num_workers, 
            commands, 
            args.num_gpu_per_job, 
            args.num_cpu_per_job, 
            args.gb_ram_per_job, 
            args.test)
