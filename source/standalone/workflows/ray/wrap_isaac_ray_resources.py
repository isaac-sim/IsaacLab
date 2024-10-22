# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import subprocess
from datetime import datetime

import isaac_ray_util
import ray


@ray.remote
def execute_command(command: str, identifier_string: str, test_mode: bool = False) -> str:
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    result_details = []
    result_details.append("---------------------------------")
    result_details.append(f"\n Invocation command: {command}")
    if test_mode:
        import torch

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
                result_details.append({"Name": name, "Memory Available": f"{memory_free} MB", "Serial Number": serial})
            num_gpus_detected = torch.cuda.device_count()
            result_details.append(f"# Detected GPUs from PyTorch: {num_gpus_detected}")
        except subprocess.CalledProcessError as e:
            print(f"Error calling nvidia-smi: {e.stderr}")
            result_details.append({"error": "Failed to retrieve GPU information"})
    else:
        try:

            def prepend_identifier(text):
                return f"{identifier_string}: {text}"

            # Start the subprocess and set up pipes for real-time output
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
            # Use a list to collect output for final summary
            output_lines = []
            with process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(prepend_identifier(line.strip()))  # Print in real-time
                    output_lines.append(line.strip())  # Collect for summary
            with process.stderr:
                for line in iter(process.stderr.readline, ""):
                    print(prepend_identifier(line.strip()))  # Print errors in real-time
                    output_lines.append(line.strip())  # Collect for summary
            process.wait()  # Wait for the process to complete
            result_details.extend([prepend_identifier(line) + "\n" for line in output_lines])
        except subprocess.SubprocessError as e:
            print(prepend_identifier(f"Exception during subprocess execution: {str(e)}"))
            result_details.append("error: Exception occurred during command execution")

    now = datetime.now().strftime("%H:%M:%S.%f")
    print(prepend_identifier(f"Job Started at {start_time}, completed at {now}"))
    result_str = f"Job Started at {start_time}, completed at {now} | Result details: {' '.join(result_details)}"
    return result_str


def main(num_workers: int, jobs: list[str], num_gpus: float, num_cpus: float, ram_gb: float, test_mode: bool):
    ray.init(address="auto", log_to_driver=True)
    print("Connected to Ray cluster.")
    print("Assuming homogeneous worker cluster resources.")
    print("Create more than one cluster for heterogeneous jobs.")

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

    print("Cluster resources:")
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
        if num_cpus is None or num_cpus < formatted_resources["CPU"]:
            num_cpus = formatted_resources["CPU"]
        if ram_gb is None or ram_gb < float(formatted_resources["memory"][0]):
            ram_gb = float(formatted_resources["memory"][0])
    job_results = []

    if num_workers:
        print(f"[WARNING]: For each node, splitting cluster resources into {num_workers}")
        num_gpus /= num_workers
        num_cpus /= num_workers
        ram_gb /= num_workers

    print(f"[INFO]: Number of GPU nodes found: {num_gpu_nodes}")
    print(f"[INFO]: Requesting resources: {num_gpus = } {num_cpus = } {ram_gb = }")

    for i, command in enumerate(jobs):
        print(f"Submitting job {i + 1} of {len(jobs)} with command '{command}'")
        job = execute_command.options(num_gpus=num_gpus, num_cpus=num_cpus, memory=ram_gb * 1024).remote(
            command, f"Job {i}", test_mode
        )
        job_results.append(job)

    results = ray.get(job_results)
    for i, result in enumerate(results, 1):
        print(f"Job {i} result: {result}")
    print("All jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    isaac_ray_util.add_cluster_args(parser)
    parser.add_argument("--test", action="store_true", help="Run nvidia-smi test instead of the arbitrary command")
    parser.add_argument(
        "--jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="!! This should be last wrapper argument!!! Commands and their arguments to execute on workers.",
    )
    args = parser.parse_args()
    print(f"Received jobs {args.jobs = }")
    jobs = " ".join(args.jobs)
    formatted_jobs = jobs.split("+")
    main(
        args.num_workers_per_node,
        formatted_jobs,
        args.num_gpu_per_job,
        args.num_cpu_per_job,
        args.gb_ram_per_job,
        args.test,
    )
