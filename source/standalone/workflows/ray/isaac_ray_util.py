# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import os
import re
import subprocess
from datetime import datetime

import ray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(directory: str) -> dict:
    """From a tensorboard directory, get the latest scalar values.

    Args:
        directory: The directory of the tensorboard logging.

    Returns:
        The latest available scalar values.
    """
    # Initialize the event accumulator with a size guidance for only the latest entry
    size_guidance = {"scalars": 1}  # Load only the latest entry for scalars
    event_acc = EventAccumulator(directory, size_guidance=size_guidance)
    event_acc.Reload()  # Load all data from the directory

    # Extract the latest scalars logged
    latest_scalars = {}
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        if events:  # Check if there is at least one entry
            latest_event = events[-1]  # Get the latest entry
            latest_scalars[tag] = latest_event.value
    return latest_scalars


def get_invocation_command_from_cfg(cfg: dict, python_cmd: str = "/workspace/isaaclab/isaaclab.sh -p") -> str:
    """Provided a python invocation, as well as a configuration with runner arguments and hydra
    arguments, combine them into a single shell command that can be run for a training run

    Args:
        cfg: A dict with the runner args and hydra args desired for a training run.
        python_cmd: Which python to use. Defaults to "/workspace/isaaclab/isaaclab.sh -p".

    Returns:
        A shell training run command (invocation)
    """
    runner_args = []
    hydra_args = []

    def process_args(args, target_list, is_hydra=False):
        for key, value in args.items():
            # for example, key: singletons | value: List
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    target_list.append(f" {subkey}={subvalue} ")
            elif isinstance(value, list):
                target_list.extend(value)
            else:
                if not is_hydra:
                    if "--" in key:  # must be command line argument
                        target_list.append(f"{key} {value}")
                    else:  # singleton like --headless or --enable_cameras
                        target_list.append(f"{value}")
                else:
                    target_list.append(f"{key}={value}")
            print(f"{target_list[-1]}")

    print(f"[INFO]: Starting workflow {cfg['workflow']}")

    process_args(cfg["runner_args"], runner_args)
    print(f"[INFO]: Retrieved workflow runner args: {runner_args}")
    process_args(cfg["hydra_args"], hydra_args, is_hydra=True)
    print(f"[INFO]: Retrieved hydra args: {hydra_args}")
    invoke_cmd = python_cmd + " " + cfg["workflow"] + " "
    invoke_cmd += " ".join(runner_args) + " " + " ".join(hydra_args)
    return invoke_cmd


@ray.remote
def remote_execute_job(
    job_cmd: str, identifier_string: str, test_mode: bool = False, extract_experiment: bool = False
) -> str | dict:
    """This method has an identical signature to :meth:execute_job , with the ray remote decorator"""
    return execute_job(
        job_cmd=job_cmd, identifier_string=identifier_string, test_mode=test_mode, extract_experiment=extract_experiment
    )


def execute_job(
    job_cmd: str,
    identifier_string: str = "job 0",
    test_mode: bool = False,
    extract_experiment: bool = False,
    persistent_dir: str | None = None,
) -> str | dict:
    """Issue a job (shell command).

    Args:
        job_cmd: The shell command to run.
        identifier_string: What prefix to add to make logs easier to differentiate
            across clusters or jobs. Defaults to "job 0".
        test_mode: When true, only run 'nvidia-smi'. Defaults to False.
        extract_experiment: When true, search for experiment details from a training run. Defaults to False.
        persistent_dir: When supplied, change to run the directory in a persistent
            directory. Can be used to avoid losing logs in the /tmp directory. Defaults to None.

    Raises:
        ValueError: If the job is unable to start, or throws an error. Most likely to happen
            due to running out of memory.

    Returns:
        Relevant information from the job
    """
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    result_details = [f"{identifier_string}: ---------------------------------"]
    result_details.append(f"\n{identifier_string}: Invocation job: {job_cmd}")

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
        print(f"{identifier_string} [INFO]: Invocation job {job_cmd}")

        if persistent_dir:
            og_dir = os.getcwd()
            os.chdir(persistent_dir)
        process = subprocess.Popen(
            job_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        if persistent_dir:
            os.chdir(og_dir)
        experiment_name = None
        logdir = None
        experiment_info_pattern = re.compile("Exact experiment name requested from command line: (.+)")
        logdir_pattern = re.compile(r"\[INFO\] Logging experiment in directory: (.+)$")
        err_pattern = re.compile("There was an error (.+)$")
        with process.stdout as stdout:
            for line in iter(stdout.readline, ""):
                line = line.strip()
                result_details.append(f"{identifier_string}: {line}")
                print(f"{identifier_string}: {line}")

                if extract_experiment:
                    exp_match = experiment_info_pattern.search(line)
                    log_match = logdir_pattern.search(line)
                    err_match = err_pattern.search(line)
                    if err_match:
                        raise ValueError("Encountered an error during trial run.")

                    if exp_match:
                        experiment_name = exp_match.group(1)
                    if log_match:
                        logdir = log_match.group(1)

                    if experiment_name and logdir:
                        result = {
                            "experiment_name": experiment_name,
                            "logdir": logdir,
                            "proc": process,
                            "result": " ".join(result_details),
                        }
                        return result

        with process.stderr as stderr:
            for line in iter(stderr.readline, ""):
                line = line.strip()
                result_details.append(f"{identifier_string}: {line}")
                print(f"{identifier_string}: {line}")

        process.wait()  # Wait for the subprocess to finish naturally if not exited early

    now = datetime.now().strftime("%H:%M:%S.%f")
    completion_info = f"{identifier_string}: Job Started at {start_time}, completed at {now}"
    print(completion_info)
    result_details.append(completion_info)

    return (" ".join(result_details),)


def get_gpu_node_resources(total_resources: bool = False, one_node_only: bool = False) -> dict:
    """Get information about available GPU node resources.

    Args:
        total_resources: When true, return total available resources. Defaults to False.
        one_node_only: When true, return resources for a single node. Defaults to False.

    Returns:
        Resource information.
    """
    if not ray.is_initialized():
        ray.init(address="auto")

    nodes = ray.nodes()
    node_resources_dict = {}
    total_cpus = 0
    total_gpus = 0
    total_memory = 0  # in bytes
    total_object_store_memory = 0  # in bytes

    for node in nodes:
        if node["Alive"] and "GPU" in node["Resources"]:
            node_id = node["NodeID"]
            resources = node["Resources"]
            cpus = resources.get("CPU", 0)
            gpus = resources.get("GPU", 0)
            memory = resources.get("memory", 0)
            object_store_memory = resources.get("object_store_memory", 0)

            node_resources_dict[node_id] = {"cpu": cpus, "gpu": gpus, "memory": memory}

            total_cpus += cpus
            total_gpus += gpus
            total_memory += memory
            total_object_store_memory += object_store_memory

    if total_resources:
        # Return summed total resources
        return {"cpu": total_cpus, "gpu": total_gpus, "memory": total_memory}

    if one_node_only and node_resources_dict:
        # Return resources of the first node in the dictionary
        first_node_id = list(node_resources_dict.keys())[0]
        return node_resources_dict[first_node_id]

    return node_resources_dict


def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, help="The name of the Ray Cluster to train on.")

    parser.add_argument(
        "--num_gpu_per_job",
        type=float,  # can actually do fractional GPUs if so desired
        help="The total amount of GPUs dispatched across all training job on the cluster",
    )
    parser.add_argument(
        "--num_cpu_per_job",
        type=float,
        help="The total amount of CPUs dispatched across all raining job on the cluster",
    )
    parser.add_argument(
        "--gb_ram_per_job",
        type=float,
        default=None,
        help="The total gigabytes of RAM dispatched across all training jobs on the cluster",
    )
    parser.add_argument(
        "--num_workers_per_node",
        type=int,
        help="Supply to split nodes into multiple workers. Meant for local development",
    )


def populate_isaac_ray_cfg_args(cfg: dict = {}) -> dict:
    """Small utility method to create empty fields if needed for a configuration."""
    if "runner_args" not in cfg:
        cfg["runner_args"] = {}
    if "hydra_args" not in cfg:
        cfg["hydra_args"] = {}
    return cfg
