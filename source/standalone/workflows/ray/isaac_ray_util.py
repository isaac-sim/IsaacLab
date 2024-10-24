# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import re
import subprocess
import time

import ray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(directory, max_entries=100):
    # Initialize the event accumulator with a size guidance
    size_guidance = {"scalars": max_entries}  # Limit the number of entries for scalars
    event_acc = EventAccumulator(directory, size_guidance=size_guidance)
    event_acc.Reload()  # Load all data from the directory

    # Extract all scalars logged
    scalars = {}
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        values = [event.value for event in events]
        scalars[tag] = values
    return scalars


def extract_experiment_info(output):
    """Extract experiment name and log directory from the subprocess output."""
    experiment_name_pattern = r"Exact experiment name requested from command line: (\S+)"
    logdir_pattern = r"\[INFO\] Logging experiment in directory: (.+)"

    experiment_name = None
    logdir = None

    # Iterate through the output lines
    for line in output.splitlines():
        experiment_match = re.search(experiment_name_pattern, line)
        logdir_match = re.search(logdir_pattern, line)

        if experiment_match:
            experiment_name = experiment_match.group(1)
        if logdir_match:
            logdir = logdir_match.group(1)

        # Break if both are found
        if experiment_name and logdir:
            break

    return experiment_name, logdir


def invoke_run(cfg, max_line_count=2000):
    runner_args = []
    hydra_args = []

    def process_args(args, target_list):
        for key, value in args.items():
            # for example, key: singletons | value: List
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    target_list.append(f" {subkey}={subvalue} ")
            elif isinstance(value, list):
                target_list.extend(value)
            else:
                target_list.append(f"{value}")
            print(f"{target_list[-1]}")

    print(f"[INFO]: Starting workflow {cfg['workflow']}")

    process_args(cfg["runner_args"], runner_args)
    print(f"[INFO]: Retrieved workflow runner args: {runner_args}")
    process_args(cfg["hydra_args"], hydra_args)
    print(f"[INFO]: Retrieved hydra args: {hydra_args}")

    proc = subprocess.Popen(
        ["/workspace/isaaclab/isaaclab.sh", "-p", cfg["workflow"], *runner_args, *hydra_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_output = ""
    lines_read = 0
    experiment_name = None
    logdir = None
    success_detected = False
    error_detected = False

    error_message = "There was an error running python"
    success_message = "epoch: 1/"  # Check for the first epoch start

    while lines_read < max_line_count and not (experiment_name and logdir and success_detected):
        line = proc.stdout.readline()
        if line:
            log_output += line
            lines_read += 1

            # Check for experiment info
            if not experiment_name or not logdir:
                experiment_match = re.search(r"Exact experiment name requested from command line: (\S+)", line)
                logdir_match = re.search(r"\[INFO\] Logging experiment in directory: (.+)", line)
                if experiment_match:
                    experiment_name = experiment_match.group(1)
                if logdir_match:
                    logdir = logdir_match.group(1)

            # Check for success or error
            if success_message in line:
                success_detected = True
            if error_message in line:
                error_detected = True
                break  # Stop processing if an error is detected

        time.sleep(0.1)  # Sleep to avoid busy wait

    if error_detected or experiment_name is None or logdir is None:
        print(f"Error during experiment run, or could not find logdir: \n {log_output}")
        return {"proc": None, "experiment_name": None, "logdir": None}
    print(f"[INFO]: Extracted experiment name {experiment_name} and logdir {logdir}")
    return {"proc": proc, "experiment_name": experiment_name, "logdir": logdir}


def get_gpu_node_resources(total_resources: bool = False, one_node_only: bool = False):
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


def populate_isaac_ray_cfg_args(cfg: dict = {}):
    if "runner_args" not in cfg:
        cfg["runner_args"] = {}
    if "hydra_args" not in cfg:
        cfg["hydra_args"] = {}
    return cfg
