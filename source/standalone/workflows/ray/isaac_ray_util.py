# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from ray import tune

# from ray.train import RunConfig


class IsaacLabTuneTrainable(tune.Trainable):
    def __init__(self, executable_path, workflow_path, args):
        self.invocation_str = executable_path + " " + workflow_path
        for arg in args:
            spaced_arg = " " + arg + " "
            self.invocation_str += spaced_arg
        print(f"[INFO] Using base invocation of {self.invocation_str} for all trials")

    def setup(self, config):
        print(f"[INFO]: From base invocation of {self.invocation_str}, adding the following config:")

        # invocation_string_with_hydra_hooks
        for key, value in config.items():
            print("---")
            print(f"{key = }: {value = }")
            print("----")

    def step(self):
        pass

import subprocess
import re
import time
import ray
from ray import tune

def extract_experiment_info(output):
    """Extract experiment name and log directory from the subprocess output."""
    experiment_name_pattern = r'Exact experiment name requested from command line: (\S+)'
    logdir_pattern = r'\[INFO\] Logging experiment in directory: (.+)'

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


def invoke_run(workflow: str, 
    run_cmd_args: list[str], 
    hydra_param_args: list[str]):
    """Invoke a training run with the desired parameters.
    
    This is a subprocess, as opposed to calling a method, due to
    """
    """Ray Tune trainable function that monitors the subprocess and extracts logs."""
    
    # Start the subprocess
    proc = subprocess.Popen(
        ["./workspace/isaaclab.sh -p ", workflow, *run_cmd_args, *hydra_param_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    log_output = ""
    
    # try:
    #     while proc.poll() is None:
    #         # Read output line by line
    #         line = proc.stdout.readline()
    #         if line:
    #             print(line.strip())  # Optionally print or log the output
    #             log_output += line

    #         # Check for early stopping by Ray Tune
    #         if tune.get_trial_resources().trial_runner.should_stop_trial():
    #             break

    #         time.sleep(.1)

    #     # Extract experiment info from the subprocess output
    #     experiment_name, logdir = extract_experiment_info(log_output)

    #     if experiment_name and logdir:
    #         # Log this info into Ray Tune
    #         return {"experiment_name": experiment_name, 
    #                 "logdir": logdir}

def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--name",
        type=str,
        help="The name of the Ray Cluster you'd like to train on.")

    parser.add_argument(
        "--cluster_gpu_count",
        type=int,  # can actually do fractional GPUs if so desired
        help="The total amount of GPUs dispatched across all training job on the cluster",
    )
    parser.add_argument(
        "--cluster_cpu_count",
        type=float,
        help="The total amount of CPUs dispatched across all raining job on the cluster",
    )
    parser.add_argument(
        "--cluster_ram_gb",
        type=float,
        help="The total gigabytes of RAM dispatched across all training jobs on the cluster",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help=(
            "The total number of workers available across the entire cluster."
            "Assumes that resources are equally distributed across cluster workers."
        ),
    )


if __name__ == "__main__":
    pass
    # trainable = IsaacLabTuneTrainable(args.executable_path, args.workflow_path, args.args)
