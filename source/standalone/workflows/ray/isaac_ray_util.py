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


#     parser.add_argument(
#     "--autotune_max_percentage_util",
#     nargs="+",
#     type=float,
#     default=[100.0, 80.0, 80.0, 80.0],
#     required=False,
#     help=(
#         "The system utilization percentage thresholds to reach before an autotune is finished. "
#         "If any one of these limits are hit, the autotune stops."
#         "Thresholds are, in order, maximum CPU percentage utilization,"
#         "maximum RAM percentage utilization, maximum GPU compute percent utilization, "
#         "amd maximum GPU memory utilization."
#     ),
# )


def add_cluster_args(parser: argparse.ArgumentParser) -> None:
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
