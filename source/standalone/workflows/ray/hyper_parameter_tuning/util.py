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


def parse_tune_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser("Submit distributed hyperparameter tuning jobs.")

    arg_parser.add_argument(
        "--executable_path",
        type=str,
        default="/workspace/isaaclab/_isaac_sim/python.sh",
        help="what executable to run the train script with ",
    )

    arg_parser.add_argument(
        "--workflow_path",
        type=str,
        required=False,
        default="/workspace/isaaclab/source/standalone/workflows/rl_games/train.py",
    )

    arg_parser.add_argument(
        "--args",
        nargs="+",
        type=str,
        default=[],
        required=False,
        help="Arguments to pass to the training script.For example, you could pass the task here.",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_tune_args()

    trainable = IsaacLabTuneTrainable(args.executable_path, args.workflow_path, args.args)
