# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from ray import tune

# from ray.train import RunConfig


class IsaacLabTrainable(tune.Trainable):
    def setup(self, config):
        pass

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
        "executable",
        type=str,
        default="/workspace/isaaclab/_isaac_sim/python.sh",
        help="what executable to run the train script with ",
    )
    arg_parser.add_argument(
        "workflow_path",
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


# if __name__ == "__main__":
#     pass
