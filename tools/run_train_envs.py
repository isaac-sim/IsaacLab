# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This scripts run training with rsl-rl library over a subset of the environments.


It calls the script ``scripts/reinforcement_learning/rsl_rl/train.py`` with the appropriate arguments.
"""

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib-name",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "skrl", "rl_games", "sb3"],
        help="The name of the library to use for training.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """The main function."""
    # get the git commit hash
    git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    # list of environments to train on
    name_of_envs = [
        # classic control
        "Isaac-Ant-v0",
        "Isaac-Cartpole-v0",
        # manipulation
        "Isaac-Lift-Cube-Franka-v0",
        "Isaac-Open-Drawer-Franka-v0",
        # dexterous manipulation
        "Isaac-Repose-Cube-Allegro-v0",
        # locomotion
        "Isaac-Velocity-Flat-Anymal-D-v0",
        "Isaac-Velocity-Rough-Anymal-D-v0",
        "Isaac-Velocity-Rough-G1-v0",
    ]

    # train on each environment
    for env_name in name_of_envs:
        subprocess.run(
            [
                "python",
                f"scripts/reinforcement_learning/{args.lib_name}/train.py",
                "--task",
                env_name,
                "--headless",
                "--run_name",
                git_commit_hash,
            ],
            check=True,
        )


if __name__ == "__main__":
    args_cli = parse_args()
    main(args_cli)
