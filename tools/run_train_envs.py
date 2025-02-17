# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This scripts run training with rsl-rl library over a subset of the environments.

It calls the script ``scripts/reinforcement_learning/${args.lib_name}/train.py`` with the appropriate arguments.

Example usage:

.. code-block:: bash
    # for rsl-rl
    python run_train_envs.py --lib-name rsl_rl

    # for skrl
    python run_train_envs.py --lib-name skrl

"""

import argparse
import subprocess

from test_settings import ISAACLAB_PATH, TEST_RL_ENVS


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

    # train on each environment
    for env_name in TEST_RL_ENVS:
        # print a colored output to catch the attention of the user
        # this should be a multi-line print statement
        print("\033[91m==============================================\033[0m")
        print("\033[91m==============================================\033[0m")
        print(f"\033[91mTraining on {env_name} with {args.lib_name}...\033[0m")
        print("\033[91m==============================================\033[0m")
        print("\033[91m==============================================\033[0m")

        # run the training script
        subprocess.run(
            [
                f"{ISAACLAB_PATH}/isaaclab.sh",
                "-p",
                f"{ISAACLAB_PATH}/scripts/reinforcement_learning/{args.lib_name}/train.py",
                "--task",
                env_name,
                "--headless",
                "--run_name",
                git_commit_hash,
            ],
            check=False,  # do not raise an error if the script fails
        )


if __name__ == "__main__":
    args_cli = parse_args()
    main(args_cli)
