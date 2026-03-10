# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from .commands.envs import command_setup_conda, command_setup_uv
from .commands.format import command_format
from .commands.install import command_install
from .commands.misc import (
    command_build_docs,
    command_new,
    command_run_docker,
    command_run_isaacsim,
    command_test,
    command_vscode_settings,
)
from .utils import (
    is_windows,
    run_python_command,
)


def cli() -> None:
    """Parse CLI arguments and run the requested command."""
    parser = argparse.ArgumentParser(
        description="Isaac Lab CLI",
        prog="isaaclab" + (".bat" if is_windows() else ".sh"),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--install",
        nargs="?",
        const="all",
        help=(
            "Install Isaac Lab sub-packages and RL frameworks.\n"
            "Accepts a comma-separated list of sub-package names, one of the RL frameworks, or a special value.\n"
            "\n"
            "Sub-packages: assets, physx, contrib, mimic, newton, rl, tasks, teleop, visualizers.\n"
            "Use -i ovrtx to install the ovrtx dependency for isaaclab_ov.\n"
            "Visualizer selectors: visualizers[all|kit|newton|rerun|viser].\n"
            "RL frameworks: rl_games, rsl_rl, sb3, skrl, robomimic.\n"
            "\n"
            "Passing an RL framework name installs all sub-packages + that framework.\n"
            "\n"
            "Special values:\n"
            "- all  - Install all sub-packages + all RL frameworks (default).\n"
            "- none - Install only the core 'isaaclab' package.\n"
            "- <empty> (-i or --install without value) - Install all sub-packages + all RL frameworks.\n"
            "- quote visualizer selectors in bash, e.g. --install 'visualizers[rerun]'.\n"
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        action="store_true",
        help="Run pre-commit to format the code and check lints.",
    )
    parser.add_argument(
        "-p",
        "--python",
        nargs=argparse.REMAINDER,
        help="Run the python executable provided by Isaac Sim or virtual environment (if active).",
    )
    parser.add_argument(
        "-s",
        "--sim",
        nargs=argparse.REMAINDER,
        help="Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.",
    )
    parser.add_argument(
        "-t",
        "--test",
        nargs=argparse.REMAINDER,
        help="Run all python pytest tests.",
    )
    parser.add_argument(
        "-o",
        "--docker",
        nargs=argparse.REMAINDER,
        help="Run the docker container helper script (docker/container.sh).",
    )
    parser.add_argument(
        "-v",
        "--vscode",
        action="store_true",
        help="Generate the VSCode settings file from template.",
    )
    parser.add_argument(
        "-d",
        "--docs",
        action="store_true",
        help="Build the documentation from source using sphinx.",
    )
    parser.add_argument(
        "-n",
        "--new",
        nargs=argparse.REMAINDER,
        help="Create a new external project or internal task from template.",
    )
    parser.add_argument(
        "-c",
        "--conda",
        nargs="?",
        const="env_isaaclab",
        help="Create a new conda environment for Isaac Lab. Default name is 'env_isaaclab'.",
    )
    parser.add_argument(
        "-u",
        "--uv",
        nargs="?",
        const="env_isaaclab",
        help="Create a new uv environment for Isaac Lab. Default name is 'env_isaaclab'.",
    )

    args = parser.parse_args()

    if args.install:
        command_install(args.install)

    elif args.format:
        command_format()

    elif args.conda:
        command_setup_conda(args.conda)

    elif args.uv:
        command_setup_uv(args.uv)

    elif args.vscode:
        command_vscode_settings()

    elif args.docs:
        command_build_docs()

    elif args.docker is not None:
        command_run_docker(args.docker)

    elif args.python is not None:
        if args.python:
            run_python_command(args.python[0], args.python[1:])
        else:
            run_python_command("-i", [])

    elif args.sim is not None:
        command_run_isaacsim(args.sim)

    elif args.new is not None:
        command_new(args.new)

    elif args.test is not None:
        command_test(args.test)

    else:
        parser.print_help()
