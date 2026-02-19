# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from .conda import setup_conda_env
from .format import format_code
from .install import install
from .utils import (
    ISAACLAB_ROOT,
    build_docs,
    is_windows,
    print_info,
    run_docker_helper,
    run_isaacsim,
    run_python_command,
    update_vscode_settings,
)
from .uv import setup_uv_env


def cli():
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
        help="Install the extensions inside Isaac Lab and learning frameworks as extra dependencies.\n"
        + "Can be used in any active conda/uv environment. Default is 'all'.",
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
        install(args.install)

    elif args.format:
        format_code()

    elif args.conda:
        setup_conda_env(args.conda)

    elif args.uv:
        setup_uv_env(args.uv)

    elif args.vscode:
        update_vscode_settings()

    elif args.docs:
        build_docs()

    elif args.docker is not None:
        run_docker_helper(args.docker)

    elif args.python is not None:
        if args.python:
            run_python_command(args.python[0], args.python[1:])
        else:
            run_python_command("-i", [])

    elif args.sim is not None:
        run_isaacsim(args.sim)

    elif args.new is not None:
        print_info("Installing template dependencies...")
        reqs = ISAACLAB_ROOT / "tools" / "template" / "requirements.txt"
        run_python_command("-m", ["pip", "install", "-q", "-r", str(reqs)])

        print_info("Running template generator...")
        cli_script = ISAACLAB_ROOT / "tools" / "template" / "cli.py"
        run_python_command(cli_script, args.new)

    elif args.test is not None:
        run_python_command("-m", ["pytest", str(ISAACLAB_ROOT / "tools")] + args.test)

    else:
        parser.print_help()
