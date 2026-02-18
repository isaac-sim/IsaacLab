# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import subprocess

from .conda import setup_conda_env
from .format import format_code
from .install import install
from .utils import (
    ISAACLAB_ROOT,
    build_docs,
    extract_isaacsim_exe,
    extract_python_exe,
    is_arm,
    is_windows,
    print_info,
    run_command,
    run_docker_helper,
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
        # Install vscode update unless we're in docker.
        if not (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")):
            update_vscode_settings()

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
        # Python execution.
        # Args.python is a list of remaining args.
        python_exe = extract_python_exe()
        # The first arg might be the script or -m, pass all.
        cmd = [python_exe] + args.python
        env = os.environ.copy()
        if is_arm():
            env["RESOURCE_NAME"] = env.get("RESOURCE_NAME", "IsaacSim")

        # We use subprocess calling python.
        subprocess.run(cmd, env=env)

    elif args.sim is not None:
        # Sim execution.
        sim_exe = extract_isaacsim_exe()

        print_info(f"Running isaac-sim from: {sim_exe}")

        cmd = sim_exe

        # Add ext folder.
        cmd.append("--ext-folder")
        cmd.append(str(ISAACLAB_ROOT / "source"))
        cmd.extend(args.sim)

        run_command(cmd, check=False)

    elif args.new is not None:
        print_info("Installing template dependencies...")
        python_exe = extract_python_exe()
        reqs = ISAACLAB_ROOT / "tools" / "template" / "requirements.txt"
        run_command([python_exe, "-m", "pip", "install", "-q", "-r", str(reqs)])

        print_info("Running template generator...")
        cli_script = ISAACLAB_ROOT / "tools" / "template" / "cli.py"
        run_python_command(cli_script, args.new)

    elif args.test is not None:
        python_exe = extract_python_exe()

        # Make sure pytest is installed.
        try:
            # Check if pytest is available as a module.
            run_python_command("pip", ["show", "pytest"], is_module=True, check=True)
        except subprocess.CalledProcessError:
            if is_windows():
                print_info("pytest not found, please run install first with 'isaaclab.bat -i'...")
            else:
                print_info("pytest not found, please run install first with './isaaclab.sh -i'...")
            return

        cmd = [
            python_exe,
            "-m",
            "pytest",
            str(ISAACLAB_ROOT / "tools"),
        ] + args.test

        run_command(cmd, check=False)

    else:
        parser.print_help()
