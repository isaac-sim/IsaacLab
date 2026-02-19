# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import subprocess
import sys

from .utils import (
    ISAACLAB_ROOT,
    determine_python_version,
    is_windows,
    print_error,
    print_info,
    print_warning,
    run_command,
)


def command_setup_uv(env_name):
    """setup uv environment for Isaac Lab"""
    # Check if uv is installed.
    if not shutil.which("uv"):
        print_error("uv could not be found. Please install uv and try again.")
        print_error("uv can be installed here:")
        print_error("https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

    # Check if already in a uv environment - use precise pattern matching.
    # (In Python we check environments differently or assume env_name is new).

    # Check if _isaac_sim symlink exists and isaacsim-rl is not installed via pip.
    if not (ISAACLAB_ROOT / "_isaac_sim").is_symlink():
        # Check pip list for isaacsim-rl - simple subprocess fallback.
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "show", "isaacsim-rl"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Installed.
        except Exception:
            # Not installed, symlink missing.
            if not (ISAACLAB_ROOT / "_isaac_sim").exists():
                print_warning(f"_isaac_sim symlink not found at {ISAACLAB_ROOT}/_isaac_sim")
                print("\tThis warning can be ignored if you plan to install Isaac Sim via pip.")
                print(
                    "\tIf you are using a binary installation of Isaac Sim, please ensure "
                    + "the symlink is created before setting up the conda environment."
                )

    env_path = ISAACLAB_ROOT / env_name

    # Determine appropriate python version based on Isaac Sim version.
    py_ver = determine_python_version()

    # Check if the environment exists.
    if not env_path.exists():
        print_info(f"Creating uv environment named '{env_name}'...")
        run_command(["uv", "venv", "--clear", "--python", py_ver, str(env_path)])
    else:
        print_info(f"uv environment '{env_name}' already exists.")

    print_info(f"Created uv environment named '{env_name}'.\n")
    if is_windows():
        print(f"\t\t1. To activate the environment, run:                {env_name}\\Scripts\\activate")
        print("\t\t2. To install Isaac Lab extensions, run:            isaaclab.bat -i")
        print("\t\t3. To perform formatting, run:                      isaaclab.bat -f")
        print(f"\t\t4. To deactivate the environment, run:              {env_name}\\Scripts\\deactivate")
    else:
        print(f"\t\t1. To activate the environment, run:                {env_name}/Scripts/activate")
        print("\t\t2. To install Isaac Lab extensions, run:            ./isaaclab.sh -i")
        print("\t\t3. To perform formatting, run:                      ./isaaclab.sh -f")
        print(f"\t\t4. To deactivate the environment, run:              {env_name}/Scripts/deactivate")
    print("\n")
