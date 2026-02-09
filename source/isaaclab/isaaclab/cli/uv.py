# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import subprocess
import sys

from .utils import (
    ISAACLAB_ROOT,
    is_isaacsim_version_5_x,
    is_windows,
    run_command,
)


def setup_uv_env(env_name):
    """setup uv environment for Isaac Lab"""
    # Check if uv is installed.
    if not shutil.which("uv"):
        print("[ERROR] uv could not be found. Please install uv and try again.")
        print("[ERROR] uv can be installed here:")
        print("[ERROR] https://docs.astral.sh/uv/getting-started/installation/")
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
                print(f"[WARNING] _isaac_sim symlink not found at {ISAACLAB_ROOT}/_isaac_sim")
                print("\tThis warning can be ignored if you plan to install Isaac Sim via pip.")
                print(
                    "\tIf you are using a binary installation of Isaac Sim, please ensure "
                    + "the symlink is created before setting up the conda environment."
                )

    env_path = ISAACLAB_ROOT / env_name

    # Determine python version.
    if is_isaacsim_version_5_x():
        py_ver = "3.11"
    else:
        py_ver = "3.12"

    # Check if the environment exists.
    if not env_path.exists():
        print(f"[INFO] Creating uv environment named '{env_name}'...")
        run_command(["uv", "venv", "--clear", "--python", py_ver, str(env_path)])
    else:
        print(f"[INFO] uv environment '{env_name}' already exists.")

    # Install activation hooks.
    if is_windows():
        activate_script = env_path / "Scripts" / "activate.bat"
        # We can append to it.
        if activate_script.exists():
            # Add variables to environment during activation.
            with open(activate_script, "a") as f:
                f.write(f"\nset ISAACLAB_PATH={ISAACLAB_ROOT}\n")
                f.write("set RESOURCE_NAME=IsaacSim\n")
    else:
        activate_script = env_path / "bin" / "activate"
        if activate_script.exists():
            # Add variables to environment during activation.
            with open(activate_script, "a") as f:
                f.write(f"\nexport ISAACLAB_PATH={ISAACLAB_ROOT}\n")
                f.write("export RESOURCE_NAME=IsaacSim\n")
                if (ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.sh").exists():
                    f.write(f". {ISAACLAB_ROOT}/_isaac_sim/setup_conda_env.sh\n")

    print(f"[INFO] Created uv environment named '{env_name}'.\n")
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
