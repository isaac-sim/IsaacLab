# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import shutil
import subprocess
import sys
from pathlib import Path

from .utils import (
    ISAACLAB_ROOT,
    determine_python_version,
    is_windows,
    print_error,
    print_info,
    print_warning,
    run_command,
)


def patch_environment_yml(yml_path, python_version="3.12"):
    """
    Read environment.yml, return content with patched python version.
    """
    with open(yml_path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "python=3." in line:
            line = re.sub(r"python=3\.\d+(?:\.\d+)?", f"python={python_version}", line)
        new_lines.append(line)
    return "".join(new_lines)


def get_conda_prefix(env_name):
    """Get the prefix of the conda environment."""
    # Use conda run to get sys.prefix
    try:
        cmd = ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.prefix)"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def setup_conda_env(env_name):
    """Setup conda environment for Isaac Lab"""

    # Check if conda is installed.
    if not shutil.which("conda"):
        print_error("Conda could not be found. Please install conda and try again.")
        sys.exit(1)

    # Check if _isaac_sim symlink exists
    symlink_missing = not (ISAACLAB_ROOT / "_isaac_sim").exists()

    # Check if pip package isaacsim-rl is installed.
    pip_package_missing = True
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "isaacsim-rl"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pip_package_missing = False  # installed
    except subprocess.CalledProcessError:
        pip_package_missing = True  # not installed

    if symlink_missing and pip_package_missing:
        print_warning(f"_isaac_sim symlink not found at {ISAACLAB_ROOT}/_isaac_sim")
        print("\tThis warning can be ignored if you plan to install Isaac Sim via pip.")
        print(
            "\tIf you are using a binary installation of Isaac Sim, please ensure the "
            + "symlink is created before setting up the conda environment."
        )

    # Check if the environment exists.
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if env_name in result.stdout:
        print_info(f"Conda environment named '{env_name}' already exists.")
        env_exists = True
    else:
        print_info(f"Creating conda environment named '{env_name}'...")
        print_info(f"Installing dependencies from {ISAACLAB_ROOT}/environment.yml")
        env_exists = False

    if not env_exists:
        # Patch Python version if needed.
        env_yml = ISAACLAB_ROOT / "environment.yml"

        # Determine appropriate python version based on Isaac Sim version.
        python_version = determine_python_version()

        # Prepare patched yml.

        # Write a temp file.
        temp_yml = ISAACLAB_ROOT / "environment_temp.yml"
        patched_content = patch_environment_yml(env_yml, python_version)
        with open(temp_yml, "w") as f:
            f.write(patched_content)

        try:
            run_command(["conda", "env", "create", "-y", "--file", str(temp_yml), "-n", env_name])
        finally:
            if temp_yml.exists():
                temp_yml.unlink()

    # Now configure activation scripts.
    conda_prefix = get_conda_prefix(env_name)
    if not conda_prefix:
        print_error(f"Could not determine prefix for env {env_name}")
        return

    # Setup directories to load Isaac Sim variables.
    activate_d = conda_prefix / "etc" / "conda" / "activate.d"
    deactivate_d = conda_prefix / "etc" / "conda" / "deactivate.d"
    activate_d.mkdir(parents=True, exist_ok=True)
    deactivate_d.mkdir(parents=True, exist_ok=True)

    if not is_windows():
        print_info(f"Created conda environment named '{env_name}'.\n")
        print(f"\t\t1. To activate the environment, run:                conda activate {env_name}")
        print("\t\t2. To install Isaac Lab extensions, run:            isaaclab.sh -i")
        print("\t\t3. To perform formatting, run:                      isaaclab.sh -f")
        print("\t\t4. To deactivate the environment, run:              conda deactivate")
        print("\n")

    if is_windows():
        print_info(f"Created conda environment named '{env_name}'.\n")
        print(f"\t\t1. To activate the environment, run:                conda activate {env_name}")
        print("\t\t2. To install Isaac Lab extensions, run:            isaaclab.bat -i")
        print("\t\t3. To perform formatting, run:                      isaaclab.bat -f")
        print("\t\t4. To deactivate the environment, run:              conda deactivate")
        print("\n")
