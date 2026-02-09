# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import subprocess
import sys
from pathlib import Path

from .utils import (
    ISAACLAB_ROOT,
    extract_isaacsim_path,
    is_isaacsim_version_5_x,
    is_windows,
    print_error,
    print_info,
    print_warning,
    run_command,
)


def patch_environment_yml(yml_path, use_python_311):
    """
    Read environment.yml, return content with patched python version.
    """
    with open(yml_path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "python=3.12" in line and use_python_311:
            new_lines.append(line.replace("python=3.12", "python=3.11"))
        else:
            new_lines.append(line)
    return "".join(new_lines)


def get_conda_prefix(env_name):
    """Get the prefix of the conda environment."""
    # Use conda run to get sys.prefix
    try:
        cmd = [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            "import sys; print(sys.prefix)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def setup_conda_env(env_name):
    """setup anaconda environment for Isaac Lab"""
    # Check if conda is installed.
    if not shutil.which("conda"):
        print_error("Conda could not be found. Please install conda and try again.")
        sys.exit(1)

    # Check if _isaac_sim symlink exists and isaacsim-rl is not installed via pip.
    check_symlink = not (ISAACLAB_ROOT / "_isaac_sim").exists()
    check_pip = True
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "isaacsim-rl"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        check_pip = False  # installed
    except subprocess.CalledProcessError:
        check_pip = True  # not installed

    if check_symlink and check_pip:
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
    else:
        print_info(f"Creating conda environment named '{env_name}'...")
        print_info(f"Installing dependencies from {ISAACLAB_ROOT}/environment.yml")

        # Patch Python version if needed, but back up first.
        env_yml = ISAACLAB_ROOT / "environment.yml"

        # Prepare patched yml.
        use_311 = is_isaacsim_version_5_x()
        if use_311:
            print_info("Detected Isaac Sim 5.X -> using python=3.11")
        else:
            print_info("Isaac Sim 6.0+ detected, installing python=3.12")

        # Write a temp file.
        temp_yml = ISAACLAB_ROOT / "environment_temp.yml"
        patched_content = patch_environment_yml(env_yml, use_311)
        with open(temp_yml, "w") as f:
            f.write(patched_content)

        try:
            run_command(
                [
                    "conda",
                    "env",
                    "create",
                    "-y",
                    "--file",
                    str(temp_yml),
                    "-n",
                    env_name,
                ]
            )
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

    # Check if we have _isaac_sim directory -> if so that means binaries were installed.
    # We need to setup conda variables to load the binaries.
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.sh"

    if not is_windows():
        setenv_sh = activate_d / "setenv.sh"
        unsetenv_sh = deactivate_d / "unsetenv.sh"

        with open(setenv_sh, "w") as f:
            f.write("#!/usr/bin/env bash\n\n")
            f.write("# For Isaac Lab\n")
            f.write(f"export ISAACLAB_PATH={ISAACLAB_ROOT}\n")
            f.write("# Show icon if not running headless\n")
            f.write("export RESOURCE_NAME=IsaacSim\n\n")

            if isaacsim_setup_conda_env_script.exists():
                f.write("# For Isaac Sim\n")
                f.write(f"source {isaacsim_setup_conda_env_script}\n")

        with open(unsetenv_sh, "w") as f:
            f.write("#!/usr/bin/env bash\n\n")
            f.write("# For Isaac Lab\n")
            f.write("unset ISAACLAB_PATH\n\n")
            f.write("# For Isaac Sim\n")
            f.write("unset RESOURCE_NAME\n\n")
            if isaacsim_setup_conda_env_script.exists():
                f.write("# For Isaac Sim\n")
                f.write("unset CARB_APP_PATH\n")
                f.write("unset EXP_PATH\n")
                f.write("unset ISAAC_PATH\n")

        print_info(f"Created conda environment named '{env_name}'.\n")
        print(f"\t\t1. To activate the environment, run:                conda activate {env_name}")
        print("\t\t2. To install Isaac Lab extensions, run:            isaaclab.sh -i")
        print("\t\t3. To perform formatting, run:                      isaaclab.sh -f")
        print("\t\t4. To deactivate the environment, run:              conda deactivate")
        print("\n")

    # Windows
    if is_windows():
        setenv_bat = activate_d / "env_vars.bat"
        unsetenv_bat = deactivate_d / "unsetenv_vars.bat"

        isaac_path = extract_isaacsim_path()

        with open(setenv_bat, "w") as f:
            f.write("@echo off\n")
            f.write('set "RESOURCE_NAME=IsaacSim"\n')
            if isaac_path and isaac_path.exists():
                f.write(f"set CARB_APP_PATH={isaac_path}\\kit\n")
                f.write(f"set EXP_PATH={isaac_path}\\apps\n")
                f.write(f"set ISAAC_PATH={isaac_path}\n")
                f.write(f"set PYTHONPATH=%PYTHONPATH%;{isaac_path}\\site\n")

        with open(unsetenv_bat, "w") as f:
            f.write("@echo off\n")
            f.write('set "RESOURCE_NAME="\n')
            f.write('set "CARB_APP_PATH="\n')
            f.write('set "EXP_PATH="\n')
            f.write('set "ISAAC_PATH="\n')

        print_info(f"Created conda environment named '{env_name}'.\n")
        print(f"\t\t1. To activate the environment, run:                conda activate {env_name}")
        print("\t\t2. To install Isaac Lab extensions, run:            isaaclab.bat -i")
        print("\t\t3. To perform formatting, run:                      isaaclab.bat -f")
        print("\t\t4. To deactivate the environment, run:              conda deactivate")
        print("\n")
