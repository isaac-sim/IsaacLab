# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from ..utils import (
    ISAACLAB_ROOT,
    determine_python_version,
    is_windows,
    print_debug,
    print_error,
    print_info,
    print_warning,
    run_command,
)


def _sanitized_conda_env():
    """
    Return an environment safe for invoking conda after Isaac Sim has added a bunch of env vars.
    Otherwise if there were different python version in the system vs IS python,
    it causes conda to fail with 'SRE mismatch error' due to incompatible python
    stdlib/runtime mix.
    """
    env = dict(os.environ)

    # Prevent mixed Python stdlib/runtime when the CLI is launched from Isaac Sim's bundled Python.
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(key, None)

    # Isaac Sim injects Kit shared libraries that can interfere with conda's Py runtime.
    env.pop("LD_LIBRARY_PATH", None)

    return env


def _patch_environment_yml(yml_path, python_version="3.12"):
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


def _get_conda_prefix(env_name):
    """Get the prefix of the conda environment."""
    # Use conda run to get sys.prefix
    try:
        env = _sanitized_conda_env()
        cmd = ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.prefix)"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def _write_conda_env_hooks(conda_prefix: Path):
    """Write conda activation/deactivation hooks for Isaac Lab environment variables."""
    activate_d = conda_prefix / "etc" / "conda" / "activate.d"
    deactivate_d = conda_prefix / "etc" / "conda" / "deactivate.d"
    activate_d.mkdir(parents=True, exist_ok=True)
    deactivate_d.mkdir(parents=True, exist_ok=True)

    activate_hook = activate_d / "setenv.sh"
    deactivate_hook = deactivate_d / "unsetenv.sh"
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.sh"

    activate_content = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash

        # for Isaac Lab
        : "${{_IL_PREV_PYTHONPATH:=${{PYTHONPATH-}}}}"
        : "${{_IL_PREV_LD_LIBRARY_PATH:=${{LD_LIBRARY_PATH-}}}}"
        export ISAACLAB_PATH="{ISAACLAB_ROOT}"
        alias isaaclab="{ISAACLAB_ROOT / "isaaclab.sh"}"

        # show icon if not running headless
        export RESOURCE_NAME="IsaacSim"

        # for Isaac Sim
        if [ -f "{isaacsim_setup_conda_env_script}" ]; then
            source "{isaacsim_setup_conda_env_script}"
        fi
        """
    )

    deactivate_content = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash

        # for Isaac Lab
        unalias isaaclab &>/dev/null
        unset ISAACLAB_PATH

        # restore paths
        if [ -v _IL_PREV_PYTHONPATH ]; then
            export PYTHONPATH="$_IL_PREV_PYTHONPATH"
            unset _IL_PREV_PYTHONPATH
        fi
        if [ -v _IL_PREV_LD_LIBRARY_PATH ]; then
            export LD_LIBRARY_PATH="$_IL_PREV_LD_LIBRARY_PATH"
            unset _IL_PREV_LD_LIBRARY_PATH
        fi

        # for Isaac Sim
        unset RESOURCE_NAME
        if [ -f "{isaacsim_setup_conda_env_script}" ]; then
            unset CARB_APP_PATH
            unset EXP_PATH
            unset ISAAC_PATH
        fi
        """
    )

    activate_hook.write_text(activate_content, encoding="utf-8")
    deactivate_hook.write_text(deactivate_content, encoding="utf-8")

    print_debug(f"Created activation hook: {activate_hook}")
    print_debug(f"Created deactivation hook: {deactivate_hook}")


def command_setup_conda(env_name):
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
    conda_env = _sanitized_conda_env()
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, env=conda_env)
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
        patched_content = _patch_environment_yml(env_yml, python_version)
        with open(temp_yml, "w") as f:
            f.write(patched_content)

        try:
            run_command(["conda", "env", "create", "-y", "--file", str(temp_yml), "-n", env_name], env=conda_env)
        finally:
            if temp_yml.exists():
                temp_yml.unlink()

    # Now configure activation scripts.
    conda_prefix = _get_conda_prefix(env_name)
    if not conda_prefix:
        print_error(f"Could not determine prefix for env {env_name}")
        return

    # Setup Isaac Lab and Isaac Sim environment variables through conda hooks.
    _write_conda_env_hooks(conda_prefix)

    if not is_windows():
        print_info("Added 'isaaclab' alias and environment hooks to conda activation scripts.")
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
